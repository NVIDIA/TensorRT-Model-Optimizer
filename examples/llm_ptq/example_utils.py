# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_max_memory
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from modelopt.torch.utils.image_processor import MllamaImageProcessor

SPECULATIVE_MODEL_LIST = ["Eagle", "Medusa"]


def is_speculative(hf_config):
    """Check if the model architecture is a speculative model."""
    return hf_config.architectures and any(
        name in hf_config.architectures[0] for name in SPECULATIVE_MODEL_LIST
    )


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs):
    print(f"Initializing tokenizer from {ckpt_path}")

    if "vila" in ckpt_path.lower():
        ckpt_path += "/llm"

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # can't set attribute 'pad_token' for "<unk>"
    # We skip this step for Nemo models
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def get_processor(
    ckpt_path, model_type, device=None, trust_remote_code=False, attn_implementation=None
):
    """
    Returns a :class:`modelopt.torch.utils.image_processor.MllamaImageProcessor` object.
    """
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if model_type == "whisper":
        processor = AutoProcessor.from_pretrained(
            ckpt_path,
            padding_side="left",
            **model_kwargs,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        assert processor.tokenizer.pad_token is not None, (
            f"Pad token for {ckpt_path} cannot be set!"
        )

        return processor
    elif model_type == "mllama":
        processor = AutoProcessor.from_pretrained(
            ckpt_path,
            padding_side="left",
            **model_kwargs,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        assert processor.tokenizer.pad_token is not None, (
            f"Pad token for {ckpt_path} cannot be set!"
        )

        return MllamaImageProcessor(processor, device)


def get_dtype(dtype):
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    return dtype


def get_lora_model(
    ckpt_path: str,
    device_map="cuda",
):
    """
    Loads a QLoRA model that has been trained using modelopt trainer.
    """
    # Load model with adapters
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map)

    # Restore modelopt state
    modelopt_state = torch.load(f"{ckpt_path}/modelopt_state.pth", weights_only=False)
    restore_from_modelopt_state(model, modelopt_state)

    # Load compressed weights
    state_dict = load_file(f"{ckpt_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    return model


def get_model(
    ckpt_path,
    device="cuda",
    gpu_mem_percentage=0.8,
    trust_remote_code=False,
    use_seq_device_map=False,
    attn_implementation=None,
    is_modelopt_qlora=False,
):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    if is_modelopt_qlora:
        model = get_lora_model(ckpt_path, device_map)
        return model

    config_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
    if attn_implementation is not None:
        config_kwargs["attn_implementation"] = attn_implementation

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = config_kwargs.copy()
    # Don't set torch_dtype for VILA models as they handle it explicitly in their builder
    if "vila" not in ckpt_path.lower():
        model_kwargs.setdefault("torch_dtype", "auto")

    if "vila" in ckpt_path.lower():
        sys.path.append(os.path.join(ckpt_path, "..", "VILA"))
        from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa: F401
        from transformers import AutoModel

        hf_vila = AutoModel.from_pretrained(
            ckpt_path,
            device_map=device_map,
            **model_kwargs,
        )
        model = hf_vila.llm
    else:
        hf_config = AutoConfig.from_pretrained(
            ckpt_path,
            **config_kwargs,
        )

        if use_seq_device_map:
            device_map = "sequential"
            # If we use sequential, set max_memory limit to ensure that the model does not occupy the full GPU
            max_memory = get_max_memory()
            max_memory = {key: value * gpu_mem_percentage for key, value in max_memory.items()}
            model_kwargs["max_memory"] = max_memory

        if hf_config.model_type == "bart":
            # device_map "auto" and "cuda" triggers error regarding meta tensor from safetensors
            device_map = None

        if is_speculative(hf_config):
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
            )
        else:
            architecture = hf_config.architectures[0]

            if not hasattr(transformers, architecture):
                warnings.warn(
                    f"Architecture {architecture} not found in transformers: {transformers.__version__}. "
                    "Falling back to AutoModelForCausalLM."
                )
                assert trust_remote_code, (
                    "Please set trust_remote_code to True if you want to use this architecture"
                )

                auto_model_module = AutoModelForCausalLM
                from_config = auto_model_module.from_config
            else:
                auto_model_module = getattr(transformers, architecture)
                from_config = auto_model_module._from_config

            with init_empty_weights():
                # When computing the device_map, assuming half precision by default,
                # unless specified by the hf_config.
                torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
                model_kwargs2 = model_kwargs.copy()
                if auto_model_module != AutoModelForCausalLM:
                    model_kwargs2.pop("trust_remote_code", None)
                model_kwargs2["torch_dtype"] = torch_dtype
                model_kwargs2.pop("max_memory", None)
                model = from_config(hf_config, **model_kwargs2)

            max_memory = get_max_memory()
            inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

            on_cpu = "cpu" in inferred_device_map.values()

            if on_cpu:
                for _device in max_memory:
                    if isinstance(_device, int):
                        max_memory[_device] *= gpu_mem_percentage

                print(
                    "Model does not fit to the GPU mem. "
                    f"We apply the following memory limit for calibration: \n{max_memory}\n"
                    "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                    "reduce the calibration `batch_size` manually."
                )
                model_kwargs["max_memory"] = max_memory

            model = auto_model_module.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
            )
    model.eval()
    if device == "cuda" and not is_model_on_gpu(model):
        print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def is_enc_dec(model_type) -> bool:
    """Return if the model is a encoder-decoder model."""
    return model_type in ["t5", "bart", "whisper"]


def apply_kv_cache_quant(quant_cfg: dict[str, Any], kv_cache_quant_cfg: dict[str, Any]):
    """Apply quantization to the kv cache of the model."""
    # Update KV cache related bmm quantizers
    # If quant_cfg["quant_cfg"] is None, it corresponds to only kv cache quantization case
    quant_cfg["quant_cfg"] = quant_cfg.get("quant_cfg", {"default": {"enable": False}})
    quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)

    # Set default algorithm for kv cache quantization if not provided.
    if not quant_cfg.get("algorithm"):
        quant_cfg["algorithm"] = "max"

    return quant_cfg


def _resolve_model_path(model_name_or_path: str, trust_remote_code: bool = False) -> str:
    """Resolve a model name or path to a local directory path.

    If the input is already a local directory, returns it as-is.
    If the input is a HuggingFace model ID, attempts to resolve it to the local cache path.

    Args:
        model_name_or_path: Either a local directory path or HuggingFace model ID
        trust_remote_code: Whether to trust remote code when loading the model

    Returns:
        Local directory path to the model files
    """
    # If it's already a local directory, return as-is
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Try to resolve HuggingFace model ID to local cache path
    try:
        # First try to load the config to trigger caching
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        # The config object should have the local path information
        # Try different ways to get the cached path
        if hasattr(config, "_name_or_path") and os.path.isdir(config._name_or_path):
            return config._name_or_path

        # Alternative: use snapshot_download if available
        if snapshot_download is not None:
            try:
                local_path = snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["*.py", "*.json"],  # Only download Python files and config
                )
                return local_path
            except Exception as e:
                print(f"Warning: Could not download model files using snapshot_download: {e}")

        # Fallback: try to find in HuggingFace cache
        from transformers.utils import TRANSFORMERS_CACHE

        # Look for the model in the cache directory
        cache_pattern = os.path.join(TRANSFORMERS_CACHE, "models--*")
        cache_dirs = glob.glob(cache_pattern)

        # Convert model name to cache directory format
        model_cache_name = model_name_or_path.replace("/", "--")
        for cache_dir in cache_dirs:
            if model_cache_name in cache_dir:
                # Look for the snapshots directory
                snapshots_dir = os.path.join(cache_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get the latest snapshot
                    snapshot_dirs = [
                        d
                        for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if snapshot_dirs:
                        latest_snapshot = max(snapshot_dirs)  # Use lexicographically latest
                        snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                        return snapshot_path

    except Exception as e:
        print(f"Warning: Could not resolve model path for {model_name_or_path}: {e}")

    # If all else fails, return the original path
    # This will cause the copy function to skip with a warning
    return model_name_or_path


def copy_custom_model_files(source_path: str, export_path: str, trust_remote_code: bool = False):
    """Copy custom model files (configuration_*.py, modeling_*.py, *.json, etc.) from source to export directory.

    This function copies custom Python files and JSON configuration files that are needed for
    models with custom code. It excludes config.json and model.safetensors.index.json as these
    are typically handled separately by the model export process.

    Args:
        source_path: Path to the original model directory or HuggingFace model ID
        export_path: Path to the exported model directory
        trust_remote_code: Whether trust_remote_code was used (only copy files if True)
    """
    if not trust_remote_code:
        return

    # Resolve the source path (handles both local paths and HF model IDs)
    resolved_source_path = _resolve_model_path(source_path, trust_remote_code)

    source_dir = Path(resolved_source_path)
    export_dir = Path(export_path)

    if not source_dir.exists():
        if resolved_source_path != source_path:
            print(
                f"Warning: Could not find local cache for HuggingFace model '{source_path}' "
                f"(resolved to '{resolved_source_path}')"
            )
        else:
            print(f"Warning: Source directory '{source_path}' does not exist")
        return

    if not export_dir.exists():
        print(f"Warning: Export directory {export_path} does not exist")
        return

    # Common patterns for custom model files that need to be copied
    custom_file_patterns = [
        "configuration_*.py",
        "modeling_*.py",
        "tokenization_*.py",
        "processing_*.py",
        "image_processing_*.py",
        "feature_extraction_*.py",
        "*.json",
    ]

    copied_files = []
    for pattern in custom_file_patterns:
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                # Skip config.json and model.safetensors.index.json as they're handled separately
                if file_path.name in ["config.json", "model.safetensors.index.json"]:
                    continue
                dest_path = export_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    print(f"Copied custom model file: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to copy {file_path.name}: {e}")

    if copied_files:
        print(f"Successfully copied {len(copied_files)} custom model files to {export_path}")
    else:
        print("No custom model files found to copy")
