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

import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from modelopt.torch.utils.image_processor import MllamaImageProcessor

SPECULATIVE_MODEL_LIST = ["Eagle", "Medusa"]


def is_speculative(hf_config):
    for name in SPECULATIVE_MODEL_LIST:
        if name in hf_config.architectures[0]:
            return True
    return False


def get_mode_type_from_engine_dir(engine_dir_str):
    # Split the path by '/' and get the last part
    last_part = os.path.basename(engine_dir_str)

    # Split the last part by '_' and get the first segment
    model_type = last_part.split("_")[0]

    return model_type


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs):
    print(f"Initializing tokenizer from {ckpt_path}")

    if "vila" in ckpt_path.lower():
        ckpt_path += "/llm"

    if ckpt_path.endswith(".yaml"):
        # Model Optimizer modification
        # For Nemo models, tokenizer is instantiated based on its config
        from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

        tokenizer = get_nemo_tokenizer(ckpt_path)

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code, **kwargs
        )

        if "qwen" in type(tokenizer).__name__.lower():
            # qwen use token id 151643 as pad and eos tokens
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

        # can't set attribute 'pad_token' for "<unk>"
        # We skip this step for Nemo models
        if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def get_processor(ckpt_path, model_type, device=None, trust_remote_code=False):
    """
    Returns a :class:`modelopt.torch.utils.image_processor.MllamaImageProcessor` object.
    """
    if model_type == "whisper":
        processor = AutoProcessor.from_pretrained(
            ckpt_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
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
            trust_remote_code=trust_remote_code,
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


def get_model(ckpt_path, device="cuda", gpu_mem_percentage=0.8, trust_remote_code=False):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}

    if "vila" in ckpt_path.lower():
        sys.path.append(os.path.join(ckpt_path, "..", "VILA"))
        from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
        from transformers import AutoModel

        hf_vila = AutoModel.from_pretrained(
            ckpt_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = hf_vila.llm
    else:
        hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)

        if is_speculative(hf_config):
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif hf_config.model_type == "llava":
            from transformers import LlavaForConditionalGeneration

            hf_llava = LlavaForConditionalGeneration.from_pretrained(
                ckpt_path, device_map=device_map, **model_kwargs
            )
            model = hf_llava.language_model
        elif hf_config.model_type == "t5":
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_path, device_map=device_map, **model_kwargs
            )
        elif hf_config.model_type == "bart":
            from transformers import AutoModelForSeq2SeqLM

            # device_map "auto" and "cuda" triggers error regarding meta tensor from safetensors
            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_path, device_map=None, **model_kwargs
            ).to(device)
        elif hf_config.model_type == "whisper":
            from transformers import WhisperForConditionalGeneration

            model = WhisperForConditionalGeneration.from_pretrained(
                ckpt_path, device_map=device_map, **model_kwargs
            )
        elif hf_config.model_type == "glm":
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_path, device_map="cuda", **model_kwargs, trust_remote_code=trust_remote_code
            )
        elif hf_config.model_type == "mllama":
            from transformers import MllamaForConditionalGeneration

            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif hf_config.model_type == "llama4":
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
                trust_remote_code=trust_remote_code,
            )

        else:
            from accelerate import infer_auto_device_map, init_empty_weights
            from accelerate.utils import get_max_memory

            with init_empty_weights():
                # When computing the device_map, assuming half precision by default,
                # unless specified by the hf_config.
                torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
                model = AutoModelForCausalLM.from_config(
                    hf_config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
                )

            max_memory = get_max_memory()
            inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

            on_cpu = "cpu" in inferred_device_map.values()

            if on_cpu:
                for device in max_memory.keys():
                    if isinstance(device, int):
                        max_memory[device] *= gpu_mem_percentage

                print(
                    "Model does not fit to the GPU mem. "
                    f"We apply the following memmory limit for calibration: \n{max_memory}\n"
                    "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                    "reduce the calibration `batch_size` manually."
                )
                model_kwargs["max_memory"] = max_memory

            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
                trust_remote_code=trust_remote_code,
            )
    model.eval()
    if device == "cuda":
        if not is_model_on_gpu(model):
            print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def is_enc_dec(model_type) -> bool:
    """Return if the model is a encoder-decoder model."""
    return model_type in ["t5", "bart", "whisper"]
