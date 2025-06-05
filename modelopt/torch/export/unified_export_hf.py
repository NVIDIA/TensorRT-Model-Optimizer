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

"""Code that export quantized Hugging Face models for deployment."""

import collections.abc
import json
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.quantization import set_quantizer_by_cfg_context
from modelopt.torch.quantization.nn import SequentialQuantizer

from .convert_hf_config import convert_hf_quant_config_format
from .layer_utils import (
    get_expert_linear_names,
    get_experts_list,
    is_layernorm,
    is_moe,
    is_quantlinear,
    set_amax_for_uncalibrated_experts,
)
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_NVFP4,
    KV_CACHE_NVFP4_AFFINE,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_W4A8_AWQ,
)
from .quant_utils import (
    fuse_prequant_layernorm,
    get_activation_scaling_factor,
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    postprocess_state_dict,
    preprocess_linear_fusion,
    quantize_llama4_experts_for_hf_export,
    to_quantized_weight,
)

__all__ = ["export_hf_checkpoint"]

SPECULATIVE_DECODING_MODULE_NAMES = ["medusa_heads", "eagle_module", "drafter"]


def _is_enabled_quantizer(quantizer):
    if hasattr(quantizer, "is_enabled") and quantizer.is_enabled:
        return True

    if isinstance(quantizer, SequentialQuantizer):
        return any(q.is_enabled for q in quantizer)

    return False


def requantize_resmooth_fused_llm_layers(model: torch.nn.Module):
    """Group modules that take the same input and register shared parameters in module."""
    # TODO: Handle DBRX MoE
    input_to_linear = defaultdict(list)
    output_to_layernorm = defaultdict(None)
    quantization_format = get_quantization_format(model)

    def _input_hook(module, input, output):
        """Update dictionary with list of all modules that share the same input."""
        # TODO: Handle DBRX MoE case
        input_to_linear[input[0]].append(module)

    def _output_hook(module, input, output):
        """Update dictionary with mapping of layernorms and their outputs."""
        output_to_layernorm[output] = module

    handles = []
    model_type = type(model).__name__.lower()

    for name, module in model.named_modules():
        # For MoE models update pre_quant_scale to average pre_quant_scale amongst experts
        if is_moe(module) and ("awq" in quantization_format):
            # update_experts_avg_prequant_scale(module)
            grouped_experts = get_experts_list(module, model_type)
            for modules in grouped_experts:
                preprocess_linear_fusion(modules, resmooth_only=True)

        # Attach hook to layernorm modules that need to be fused
        if is_layernorm(module):
            module.name = name
            handle = module.register_forward_hook(_output_hook)
            handles.append(handle)
        elif is_quantlinear(module) and (
            _is_enabled_quantizer(module.input_quantizer)
            or _is_enabled_quantizer(module.weight_quantizer)
        ):
            module.name = name
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    with torch.no_grad():
        fake_input = torch.ones([1, 2], dtype=torch.long).to(model.device)
        decoder_fake_input = fake_input
        if model_type.startswith("whisper"):
            # For Whisper models, we need to pass a fake input with the specific sequence length
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model.name_or_path)
            fake_input = torch.ones(
                [1, model.config.num_mel_bins, feature_extractor.nb_max_frames], dtype=model.dtype
            ).to(model.device)

        # Run forward pass so that all modules sharing the same input are collected using forward hook.

        with set_quantizer_by_cfg_context(model, {"*": {"enable": False}}):
            if getattr(model.config, "is_encoder_decoder", False):
                # For encoder-decoder models, we need to pass both the encoder and decoder input ids
                model(fake_input, decoder_input_ids=decoder_fake_input)
            else:
                model(fake_input)

        for handle in handles:
            handle.remove()

    for tensor, modules in input_to_linear.items():
        quantization_format = get_quantization_format(modules[0])
        if len(modules) > 1 and quantization_format not in [
            QUANTIZATION_FP8,
            QUANTIZATION_NONE,
            QUANTIZATION_FP8_PB_REAL,
        ]:
            # Fuse modules that have the same input
            preprocess_linear_fusion(modules)

        # Fuse layernorms
        if (
            quantization_format is not QUANTIZATION_NONE
            and "awq" in quantization_format
            and tensor in output_to_layernorm
        ):
            # Pre quant scale of modules is already updated to avg_pre_quant_scale
            fuse_prequant_layernorm(output_to_layernorm[tensor], modules)


def _export_hf_checkpoint(
    model: nn.Module, dtype: torch.dtype | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
    """
    if dtype is None:
        dtype = model.config.torch_dtype
    elif dtype != model.config.torch_dtype:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )

    # Create a model layer pool
    # If `model.model` exists use that, otherwise use `model` itself, e.g., Nemotron-H
    root = getattr(model, "model", model)
    # If that has a `.layers`, use it, otherwise fall back to the object itself
    root = getattr(root, "layers", root)
    layer_pool = {f"model.layers.{name}": sub_module for name, sub_module in root.named_modules()}

    # Handle input quantizers of experts that are not calibrated
    for name, sub_module in model.named_modules():
        if is_moe(sub_module) and hasattr(sub_module, "experts"):
            expert_linear_names = get_expert_linear_names(sub_module)
            for linear_name in expert_linear_names:
                # Handle DBRX experts specifically
                if "QuantDbrxExperts" in type(sub_module.experts).__name__:
                    # For DBRX, experts are in sub_module.experts.mlp and linear layers are ModuleLists
                    experts_mlp = sub_module.experts.mlp
                    if hasattr(experts_mlp, linear_name):
                        linear_modulelist = getattr(experts_mlp, linear_name)
                        if hasattr(linear_modulelist, "__iter__"):
                            set_amax_for_uncalibrated_experts(list(linear_modulelist))
                elif isinstance(sub_module.experts, collections.abc.Iterable):
                    # For other MoE models (like Mixtral) with iterable experts
                    try:
                        set_amax_for_uncalibrated_experts(
                            [getattr(expert, linear_name) for expert in sub_module.experts]
                        )
                    except AttributeError as e:
                        # Provide more helpful debugging information
                        expert_types = [type(expert).__name__ for expert in sub_module.experts]
                        raise AttributeError(
                            f"Failed to access attribute '{linear_name}' on experts. "
                            f"MoE module type: {type(sub_module).__name__}, "
                            f"Expert types: {expert_types}, "
                            f"Expected linear names: {expert_linear_names}. "
                            f"This suggests the get_expert_linear_names function may need "
                            f"to be updated for this model architecture. "
                            f"Original error: {e}"
                        ) from e
                else:
                    # Unsupported MoE model structure
                    raise NotImplementedError(
                        f"MoE model with experts type '{type(sub_module.experts).__name__}' is not supported in export."
                        f"Please file an issue or add support for this model architecture."
                    )

    # NOTE: Speculative decoding models have extra modules that may be quantized
    # Need to add these modules to the layer_pool
    for key in SPECULATIVE_DECODING_MODULE_NAMES:
        if hasattr(model, key):
            for name, sub_module in getattr(model, key).named_modules():
                layer_pool.update({f"{key}.{name}": sub_module})

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    # Remove all hooks from the model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except ImportError:
        warnings.warn("accelerate is not installed, hooks will not be removed")

    quant_config = get_quant_config(layer_pool)

    kv_cache_max_bound = 0
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    cache_bound_mapping = {
        KV_CACHE_NVFP4: 6 * 448,
        KV_CACHE_NVFP4_AFFINE: 6 * 448,
        KV_CACHE_FP8: 448,
    }

    # Only update kv_cache_max_bound if a quantization is applied.
    if kv_cache_format != QUANTIZATION_NONE:
        kv_cache_max_bound = cache_bound_mapping.get(kv_cache_format)

    # Track if any layers are quantized to properly set exclude_modules
    has_quantized_layers = False

    for name, sub_module in layer_pool.items():
        if is_quantlinear(sub_module):
            quantization_format = get_quantization_format(sub_module)
            block_size = get_weight_block_size(sub_module)

            # Track if any layer is quantized
            if quantization_format != QUANTIZATION_NONE:
                has_quantized_layers = True

            if quantization_format == QUANTIZATION_FP8:
                # Convert amax to float32
                sub_module.weight_quantizer._amax = sub_module.weight_quantizer._amax.to(
                    torch.float32
                )

                if sub_module.weight_quantizer._amax.dim() == 1:
                    weight_scaling_factor = torch.tensor(
                        sub_module.weight_quantizer.amax.item()
                        / sub_module.weight_quantizer.maxbound
                    )
                else:
                    # Per-channel amax
                    weight_scaling_factor = torch.tensor(
                        sub_module.weight_quantizer.amax / sub_module.weight_quantizer.maxbound
                    )

                sub_module.register_buffer(
                    "weight_scale",
                    weight_scaling_factor,
                )

                if hasattr(sub_module.input_quantizer, "_amax"):
                    sub_module.input_quantizer._amax = sub_module.input_quantizer._amax.to(
                        torch.float32
                    )

                    sub_module.register_buffer(
                        "input_scale",
                        get_activation_scaling_factor(sub_module).squeeze(),
                    )

                if hasattr(sub_module.output_quantizer, "_amax"):
                    sub_module.output_quantizer._amax = sub_module.output_quantizer._amax.to(
                        torch.float32
                    )

            if quantization_format in [
                QUANTIZATION_NVFP4_AWQ,
                QUANTIZATION_NVFP4,
                QUANTIZATION_W4A8_AWQ,
            ]:
                # Register weight_scale_2
                sub_module.register_buffer(
                    "weight_scale_2",
                    get_weight_scaling_factor_2(sub_module).squeeze(),
                )

            if quantization_format not in [QUANTIZATION_FP8, QUANTIZATION_NONE]:
                # Register weight_scale and input_scale
                if quantization_format == QUANTIZATION_FP8_PB_REAL:
                    sub_module.register_buffer(
                        "weight_scale",
                        sub_module.weight_quantizer._scale.to(torch.float32),
                    )
                    del sub_module.weight_quantizer._scale
                else:
                    sub_module.register_buffer(
                        "weight_scale", get_weight_scaling_factor(sub_module)
                    )
                    # Remove size-1 dimensions for blocked fp8 scales
                    sub_module.weight_scale.squeeze()

                if (
                    hasattr(sub_module, "input_quantizer")
                    and "disabled" not in repr(sub_module.input_quantizer)
                    and sub_module.input_quantizer.amax is not None
                ):
                    sub_module.register_buffer(
                        "input_scale", get_activation_scaling_factor(sub_module).squeeze()
                    )

            # Check if quantization format is None, to support auto_quant
            if quantization_format != QUANTIZATION_NONE:
                quantized_weight = to_quantized_weight(
                    sub_module.weight.to(dtype),
                    sub_module.weight_scale,
                    quantization_format,
                    sub_module.weight_scale_2 if hasattr(sub_module, "weight_scale_2") else None,
                    block_size,
                )
                sub_module.weight = nn.Parameter(quantized_weight, requires_grad=False)
        elif "Llama4TextExperts" in type(sub_module).__name__:
            quantize_llama4_experts_for_hf_export(sub_module)

    quantized_state_dict = model.state_dict()

    quantized_state_dict = postprocess_state_dict(
        quantized_state_dict, kv_cache_max_bound, kv_cache_format
    )

    # Check if any layers are quantized
    if has_quantized_layers:
        quant_config["quantization"].setdefault("exclude_modules", []).append("lm_head")

    return quantized_state_dict, quant_config


def export_hf_checkpoint(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
):
    """Exports the torch model to unified checkpoint and saves to export_dir.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        export_dir: the target export path.
        save_modelopt_state: whether to save the modelopt state_dict.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        post_state_dict, hf_quant_config = _export_hf_checkpoint(model, dtype)

        # Save hf_quant_config.json for backward compatibility
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        if not save_modelopt_state:
            model._disable_modelopt_save = True
        model.save_pretrained(export_dir, state_dict=post_state_dict)

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        config_data["quantization_config"] = hf_quant_config

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)

    except Exception as e:
        fallback_model_path = f"{export_dir}/modelopt_model.pth"
        torch.save(model.state_dict(), fallback_model_path)
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        raise e
