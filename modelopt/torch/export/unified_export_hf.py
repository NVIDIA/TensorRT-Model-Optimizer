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

import json
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from modelopt.torch.quantization import set_quantizer_by_cfg_context
from modelopt.torch.quantization.nn import SequentialQuantizer

from .layer_utils import get_experts_list, is_layernorm, is_moe, is_quantlinear
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_NVFP4,
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
        # Run forward pass so that all modules sharing the same input are collected using forward hook.

        with set_quantizer_by_cfg_context(model, {"*": {"enable": False}}):
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
            and tensor in output_to_layernorm.keys()
        ):
            # Pre quant scale of modules is already updated to avg_pre_quant_scale
            fuse_prequant_layernorm(output_to_layernorm[tensor], modules)


def _export_hf_checkpoint(
    model: nn.Module, dtype: Optional[torch.dtype] = None
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
    else:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )

    # Base model layers
    layer_pool = {
        f"model.layers.{name}": sub_module
        for name, sub_module in model.model.layers.named_modules()
    }
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
        pass

    quant_config = get_quant_config(layer_pool)

    kv_cache_max_bound = 0
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    cache_bound_mapping = {
        KV_CACHE_NVFP4: 6 * 448,
        KV_CACHE_FP8: 448,
    }

    # Only update kv_cache_max_bound if a quantization is applied.
    if kv_cache_format != QUANTIZATION_NONE:
        kv_cache_max_bound = cache_bound_mapping.get(kv_cache_format)

    for name, sub_module in layer_pool.items():
        if is_quantlinear(sub_module):
            quantization_format = get_quantization_format(sub_module)
            block_size = get_weight_block_size(sub_module)

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

                if hasattr(sub_module, "input_quantizer") and "disabled" not in repr(
                    sub_module.input_quantizer
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

    if quantization_format != QUANTIZATION_NONE:
        quant_config["quantization"].setdefault("exclude_modules", []).append("lm_head")

    return quantized_state_dict, quant_config


def export_hf_checkpoint(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
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

        # Save config
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        # Save model
        if not save_modelopt_state:
            model._disable_modelopt_save = True
        model.save_pretrained(export_dir, state_dict=post_state_dict)

    except Exception as e:
        fallback_model_path = f"{export_dir}/modelopt_model.pth"
        torch.save(model.state_dict(), fallback_model_path)
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        raise e
