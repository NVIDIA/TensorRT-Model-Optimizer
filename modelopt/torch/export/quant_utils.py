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

"""Utils for quantization including scaling factors adjustments."""

import warnings
from typing import Any, Optional, Union
from warnings import warn

import torch
import torch.nn as nn

from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import is_quantized_linear

from ..quantization.nn import SequentialQuantizer, TensorQuantizer
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_INT8,
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_W4A8_AWQ,
)


def get_scaling_factor_from_weight(weight, group_size) -> torch.tensor:
    """Calculate the weight scaling factor for a given group size."""
    [n, k] = weight.shape

    if group_size != 0:
        # int4_awq
        if k % group_size != 0:
            raise NotImplementedError(
                "Weight shape is not divisible for block size for block quantization."
            )
        weight = weight.reshape(n, k // group_size, group_size)
        maxbound = 7.0
    else:
        # int8_sq
        maxbound = 127.0
    amax = weight.abs().max(dim=-1)[0].float()

    weights_scaling_factor = amax / maxbound

    # Let's filter the zeros in the scaling factor if the weights are zero
    # to avoid the divided-by-zero error..
    weights_scaling_factor[weights_scaling_factor == 0] = 1.0

    return weights_scaling_factor


def resmooth_and_get_scale(
    merged_weights: torch.Tensor,
    pre_quant_scales: list[torch.Tensor],
    ranks: int,
    group_size: int,
    new_pre_quant_scale: torch.Tensor = None,
    quantization: Optional[str] = QUANTIZATION_NONE,
):
    """Resmooths weights from a single or multiple ranks and get scaling factors and amax.

    Args:
        merged_weights: Merged weights from ranks.
        pre_quant_scales: List of pre-quantization scales for each rank.
        ranks: Number of ranks.
        group_size: Group size of the quantization block.
        new_pre_quant_scale (optional): If not provided, weights will be resmoothed using
            the average of pre_quant_scales.

    Returns:
        weights: Resmoothed weights.
        weight_scaling_factors: Resmoothed scaling factors.
        avg_pre_quant_scale: Calculated average of the quantization scale.
    """
    if new_pre_quant_scale is None:
        new_pre_quant_scale = torch.stack(pre_quant_scales).mean(dim=0)

    assert (
        len(pre_quant_scales) > 0 and new_pre_quant_scale.numel() == merged_weights.shape[1]
    ), "Shape of pre_quant_scales and weights do not match."
    weights = torch.chunk(merged_weights, ranks, dim=0)

    scales = []
    new_weights = []
    for i, p_scaling_factor in enumerate(pre_quant_scales):
        # De smooth & Re smooth
        weight = (
            weights[i]
            * p_scaling_factor.type(weights[i].dtype)
            / new_pre_quant_scale.type(weights[i].dtype)
        )
        new_weights.append(weight)
        # If NVFP4_AWQ then we view the scales as uint8 to allow for cat later
        if quantization == QUANTIZATION_NVFP4_AWQ:
            scale, _ = NVFP4QTensor.get_weights_scaling_factor(weight, group_size).view(torch.uint8)
        else:
            scale = get_scaling_factor_from_weight(weight, group_size)
        scales.append(scale)

    resmoothed_scales = torch.cat(scales, dim=0)

    return (
        torch.cat(new_weights, dim=0),
        resmoothed_scales.view(torch.float8_e4m3fn)
        if quantization == QUANTIZATION_NVFP4_AWQ
        else resmoothed_scales,  # if NVFP4_AWQ we view the scales back as float8_e4m3fn after cat
        new_pre_quant_scale,
    )


def adjust_attn_amax_values(module):
    """Adjusts the amax values for the attention layers."""
    projection_prefixes = ["q", "k", "v"]
    max_amax = float("-inf")
    proj_layers = []

    # Find all projection layers whose names contain 'q', 'k', or 'v'
    for name, sub_module in module.named_children():
        for prefix in projection_prefixes:
            if (
                prefix in name
                and hasattr(sub_module, "weight_quantizer")
                and hasattr(sub_module.weight_quantizer, "amax")
            ):
                proj_layers.append(sub_module)
                max_amax = max(max_amax, sub_module.weight_quantizer.amax.item())

    if not proj_layers:
        raise ValueError(
            "No projection layers with the specified prefixes ('q', 'k', 'v') have amax attributes"
        )

    assert max_amax > 0, "max_amax must be positive."

    # Set all amax values to the maximum found
    for proj_layer in proj_layers:
        proj_layer.weight_quantizer.amax.fill_(max_amax)


def get_scaling_factor(quantizer: TensorQuantizer) -> torch.Tensor:
    """Returns scaling factor from the quantizer as torch.Tensor."""
    if not quantizer.is_enabled:
        return None

    amax = quantizer.export_amax()
    if amax is None:
        return None

    # tensorrt_llm uses float as the scaling_factors.
    scaling_factor = amax.float() / quantizer.maxbound

    assert torch.all(scaling_factor > 0), f"scaling factor {scaling_factor} not positive."

    return scaling_factor


def get_activation_scaling_factor(module: nn.Module) -> torch.Tensor:
    """Returns the activation scaling factor."""
    # If NVFP4, return activation scaling factor from NVFP4QTensor
    if get_quantization_format(module) in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
    ] and hasattr(module, "input_quantizer"):
        return NVFP4QTensor.get_activation_scaling_factor(module.input_quantizer)
    return (
        get_scaling_factor(module.input_quantizer) if hasattr(module, "input_quantizer") else None
    )


def get_weight_scaling_factor(module: nn.Module) -> torch.Tensor:
    """Returns the weight scaling factor."""
    # module.weight_quantizer could be a TensorQuantizer (for algorithms except W4A8) or
    # a SequentialQuantizer (for W4A8). In the latter case, we need to get the scaling factor from the
    # first quantizer of the SequentialQuantizer instance.
    if hasattr(module, "weight_quantizer") and isinstance(
        module.weight_quantizer, SequentialQuantizer
    ):
        return get_scaling_factor(module.weight_quantizer[0])

    # If NVFP4, we need to return quantized per_block scaling factors
    if get_quantization_format(module) in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
    ] and hasattr(module, "weight_quantizer"):
        return NVFP4QTensor.get_weights_scaling_factor(
            module.weight,
            module.weight_quantizer.block_sizes[-1],
            NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(module.weight_quantizer),
        )[0]

    return (
        get_scaling_factor(module.weight_quantizer) if hasattr(module, "weight_quantizer") else None
    )


def get_weight_scaling_factor_2(module: nn.Module) -> torch.Tensor:
    """Returns the secondary weight scaling factor."""
    if get_quantization_format(module) in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
    ] and hasattr(module, "weight_quantizer"):
        return NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(module.weight_quantizer)
    if (
        not hasattr(module, "weight_quantizer")
        or not isinstance(module.weight_quantizer, SequentialQuantizer)
        or not module.weight_quantizer[-1].is_enabled
    ):
        return None
    assert (
        len(module.weight_quantizer) == 2
    ), "modelopt only supports 2 sequential quantization layers for now"
    return get_scaling_factor(module.weight_quantizer[-1])


def get_prequant_scaling_factor(module: nn.Module) -> torch.Tensor:
    """Returns the prequant scaling factor."""
    prequant_scaling_factor = (
        module.input_quantizer._pre_quant_scale.squeeze()
        if hasattr(module, "input_quantizer")
        and hasattr(module.input_quantizer, "_pre_quant_scale")
        else None
    )

    if prequant_scaling_factor is not None:
        assert torch.all(
            prequant_scaling_factor > 0
        ), f"prequant scaling factor {prequant_scaling_factor} not positive."
    return prequant_scaling_factor


def get_kv_cache_scaling_factor(qkv_modules: list[nn.Module]) -> torch.Tensor:
    """Returns the kv_cache scaling factor if output quantizer is set. Else returns None by default."""
    scaling_factors = [
        get_scaling_factor(module.output_quantizer)
        for module in qkv_modules
        if hasattr(module, "output_quantizer")
    ]

    scaling_factors = [
        scaling_factor for scaling_factor in scaling_factors if scaling_factor is not None
    ]

    if not scaling_factors:
        return None

    scaling_factor = torch.stack(scaling_factors).max(dim=0).values

    # For FP8, we recommend default kv cache scaling factor to be 1.
    if get_kv_cache_dtype(qkv_modules) == KV_CACHE_FP8:
        if scaling_factor.item() > 0.5:
            warn(
                f"!!!!\nWarning: Large KV activations detected: {scaling_factor.item()}, "
                "Quantized KV cache may lead to higher accuracy drop.\n!!!!"
            )
        scaling_factor = torch.max(
            scaling_factor,
            torch.tensor([1.0], dtype=torch.float, device=scaling_factor.device),
        )
    return scaling_factor


def get_kv_cache_dtype(modules: Union[list[nn.Module], nn.Module]) -> Optional[str]:
    """Returns the kv_cache dtype.

    If num_bits of output_quantizer is (4, 3) then returns FP8; if it is 8, returns int8,
    otherwise returns None.

    Args:
        modules (Union[list[nn.Module], nn.Module]): The module or list of modules to inspect.

    Returns:
        str: The kv_cache dtype.
    """
    num_bits_list = []

    if isinstance(modules, nn.Module):
        modules = [modules]

    for module in modules:
        if hasattr(module, "output_quantizer") and module.output_quantizer.is_enabled:
            num_bits_list.append(module.output_quantizer.num_bits)

    if (4, 3) in num_bits_list:
        return KV_CACHE_FP8
    elif 8 in num_bits_list:
        return KV_CACHE_INT8
    else:
        return QUANTIZATION_NONE


def get_weight_block_size(module: nn.Module) -> int:
    """Returns the weight block size."""
    if not hasattr(module, "weight_quantizer"):
        return 0

    weight_quantizer = module.weight_quantizer

    if isinstance(weight_quantizer, SequentialQuantizer):
        weight_quantizer = weight_quantizer[0]

    if not weight_quantizer.is_enabled:
        return 0

    block_sizes = weight_quantizer.block_sizes

    if block_sizes:
        return block_sizes[-1]
    return 0


def get_quantization_format(module) -> Optional[str]:
    """Gets the quantization string.

    Gets the quantization string by iterating through the module and its children.
    The first non-None quantization string is returned.
    """

    def _is_enabled(quantizer):
        if isinstance(quantizer, SequentialQuantizer):
            return any([_is_enabled(q) for q in quantizer])
        return quantizer.is_enabled

    def _get_quantization_from_linear_layer(layer):
        if not hasattr(layer, "weight_quantizer") or not _is_enabled(layer.weight_quantizer):
            return QUANTIZATION_NONE
        w_quantizer = layer.weight_quantizer
        if isinstance(w_quantizer, SequentialQuantizer):
            assert (
                len(w_quantizer) == 2
                and w_quantizer[0].num_bits == 4
                and w_quantizer[1].num_bits == (4, 3)
            ), "Unsupported quantizer"
            assert (
                w_quantizer[0].block_sizes
                and len(w_quantizer[0].block_sizes) > 0
                and w_quantizer[0].block_sizes[-1] > 0
            ), "Invalid block_sizes"
            return QUANTIZATION_W4A8_AWQ
        if w_quantizer.num_bits == 4:
            assert (
                len(w_quantizer.block_sizes) > 0 and w_quantizer.block_sizes[-1] > 0
            ), "Invalid block_sizes"
            return QUANTIZATION_INT4_AWQ
        elif w_quantizer.num_bits == 8:
            return QUANTIZATION_INT8_SQ
        elif w_quantizer.num_bits == (4, 3):
            return QUANTIZATION_FP8
        elif (
            w_quantizer.num_bits == (2, 1)
            and hasattr(layer, "input_quantizer")
            and hasattr(layer.input_quantizer, "_pre_quant_scale")
        ):
            return QUANTIZATION_NVFP4_AWQ
        elif w_quantizer.num_bits == (2, 1):
            return QUANTIZATION_NVFP4
        else:
            raise NotImplementedError(
                f"Unsupported quantizer with num_bits: {w_quantizer.num_bits}"
            )

    if is_quantized_linear(module):
        return _get_quantization_from_linear_layer(module)

    for _, layer in module.named_children():
        if is_quantized_linear(layer):
            quantization = _get_quantization_from_linear_layer(layer)
        else:
            quantization = get_quantization_format(layer)

        # Try to see if other layers has quantization
        if quantization != QUANTIZATION_NONE:
            return quantization

    return QUANTIZATION_NONE


def process_layer_quant_config(layer_config_dict):
    """Processes per layer quantization information for TRTLLM export to quant_cfg.json."""
    per_layer_config: dict[str, Any] = {
        "quant_algo": None,
        "kv_cache_quant_algo": None,
        "quantized_layers": {},
    }
    layer_config: dict[str, Any] = {}
    # Set of quantization formats used.
    quantization_formats = set()

    for k, v in layer_config_dict.items():
        if "awq_block_size" in k:
            continue

        # Get layer name for constructing quantized_layers dictionary under per_layer_config
        prefix = ".".join(k.rsplit(".", 1)[:-1])
        quantization_formats.add(v)
        if v == "fp8":
            layer_config = {"quant_algo": "FP8"}
        elif v == "int4_awq":
            layer_config = {
                "quant_algo": "W4A16_AWQ",
                "group_size": layer_config_dict[prefix + ".awq_block_size"],
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        elif v == "w4a8_awq":
            layer_config = {
                "quant_algo": "W4A8_AWQ",
                "group_size": layer_config_dict[prefix + ".awq_block_size"],
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        elif v == "int8_sq":
            layer_config = {"quant_algo": "W8A8_SQ_PER_CHANNEL"}
        elif v == "nvfp4":
            layer_config = {
                "quant_algo": "NVFP4",
                "group_size": layer_config_dict[prefix + ".awq_block_size"],
            }
        else:
            layer_config = {"quant_algo": v}

        per_layer_config["quantized_layers"].update({prefix: layer_config})

    # If we have more than one quantization format, infer MIXED_PRECISION
    quantization_formats.discard(None)

    if len(quantization_formats) > 1:
        per_layer_config["quant_algo"] = "MIXED_PRECISION"
    else:
        # We return empty dictionary if we do not have more than one quantization format as
        # per layer quantization information is redundant
        per_layer_config = {}

    return per_layer_config


def to_quantized_weight(
    weight: torch.Tensor,
    weights_scaling_factor: torch.Tensor,
    quantization: str,
    weights_scaling_factor2: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
):
    """Converts the weight to the quantized (packed) format."""
    if weights_scaling_factor is not None:
        weights_scaling_factor = weights_scaling_factor.to(weight.device)

    if weights_scaling_factor2 is not None:
        weights_scaling_factor2 = weights_scaling_factor2.to(weight.device)

    if quantization == QUANTIZATION_FP8:
        if weight.dim() == 3:
            # for MOE stacked weights
            return (weight / weights_scaling_factor.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return (weight / weights_scaling_factor).to(torch.float8_e4m3fn)

    if quantization == QUANTIZATION_INT8_SQ:
        return (weight / weights_scaling_factor[:, None]).round().clamp(-128, 127).to(torch.int8)

    if quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
        out_dim = weight.shape[-2]
        assert (
            out_dim % 2 == 0
        ), f"Cannot pack weight. Out dimension {out_dim} is not an even number."
        in_dim = weight.shape[-1]
        block_size = weight.shape[-1] // weights_scaling_factor.shape[-1]
        int8_tensor = (
            (weight / weights_scaling_factor[..., :, torch.arange(in_dim) // block_size])
            .round()
            .clamp(-8, 7)
            .to(torch.int8)
        )

        if int8_tensor.dim() == 3:
            # Case of MoE, where weights are stacked
            transpose = int8_tensor.permute(0, 2, 1)  # (experts, in_dim, out_dim)
            int8_tensor = transpose.reshape(
                -1,
                in_dim,
                out_dim // 2,
                2,
            )
            int4x2_tensor = (int8_tensor[..., 0] & 0x0F) | (int8_tensor[..., 1] << 4)
            # The shape of the returned weight is (experts, out_dim // 2, in_dim)
            return int4x2_tensor.permute(0, 2, 1).contiguous()

        int8_tensor = int8_tensor.T.reshape(in_dim, out_dim // 2, 2)  # (in_dim, out_dim)
        int4x2_tensor = (int8_tensor[..., 0] & 0x0F) | (int8_tensor[..., 1] << 4)
        # The shape of the returned weight is (out_dim // 2, in_dim)
        return int4x2_tensor.T.contiguous()

    if quantization in [QUANTIZATION_NVFP4, QUANTIZATION_NVFP4_AWQ]:
        assert block_size is not None, "Block size not passed. Unable to quantize to NVFP4 format."
        assert (
            weights_scaling_factor2 is not None
        ), "Weights scaling factor 2 not passed. Unable to quantize to NVFP4 format"
        # If MoE reshape weights_scaling_factor2 to enable quantize operations
        return NVFP4QTensor.quantize(
            weight,
            block_size,
            weights_scaling_factor,
            weights_scaling_factor2.view(-1, 1, 1)
            if weights_scaling_factor2.dim() != 0
            else weights_scaling_factor2,
        )[0]._quantized_data

    raise NotImplementedError(f"quantization format {quantization} not supported")


def from_quantized_weight(
    weight: torch.Tensor,
    weights_scaling_factor: torch.Tensor,
    quantization: str,
    torch_dtype,
):
    """Converts the quantized weight to the target torch_dtype format."""
    if weight.element_size() >= 2 or weights_scaling_factor is None or not quantization:
        # No need to unquantize the weight.
        return weight.to(torch_dtype)

    if quantization == QUANTIZATION_FP8:
        # safe tensors does not support fp8 yet. So we pack the tensors as int8
        return weight.view(torch.float8_e4m3fn).to(torch_dtype) * weights_scaling_factor.to(
            torch_dtype
        )

    if quantization == QUANTIZATION_INT8_SQ:
        return weight.to(torch_dtype) * weights_scaling_factor[:, None].to(torch_dtype)

    raise NotImplementedError(f"quantization format {quantization} not supported")


def postprocess_state_dict(state_dict: dict, maxbound: float, quantization: Optional[str]) -> dict:
    """Filters out keys related to weight quantizers and updates KV cache related keys.

    Args:
        state_dict (dict): The full model state_dict.
        maxbound (float): The maximum bound value for the output quantizer.
        quantization (Optional[str]): The KV cache quantization format.

    Returns:
        dict: Filtered state_dict without unnecessary keys like '_amax' and non KV cache output quantizers.
    """
    replacements = {
        "k_proj.output_quantizer._amax": "k_proj.k_scale",
        "v_proj.output_quantizer._amax": "v_proj.v_scale",
        "input_quantizer._pre_quant_scale": "pre_quant_scale",
    }

    post_state_dict = {}

    for key, value in state_dict.items():
        # Skip keys not related to quantizers
        if "output_quantizer" not in key and "_amax" not in key:
            post_state_dict[key] = value
            continue

        # Apply replacements if the key matches any suffix in the replacements dict
        for old_suffix, new_suffix in replacements.items():
            if key.endswith(old_suffix):
                assert quantization == KV_CACHE_FP8, "Invalid quantization format."
                assert maxbound > 0, "Maxbound must be greater than zero."

                prefix = key[: -len(old_suffix)]
                if "_amax" in key:
                    value = value / maxbound

                    # Warn if scale exceeds threshold
                    if value.item() > 0.5:
                        warnings.warn(
                            "Large KV activations detected. Quantized KV cache may lead to higher accuracy drop. "
                            "Setting KV cache scaling factor to at least 1."
                        )

                    # Ensure scale is at least 1
                    value.clamp_(min=1.0)

                post_state_dict[prefix + new_suffix] = value
                break

    # Squeeze tensors with a leading dimension of 1
    for key, value in post_state_dict.items():
        if isinstance(value, torch.Tensor) and value.dim() == 3 and value.shape[0] == 1:
            post_state_dict[key] = value.squeeze(0)

    return post_state_dict


def all_items_same(item_list):
    """Checks if all elements in the provided list are the same."""
    return all(x == item_list[0] for x in item_list)


def _fuse_prequant_layernorm(
    layernorm_module: torch.nn.Module,
    modules: list[torch.Tensor],
):
    """Scales layernorm weights with avg_pre_quant_scale of the modules list and sets pre_quant_scales to be deleted."""
    layernorm_module.weight = torch.nn.Parameter(
        layernorm_module.weight * getattr(modules[0].input_quantizer, "_pre_quant_scale")
    )
    # Pre_quant_scales of modules must not be exported, since they have been fused with layernorm
    for module in modules:
        delattr(module.input_quantizer, "_pre_quant_scale")


def preprocess_linear_fusion(modules: list[torch.nn.Module], resmooth_only=False):
    """Preprocess the quantized linears that we plan to fuse.

    Use resmooth_only for MOE experts as each individual expert is not fused.
    """
    quantization_format_list = [get_quantization_format(module) for module in modules]
    assert all_items_same(quantization_format_list), "Modules have different quantization formats"

    # Activation
    if hasattr(modules[0], "input_quantizer"):
        # Resmooth
        if modules[0].input_quantizer.pre_quant_scale is not None:
            avg_prequant_scale = torch.mean(
                torch.stack([module.input_quantizer.pre_quant_scale for module in modules]),
                dim=0,
            )

            for module in modules:
                if not torch.equal(module.input_quantizer.pre_quant_scale, avg_prequant_scale):
                    module.weight = nn.Parameter(
                        module.weight
                        * module.input_quantizer.pre_quant_scale.type(module.weight.dtype)
                        / avg_prequant_scale.type(module.weight.dtype)
                    )
                    module.input_quantizer.pre_quant_scale = avg_prequant_scale

                    # Redo weights collection
                    module.weight_quantizer.reset_amax()
                    enable_stats_collection(module.weight_quantizer)
                    module.weight_quantizer(module.weight)
                    finish_stats_collection(module.weight_quantizer)

        if resmooth_only:
            return

        if modules[0].input_quantizer.is_enabled:
            assert (
                modules[0].input_quantizer.amax.numel() == 1
            ), "Only support scalar input quant amax"

            input_amax = torch.max(torch.stack([module.input_quantizer.amax for module in modules]))
            for module in modules:
                module.input_quantizer.amax = input_amax

    # Weight
    if hasattr(modules[0], "weight_quantizer"):
        is_seq_quant = isinstance(modules[0].weight_quantizer, SequentialQuantizer)

        if is_seq_quant:
            if modules[0].weight_quantizer[-1].is_enabled:
                assert len(modules[0].weight_quantizer) == 2
                weight_amax = torch.max(
                    torch.stack([module.weight_quantizer[-1].amax for module in modules])
                )
                for module in modules:
                    module.weight_quantizer[-1].amax = weight_amax

        elif (
            modules[0].weight_quantizer.is_enabled and modules[0].weight_quantizer.amax.numel() == 1
        ):
            weight_amax = torch.max(
                torch.stack([module.weight_quantizer.amax for module in modules])
            )
            for module in modules:
                module.weight_quantizer.amax = weight_amax
