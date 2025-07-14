# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This module provides a GEMM function for nvfp4 quantization."""

from typing import Any

import torch
from torch.autograd import Function

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.gemm_registry import gemm_registry
from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
from modelopt.torch.quantization.qtensor import NVFP4QTensor, QTensorWrapper


def _to_nvfp4(inputs, amax=None):
    """Convert inputs to nvfp4."""
    vec_size = 16
    if amax is None:
        amax = inputs.abs().amax()
    global_scale = 448.0 * 6.0 / (amax.float())
    fp4, scale = torch.ops.trtllm.fp4_quantize(inputs, global_scale, vec_size, False)
    return fp4, scale, global_scale


def nvfp4_gemm(quant_module, input_tensor, bias=None):
    """GEMM function for fp4 quantization."""
    weight = quant_module.weight
    if isinstance(weight, QTensorWrapper):  # weight is already compressed.
        weight = weight.get_qtensor()
        assert isinstance(weight, NVFP4QTensor)
        weight_fp4 = weight._quantized_data
        weight_scale = quant_module.weight_quantizer._scale
        weight_global_scale = 1.0 / quant_module.weight_quantizer._double_scale
    else:
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        if weight.dtype == torch.float32:
            weight = weight.to(torch.float16)
        weight_fp4, weight_scale, weight_global_scale = _to_nvfp4(
            weight, quant_module.weight_quantizer.amax
        )

    if input_tensor.dtype == torch.float32:
        input_tensor = input_tensor.to(torch.float16)
    input_amax = quant_module.input_quantizer.amax
    if input_amax is None:
        input_amax = input_tensor.abs().amax()
    input_global_scale = 448.0 * 6.0 / input_amax.float()
    output = torch.ops.quant.fp4_linear(
        input_tensor,
        weight_fp4,
        bias=bias,
        input_scale=input_global_scale,
        weight_scale=weight_scale,
        alpha=1.0 / weight_global_scale / input_global_scale,
    )
    return output


class Nvfp4Linear(Function):
    """Linear layer with FP4 quantization."""

    @staticmethod
    def forward(ctx, quant_module, input_tensor, weight, bias=None):
        """Forward method."""
        ctx.save_for_backward(
            input_tensor if weight.requires_grad else None,
            weight if input_tensor.requires_grad else None,
            torch.empty(0, dtype=torch.uint8) if bias is not None and bias.requires_grad else None,
            getattr(quant_module.weight_quantizer, "_scale", None),
            getattr(quant_module.weight_quantizer, "_double_scale", None),
        )
        ret = nvfp4_gemm(quant_module, input_tensor, bias)
        return ret

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward method.

        For input, we will save the unquantized input and use it directly to compute the weight
        gradient. For weight, if the weight is compressed, we will save the quantized weight and
        dequantize it to compute the input gradient. If the weight is not compressed, we will save
        the unquantized weight and use it directly to compute the input gradient.
        """
        input_tensor, weight, compute_bias_grad, scale, double_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if weight is not None:
            if isinstance(weight, QTensorWrapper):
                weight = weight.get_qtensor()
                assert isinstance(weight, NVFP4QTensor)
                # Only block_size=16 is supported, this is also check in _nvfp4_availability_check
                weight = weight.dequantize(
                    scale=scale, double_scale=double_scale, block_sizes={-1: 16}
                )
            grad_input = grad_outputs @ weight
        if input_tensor is not None:
            grad_weight = grad_outputs.transpose(-2, -1) @ input_tensor
        if compute_bias_grad is not None:
            # Sum all dimensions except the last one
            grad_bias = grad_outputs.sum(dim=list(range(grad_outputs.dim() - 1)))
        return None, grad_input, grad_weight, grad_bias

    @classmethod
    def apply(cls, *args, **kwargs):
        """Get rid of kwargs because super does not support kwargs."""
        additional_args = tuple(kwargs.values())
        return super().apply(*args, *additional_args)


def _nvfp4_availability_check(module, input, args, kwargs):
    """Comprehensive check for FP4 GEMM availability."""
    # NOTE: Having the import at the top causes mpirun commands inside pytest (vlm_ptq) to fail without any error
    try:
        import tensorrt_llm._torch.auto_deploy.custom_ops.quant  # noqa: F401
    except ImportError:
        return False

    # Check hardware support
    if not torch.cuda.is_available() or not fp4_compatible():
        return False

    # Check module type
    if not isinstance(module, RealQuantLinear):
        return False

    # Check quantizer presence and configuration
    if not hasattr(module, "input_quantizer") or not hasattr(module, "weight_quantizer"):
        return False

    quant_cfg: dict[str, Any] = mtq.NVFP4_DEFAULT_CFG["quant_cfg"]
    # Quantizer configs
    input_cfg = quant_cfg["*input_quantizer"]
    weight_cfg = quant_cfg["*weight_quantizer"]

    # Check input quantizer config
    for key, value in input_cfg.items():
        if key == "enable":
            continue
        if (
            not hasattr(module.input_quantizer, key)
            or getattr(module.input_quantizer, key) != value
        ):
            return False

    # Check weight quantizer config
    for key, value in weight_cfg.items():
        if key == "enable":
            continue
        if (
            not hasattr(module.weight_quantizer, key)
            or getattr(module.weight_quantizer, key) != value
        ):
            return False

    # When the input.shape[1] is not the multiple of 64, GEMM will sometimes output NaN.
    # When the weight.shape[0] is not the multiple of 32, GEMM will not support.
    if input.shape[1] % 64 != 0 or module.weight.shape[0] % 32 != 0:
        return False

    # Only block_size==16 is supported.
    return not (
        module.input_quantizer.block_sizes[-1] != 16
        or module.weight_quantizer.block_sizes[-1] != 16
    )


# Register default implementations
gemm_registry.register(
    gemm_func=Nvfp4Linear.apply,
    availability_check=_nvfp4_availability_check,
)
