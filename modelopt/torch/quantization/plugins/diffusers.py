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

"""Support quantization of diffusers layers."""

import functools
from collections.abc import Callable, Iterator
from functools import partial
from types import ModuleType

import onnx
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from torch.autograd import Function
from torch.nn import functional as F
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

from ..export_onnx import export_fp8_mha
from ..nn import (
    QuantConv2d,
    QuantInputBase,
    QuantLinear,
    QuantLinearConvBase,
    QuantModuleRegistry,
    TensorQuantizer,
)
from .custom import _QuantFunctionalMixin

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)

onnx_dtype_map = {
    "BFloat16": onnx.TensorProto.BFLOAT16,
    "Float": onnx.TensorProto.FLOAT,
    "Float8": onnx.TensorProto.FLOAT8E4M3FN,
    "Half": onnx.TensorProto.FLOAT16,
    "INT8": onnx.TensorProto.INT8,
    "UINT8": onnx.TensorProto.UINT8,
}
mha_valid_precisions = {"Half", "BFloat16"}


class _QuantLoRACompatibleLinearConvBase(QuantLinearConvBase):
    def _setup(self):
        assert self.lora_layer is None, (
            f"To quantize {self}, lora_layer should be None. Please fuse the LoRA layer before"
            " quantization."
        )
        return super()._setup()


@QuantModuleRegistry.register({LoRACompatibleConv: "LoRACompatibleConv"})
class _QuantLoRACompatibleConv(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantConv2d.default_quant_desc_weight


@QuantModuleRegistry.register({LoRACompatibleLinear: "LoRACompatibleLinear"})
class _QuantLoRACompatibleLinear(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantLinear.default_quant_desc_weight


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    attn, v = input, mat2
    return self.bmm2_output_quantizer(
        torch._bmm(self.softmax_quantizer(attn), self.v_bmm_quantizer(v), *args, **kwargs)
    )


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


def _quantized_sdpa(self, *args, **kwargs):
    fp8_sdpa = FP8SDPA.apply
    parameters = [
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "q_quantized_scale",
        "k_quantized_scale",
        "v_quantized_scale",
        "high_precision_flag",
    ]
    default_values = [None, None, None, None, 0.0, False, None, None, None, None, "Half"]
    param_dict = dict(zip(parameters, default_values))
    for i, arg in enumerate(args):
        param_dict[parameters[i]] = arg
    param_dict.update(kwargs)
    fp8_sdpa_args = [param_dict[param] for param in parameters]
    while fp8_sdpa_args and fp8_sdpa_args[-1] is None:
        fp8_sdpa_args.pop()
    query, key, value = fp8_sdpa_args[:3]

    if not torch.onnx.is_in_onnx_export():
        query = self.q_bmm_quantizer(query)
        key = self.k_bmm_quantizer(key)
        value = self.v_bmm_quantizer(value)

    if (
        not self.q_bmm_quantizer._dynamic
        and not self.k_bmm_quantizer._dynamic
        and not self.v_bmm_quantizer._dynamic
    ):
        q_quantized_scale = self.q_bmm_quantizer._get_amax(query)
        k_quantized_scale = self.k_bmm_quantizer._get_amax(key)
        v_quantized_scale = self.v_bmm_quantizer._get_amax(value)
    else:
        assert (
            self.q_bmm_quantizer._dynamic
            and self.k_bmm_quantizer._dynamic
            and self.v_bmm_quantizer._dynamic
        ), "QKV QDQS must be in the same type"
        q_quantized_scale, k_quantized_scale, v_quantized_scale = None, None, None

    # Get block sizes lists for each quantizer if needed
    q_block_sizes = self.q_bmm_quantizer._get_block_sizes_list(query.shape)  # type: ignore[union-attr]
    k_block_sizes = self.k_bmm_quantizer._get_block_sizes_list(key.shape)  # type: ignore[union-attr]
    v_block_sizes = self.v_bmm_quantizer._get_block_sizes_list(value.shape)  # type: ignore[union-attr]

    # We don't need to calibrate the output of softmax
    return self.bmm2_output_quantizer(
        fp8_sdpa(
            query,
            key,
            value,
            *fp8_sdpa_args[3:7],
            q_quantized_scale,
            k_quantized_scale,
            v_quantized_scale,
            self.q_bmm_quantizer.trt_high_precision_dtype
            if hasattr(self.q_bmm_quantizer, "trt_high_precision_dtype")
            else "Half",
            self._disable_fp8_mha if hasattr(self, "_disable_fp8_mha") else True,
            q_block_sizes,
            k_block_sizes,
            v_block_sizes,
        )
    )


class _QuantAttention(_QuantFunctionalMixin):
    """FP8 processor for performing attention-related computations."""

    _functionals_to_replace = [
        (torch, "bmm", _quantized_bmm),
        (torch, "baddbmm", _quantized_baddbmm),
        (F, "scaled_dot_product_attention", _quantized_sdpa),
    ]

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        for package, func_name, quantized_func in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(quantized_func, self)
            yield package, func_name, quantized_func

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.softmax_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.bmm2_output_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)


QuantModuleRegistry.register({Attention: "Attention"})(_QuantAttention)


original_scaled_dot_product_attention = F.scaled_dot_product_attention


class FP8SDPA(Function):
    """A customized FP8 SDPA op for the onnx export."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        q_quantized_scale=None,
        k_quantized_scale=None,
        v_quantized_scale=None,
        high_precision_flag=None,
        disable_fp8_mha=True,
        q_block_shape: list | None = None,
        k_block_shape: list | None = None,
        v_block_shape: list | None = None,
    ):
        """Forward method."""
        ctx.save_for_backward(query, key, value, attn_mask)
        ctx.q_quantized_scale = q_quantized_scale
        ctx.k_quantized_scale = k_quantized_scale
        ctx.v_quantized_scale = v_quantized_scale
        # During runtime, ignore x or use it as needed
        return original_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    @staticmethod
    @symbolic_helper.parse_args(
        "v", "v", "v", "v", "f", "b", "v", "t", "t", "t", "s", "b", "is", "is", "is"
    )
    def symbolic(
        g: jit_utils.GraphContext,
        query: torch._C.Value,
        key: torch._C.Value,
        value: torch._C.Value,
        attn_mask: torch._C.Value | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: torch._C.Value | None = None,
        q_quantized_scale: float | None = 1.0,
        k_quantized_scale: float | None = 1.0,
        v_quantized_scale: float | None = 1.0,
        high_precision_flag: str = "Half",
        disable_fp8_mha: bool = True,
        q_block_shape: list | None = None,
        k_block_shape: list | None = None,
        v_block_shape: list | None = None,
    ):
        """Symbolic method."""
        return export_fp8_mha(
            g,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            q_quantized_scale,
            k_quantized_scale,
            v_quantized_scale,
            high_precision_flag,
            disable_fp8_mha,
            q_block_shape,
            k_block_shape,
            v_block_shape,
        )
