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

"""Support quantization for SGLANG layers."""

# diff order
#
import sglang.srt.layers.quantization.base_config  # noqa: F401


class Nothing:
    pass


import importlib
import os
import types
from contextvars import ContextVar

import sglang.srt.layers.linear as sglang_linear
import sglang.srt.layers.moe.fused_moe_triton.layer as sglang_fused_moe
import sglang.srt.layers.quantization.unquant as sglang_quantization
import torch

from ...utils.distributed import ParallelState
from ..nn import QuantLinearConvBase, QuantModule, QuantModuleRegistry, TensorQuantizer

tk = importlib.import_module("triton_kernels.matmul_ogs")


_active_sglang_fused_moe = ContextVar("active_sglang_fused_moe", default=None)

# Install or upgrade the dispatcher in a versioned manner to avoid NameError on older installs
_DISPATCH_VERSION = 2
if (not hasattr(tk, "_modelopt_dispatch_version")) or (
    getattr(tk, "_modelopt_dispatch_version", 0) < _DISPATCH_VERSION
):
    _orig_fn_install = tk.matmul_ogs
    if not hasattr(tk, "_orig_matmul_ogs"):
        tk._orig_matmul_ogs = types.FunctionType(
            _orig_fn_install.__code__,
            _orig_fn_install.__globals__,
            _orig_fn_install.__name__,
            _orig_fn_install.__defaults__,
            _orig_fn_install.__closure__,
        )

    def _dispatching_matmul_ogs(
        A: torch.Tensor,
        B: torch.Tensor,
        *args,
        _get=_active_sglang_fused_moe.get,
        _orig=tk._orig_matmul_ogs,
        _dbg=(os.getenv("MODEL_OPT_SGLANG_DEBUG", "0") == "1"),
        **kwargs,
    ):
        inst = _get()
        if _dbg:
            print("[modelopt][sglang] matmul_ogs dispatch invoked; inst set:", inst is not None)
        if inst is None:
            return _orig(A, B, *args, **kwargs)
        return inst._matmul_original(A, B, *args, **kwargs)

    _orig_fn_install.__code__ = _dispatching_matmul_ogs.__code__
    _orig_fn_install.__defaults__ = _dispatching_matmul_ogs.__defaults__
    _orig_fn_install.__kwdefaults__ = _dispatching_matmul_ogs.__kwdefaults__
    tk._modelopt_dispatch_version = _DISPATCH_VERSION


class FakeQuantMethod:
    """A class that implements fake quantization methods for SGLANG models.

    This class provides functionality to apply quantization methods to model layers
    in a way that's compatible with SGLANG's architecture.
    """

    def __init__(self, quant_method):
        """Initialize the FakeQuantMethod.

        Args:
            quant_method: The quantization method to be applied to the model layers.
        """
        self.quant_method = quant_method

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the quantization method to a given layer.

        Args:
            layer (torch.nn.Module): The neural network layer to be quantized.
            x (torch.Tensor): The input tensor to the layer.
            bias (torch.Tensor | None, optional): The bias tensor to the layer. Defaults to None.

        Returns:
            torch.Tensor: The quantized output tensor.
        """
        x = layer.input_quantizer(x)
        if layer.weight_quantizer.is_enabled:
            original_weight = layer.weight
            layer.weight = layer.weight_quantizer(layer.weight)
            output = self.quant_method.apply(layer, x, bias)
            layer.weight = original_weight
        else:
            output = self.quant_method.apply(layer, x, bias)
        output = layer.output_quantizer(output)
        return output


class _SGLANGParallelLinear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.output_quantizer.disable()
        assert type(self.quant_method) is sglang_quantization.UnquantizedLinearMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.fake_quant_method = FakeQuantMethod(self.quant_method)
        self.parallel_state = ParallelState(-1, -1)

    def forward(self, input_):
        # This context manager will conflict with torch.compile
        # with replace_function(self, "quant_method", self.fake_quant_method):
        # Manually replace quant_method instead
        self._quant_method = self.quant_method
        self.quant_method = self.fake_quant_method
        output = super().forward(input_)
        self.quant_method = self._quant_method
        return output


# @QuantModuleRegistry.register({sglang_linear.ReplicatedLinear: "sglang_ReplicatedLinear"})
# class _QuantSGLANGReplicatedLinear(_SGLANGParallelLinear):
#     pass


@QuantModuleRegistry.register({sglang_linear.RowParallelLinear: "sglang_RowParallelLinear"})
class _QuantSGLANGRowParallelLinear(_SGLANGParallelLinear):
    pass


@QuantModuleRegistry.register({sglang_linear.ColumnParallelLinear: "sglang_ColumnParallelLinear"})
class _QuantSGLANGColumnParallelLinear(_SGLANGParallelLinear):
    pass


@QuantModuleRegistry.register(
    {sglang_linear.MergedColumnParallelLinear: "sglang_MergedColumnParallelLinear"}
)
class _QuantSGLANGMergedColumnParallelLinear(_SGLANGParallelLinear):
    pass


@QuantModuleRegistry.register({sglang_linear.QKVParallelLinear: "sglang_QKVParallelLinear"})
class _QuantSGLANGQKVParallelLinear(_SGLANGParallelLinear):
    pass


# ReplicatedLinear is for MoE router and should not be quantized


class _TransposedQuantization(torch.autograd.Function):
    """Applies transposed quantization.

    This is useful for weight quantization of some MoEs such as gpt-oss or Llama4 which has expert weights
    of shape (num_experts, in_dim, out_dim). Per-channel/Per-block quantization from ModelOpt
    assumes that `in_dim` is -1 dim. Hence for quantizing such MoE weights, lets use transposed quantization.
    """

    # Note: TransposedQuantization uses STE with no clipping
    @staticmethod
    def forward(ctx, inputs, quantizer):
        return quantizer(inputs.transpose(-1, -2).contiguous()).transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# FusedMoE layer requires handling for UnquantizedFusedMoEMethod
@QuantModuleRegistry.register({sglang_fused_moe.FusedMoE: "sglang_FusedMoE"})
class _QuantSGLANGFusedMoE(QuantModule):
    def _setup(self):
        self.w13_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w2_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w13_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w2_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w13_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w2_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w13_output_quantizer.disable()
        self.w2_output_quantizer.disable()
        assert type(self.quant_method) is sglang_quantization.UnquantizedFusedMoEMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.parallel_state = ParallelState(-1, -1)

    def _fold_weight_moe(self):
        self.w13_weight.data.copy_(self.w13_weight_quantizer(self.w13_weight))
        self.w2_weight.data.copy_(self.w2_weight_quantizer(self.w2_weight))
        self.w13_weight_quantizer.disable()
        self.w2_weight_quantizer.disable()

    def _matmul_original(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        # C: torch.Tensor,
        *args,
        **kwargs,
    ):
        # warnings.warn("using patched matmul_ogs", stacklevel=2)
        if B is self.w13_weight:
            # First layer of expert
            A = self.w13_input_quantizer(A)
            if self.w13_weight_quantizer.is_enabled:
                original_weight = self.w13_weight
                if self.quant_method.use_triton_kernels:
                    # note the triton kernels weight size is (num_experts, input_dim, output_dim)
                    self.w13_weight = torch.nn.Parameter(
                        _TransposedQuantization.apply(self.w13_weight, self.w13_weight_quantizer)
                    )
                else:
                    self.w13_weight = torch.nn.Parameter(self.w13_weight_quantizer(self.w13_weight))
                # vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
                out = tk._orig_matmul_ogs(A, B, *args, **kwargs)
                self.w13_weight = original_weight
            else:
                out = tk._orig_matmul_ogs(A, B, *args, **kwargs)
            if self.w13_output_quantizer.is_enabled:
                out = self.w13_output_quantizer(out)
        elif B is self.w2_weight:
            A = self.w2_input_quantizer(A)
            if self.w2_weight_quantizer.is_enabled:
                original_weight = self.w2_weight
                if self.quant_method.use_triton_kernels:
                    self.w2_weight = torch.nn.Parameter(
                        _TransposedQuantization.apply(self.w2_weight, self.w2_weight_quantizer)
                    )
                else:
                    self.w2_weight = torch.nn.Parameter(self.w2_weight_quantizer(self.w2_weight))
                out = tk._orig_matmul_ogs(A, B, *args, **kwargs)
                self.w2_weight = original_weight
            else:
                out = tk._orig_matmul_ogs(A, B, *args, **kwargs)
            if self.w2_output_quantizer.is_enabled:
                out = self.w2_output_quantizer(out)
        else:
            raise ValueError("Cannot determine first or second layer of expert")
        return out

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # This is again due to the bad coding of vLLM
        # fused_moe submodule is overwritten by the fused_moe function
        # so we need to import the fused_moe module explicitly
        # assert vllm_fused_moe_package.invoke_fused_moe_kernel is not None
        # This context manager will conflict with torch.compile
        # with replace_function(
        #     vllm_fused_moe_package,
        #     "invoke_fused_moe_kernel",
        #     self.invoke_fused_moe_quantized,
        # ):
        # self._invoke_fused_moe_quantized = self.invoke_fused_moe_quantized
        # self.invoke_fused_moe_quantized = self.invoke_fused_moe_quantized
        token = _active_sglang_fused_moe.set(self)
        try:
            output = super().forward(hidden_states, router_logits)
        finally:
            _active_sglang_fused_moe.reset(token)
        return output

    @property
    def mopt_ckpt_versn(self):
        """Checkpoint version of the modelopt."""
        return None

    @mopt_ckpt_versn.setter
    def mopt_ckpt_versn(self, version: str):
        """Set the checkpoint version for the TensorQuantizer states."""
        # vLLM defined an apply method that overwrites nn.Module.apply
        # To avoid conflicting, disable the apply call here
        # self.apply(_set_ckpt_version)
