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

"""Support quantization for megatron linear layers."""

import warnings
from typing import Any

import megatron.core.parallel_state as mcore_parallel
import megatron.core.tensor_parallel.layers as megatron_parallel
import megatron.core.transformer.mlp as megatron_mlp
import torch
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none

from modelopt.torch.opt.plugins.megatron import (
    _MegatronMLP,
    register_modelopt_extra_state_callbacks,
)
from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry, TensorQuantizer
from ..nn.modules.quant_linear import RealQuantLinear
from ..qtensor import QTensorWrapper
from .custom import CUSTOM_MODEL_PLUGINS, _ParallelLinear

__all__ = []


def real_quant_module_get_extra_state(self) -> dict:
    """Populating real_quantizer_state and q_tensor_state."""
    extra_state = {}

    if isinstance(self, RealQuantLinear) and isinstance(self.weight, QTensorWrapper):
        real_quantizer_state = self.weight_quantizer.get_modelopt_state()
        q_tensor_state = self.weight.get_state()
    elif isinstance(self, RealQuantLinear):
        real_quantizer_state = self.weight_quantizer.get_modelopt_state()
        q_tensor_state = {}
    else:
        real_quantizer_state = None
        q_tensor_state = None

    extra_state["modelopt_real_quantizer_state"] = real_quantizer_state
    extra_state["modelopt_q_tensor_state"] = q_tensor_state

    return extra_state


def quant_module_get_extra_state(self) -> dict:
    """Populating the extra_state when state_dict() is called.

    quantizer_state, real_quantizer_state, and q_tensor_state are usually stored
    with in the modelopt_state metadata where the keys are the full module name. The issue
    is that NeMo-MCore model's full module name can change
    if pipeline-parallelism (PP) and expert-parallelism (EP)
    are changing. Alternatively, we store quantizer_state in
    QuantModule's extra_state with QuantModule.get_extra_state()
    which avoids the need to store the full module name.
    """
    extra_state = {}

    is_enabled = self.weight_quantizer.is_enabled if hasattr(self, "weight_quantizer") else False

    if not is_enabled:
        return extra_state

    quantizer_state = {}
    for name, module in self.named_modules():
        if isinstance(module, TensorQuantizer):
            quantizer_state[name] = module.get_modelopt_state()

    extra_state["modelopt_quantizer_state"] = quantizer_state

    # Handle real_quantizer_state and q_tensor_state
    extra_state.update(real_quant_module_get_extra_state(self))

    return extra_state


def real_quant_module_set_extra_state(self, state: Any):
    """Restore q_tensor_state when load_state_dict() is called.

    We skip restoring real_quantizer_state (if exists), since it is the same as
    the weight_quantizer fake quantizer_state.

    Finally, q_tensor_state is restored if meta device initialization is used. During
    meta-device initialization, real_quantize is not called.
    QTensorWrapper should replace the original weight parameter. Due to TP, we also need
    to adjust q_tensor_data_shape and its metadata shape attribute to use the local weight shape.

    When not using meta device initialization, real_quantize is called during compress mode
    restore where the QTensor will be recomputed based on the local weights. Hence we don't
    need to restore q_tensor_state.

    Note:
        The entire restore process can happen on meta device and be materialized later
        with to_empty(). However, to_empty() will reassign the parameter and the
        QTensorWrapper will be removed. We patch RealQuantLinear._apply to preserve
        QTensorWrapper when to_empty() is applied.
    """
    q_tensor_state = state.get("modelopt_q_tensor_state", None)

    if q_tensor_state is not None:
        q_tensor_metadata = q_tensor_state["metadata"]
        q_tensor_metadata["shape"] = self.weight.shape
        q_tensor_data_dtype = q_tensor_state["quantized_data.dtype"]
        q_tensor_shape = self.weight.shape

        # If q_tensor_data_type is uint8, then it is compressed format of 2 elements.
        if q_tensor_data_dtype == torch.uint8:
            q_tensor_shape = list(q_tensor_shape)
            q_tensor_shape[-1] = q_tensor_shape[-1] // 2
            q_tensor_shape = torch.Size(q_tensor_shape)

        self._parameters["weight"] = QTensorWrapper(
            qtensor=torch.empty(
                q_tensor_shape,  # Use the local shape directly (TP-aware)
                dtype=q_tensor_data_dtype,
                device=self.weight.device,
            ),
            metadata=q_tensor_metadata,
        )


def quant_module_set_extra_state(self, state: Any):
    """Restore quantizer_state when load_state_dict() is called.

    With quantizer_state stored in extra_state (NeMo-MCore `torch-dist`),
    set_extra_state() is used to perform the functionality
    conversion.restore_quantizer_state().
    load_state_dict() are called twice during NeMo-MCore resume.
    The state_dict only contains the extra_state in the first time.
    set_extra_state() is trigger by the end of the load_state_dict()
    where QuantModule.modelopt_post_restore() will reinitialize
    amax and scalars to the correct shape.
    The 2nd load_state_dict() is loading all states including amax and
    scalars. We disable QuantModule.modelopt_post_restore() to avoid
    reinitialization since set_extra_state() is called at the end.

    We first restore all fake quantizer_state. Per QuantModule can have
    weight_quantizer, input_quantizer, and output_quantizer.

    Once all quantizer_state are resumed, modelopt_post_restore() is called
    to adjust the shape of all buffers (amax, pre_qunat_scale, _scale, ...) since
    the local shape can be different from the shape in the state due to change
    in tensor parallelism (TP).
    """
    if state is None or not self.allow_post_restore:
        return

    quantizer_state = state.get("modelopt_quantizer_state", None)

    if quantizer_state is not None:
        for name, module in self.named_modules():
            if isinstance(module, TensorQuantizer):
                module.set_from_modelopt_state(quantizer_state[name], properties_only=False)
        self.modelopt_post_restore()

    # Handle real_quantizer_state and q_tensor_state
    real_quant_module_set_extra_state(self, state)

    self.allow_post_restore = False


def megatron_replace_quant_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model quantization support.

    This callback is called before the QuantModule replacement to reuse the current
    custom callback infra. However, it is meant to target each QuantModule.
    Since the callback is called when megatron is installed, we do a type check on
    MegatronModule first. For each MegatronModule,
    1. We change TransformerConfig to enable heterogenous distributed checkpointing.
    2. We enable all sub- QuantModule to store quantizer_state as extra_state by
       typing-matching the QuantModuleRegistry.
    """

    def _register_extra_state_callbacks(model: torch.nn.Module):
        for name, module in model.named_modules():
            if type(module) in QuantModuleRegistry:
                # This module will be replaced as a QuantModule
                register_modelopt_extra_state_callbacks(
                    module,
                    quant_module_get_extra_state,
                    quant_module_set_extra_state,
                )

    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            if "vision_model" not in name:
                # We only enable hetereogenous_dist_checkpoint for language model, vision model is not quantized
                module.config.hetereogenous_dist_checkpoint = True
            _register_extra_state_callbacks(module)


CUSTOM_MODEL_PLUGINS.add(megatron_replace_quant_module_hook)


class _MegatronParallelLinear(_ParallelLinear):
    _functionals_to_replace = [
        (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
        (megatron_parallel, "linear_with_frozen_weight"),
    ]

    def _setup(self):
        self.parallel_state = ParallelState(
            getattr(mcore_parallel, "get_expert_data_parallel_group", "get_data_parallel_group")(),
            mcore_parallel.get_tensor_model_parallel_group(),
        )
        super()._setup()

    def _process_quantizer_amax(self, k, v, quantizer_state_dict):
        if v.ndim == 4:
            quantizer_state_dict[k] = v.squeeze(1).squeeze(-1)
        else:
            quantizer_state_dict[k] = (
                v.view(self.weight.shape[0], -1) if v.numel() > 1 else v.view(-1)
            )

    def _process_activation_quantizer_pre_quant_scale(self, k, v, quantizer_state_dict):
        quantizer_state_dict[k] = v

    def _get_shard_axis_dict(self, state_dict):
        raise NotImplementedError

    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        """Determine whether a parameter should be kept in the quantizer_state_dict.

        Used to include additional quantization parameters (e.g., _scale for real quant)
        beyond the default amax and pre_quant_scale tensors.

        Note: When adding parameters here, update _get_shard_axis_dict accordingly for sharding.
        """
        return False

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # [WAR]: although we disable output_layer quantization by default but it will
        # still be picked up by mtq.quantize since it is a ColumnParallelLinear. We need
        # to further ensure that its sharded state_dict has no scalars or amax since
        # 1) NeMo-MCore's vocabulary padding may change but we didn't support this feature
        # 2) When embedding and output_layer are sharing weights, PP>1 will have
        #    output_layer.input_quantizer._amax but TP-only does not. This lead to
        #    state_dict mismatch.
        if prefix.endswith("output_layer."):
            # assert not any("_quantizer" in k for k in self.state_dict()), "quantized output_layer"
            return super().sharded_state_dict(prefix, sharded_offsets)

        quantizer_state_dict = {}
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if "_quantizer" in k and "_amax" in k:
                self._process_quantizer_amax(k, v, quantizer_state_dict)
            elif k == "input_quantizer._pre_quant_scale":
                self._process_activation_quantizer_pre_quant_scale(k, v, quantizer_state_dict)
            elif self._parameter_to_keep_in_quantizer_state_dict(k):
                quantizer_state_dict[k] = v
            elif "quantizer" in k:
                warnings.warn(
                    f"Quantizer state {k} is not supported for sharded_state_dict. "
                    "Please use regular state_dict."
                )
        sharded_axis_dict = self._get_shard_axis_dict(quantizer_state_dict)
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets)
        sharded_state_dict.update(
            **make_sharded_tensors_for_checkpoint(
                quantizer_state_dict, prefix, sharded_axis_dict, sharded_offsets
            )
        )
        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for k in list(state_dict.keys()):
            if not any(qt + "_quantizer" in k for qt in ["weight", "input", "output"]):
                continue
            name = k.split(prefix)[-1] if prefix else k
            state_dict[k] = state_dict[k].view_as(self.state_dict()[name])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@QuantModuleRegistry.register(
    {megatron_parallel.ColumnParallelLinear: "megatron_ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    _is_column_parallel = True

    def _get_shard_axis_dict(self, state_dict):
        """Getting the sharded axis for amax and pre_quant_scale.

        By default, ColumnParallelLinear shards the output dimension (dim=0). However,
        depending the quantization algorithm, not all amax or pre_quant_scale need
        to be sharded.

        We check the quantizer.axis to decide whether an amax needs to be sharded.
        Except for dynamic block quantization (NVFP4, axis: None) or per-tensor (FP8,
        axis: None), the rest of algorithms all need to be sharded

        Prequant scaling is applied per-input-channel; hence no sharding is required.
        """
        shard_axis_dict = {}
        for k in state_dict:
            if "weight_quantizer." in k:
                weight_quantizer_axis = self.get_submodule(k.rsplit(".", 1)[0]).axis
                if weight_quantizer_axis is not None:
                    shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_parallel.RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    _is_row_parallel = True

    def _get_shard_axis_dict(self, state_dict):
        """Getting the sharded axis for amax and pre_quant_scale.

        By default, RowParallelLinear shards the input dimension (dim=1). However,
        depending the quantization algorithm, not all amax or pre_quant_scale need
        to be shard.

        We check the quantizer.axis to decide whether an amax needs to be sharded.
        Only static block quantization needs to be sharded and its axis is either (0,) or (0, 2).
        The first case is used in AWQ the later case is used in blocked 2D quantization.
        Dynamic block quantization (NVFP4 axis:None), per-tensor (FP8, axis: None)
        and per-channel (INT8_SQ or FP8_PER_CHANNEL, axis: 1) do not require input sharding.

        Prequant scaling is applied per-input-channel; hence it is always sharded.
        """
        shard_axis_dict = {}
        for k in state_dict:
            if "weight_quantizer." in k:
                weight_quantizer_axis = None
                if isinstance(self.weight_quantizer, TensorQuantizer):
                    weight_quantizer_axis = self.weight_quantizer.axis
                elif "weight_quantizer.0." in k:
                    weight_quantizer_axis = self.weight_quantizer[0].axis
                elif "weight_quantizer.1." in k:
                    weight_quantizer_axis = self.weight_quantizer[1].axis
                if isinstance(weight_quantizer_axis, tuple):
                    shard_axis_dict[k] = 1
            if k == "input_quantizer._pre_quant_scale":
                shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_mlp.MLP: "megatron_MegatronMLP"})
class _QuantMegatronMLP(_MegatronMLP):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`."""

    _modelopt_state_keys = [
        r"weight_quantizer\.(\d+\.)*_amax$",
        r"weight_quantizer\.(\d+\.)*_scale$",
    ]


class _RealQuantMegatronParallelLinear(RealQuantLinear):
    allow_real_quant_gemm = True
    _scale_tensor_shard_axis = None

    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        return any(k in key for k in self.list_of_scale_tensors)

    def _get_shard_axis_dict(self, state_dict):
        shard_axis_dict = super()._get_shard_axis_dict(state_dict)
        for k in state_dict:
            if (
                any(k.endswith(suffix) for suffix in self.list_of_scale_tensors)
                and state_dict[k].dim() > 1
            ):
                assert self._scale_tensor_shard_axis is not None, (
                    "scale_tensor_shard_axis is not set, please set it in the subclass"
                )
                shard_axis_dict[k] = self._scale_tensor_shard_axis
        return shard_axis_dict

    def modelopt_post_restore(self, prefix: str = ""):
        """Post restore to correctly configure the realquant scales.

        ModelOpt restores the TensorQuantizer states such as `_amax` and `_pre_quant_scale` to their
        shape before saving. However this is not enough for MCore/distributed frameworks since the tensor parallelism
        could change between saving and restoring. If the tensor parallelism changes, the shape of the quantizer
        states also changes. So we need to re-calculate the quantizer states.

        Note:
            During real quantization, weight_quantizer._fake_quant is set to False which trigger the real quant
            forward path and lead to error. We enable the weight_quantizer fake_quant forward path while recompute
            the correct shape.
        """
        self.weight_quantizer._fake_quant = True
        super().modelopt_post_restore(prefix=prefix)
        self.weight_quantizer._fake_quant = False

        if hasattr(self.weight_quantizer, "_scale"):
            # Recompute all real quantization buffer shapes
            self.weight_quantizer._real_quantize(self.weight)

    def _forward_impl(self, input, *args, **kwargs):
        """Use real quant gemm if available.

        Here the forward is patched such that real quant gemm can be called if available. Both conditions
        below must be satisfied (static and dynamic check based on input args) to use the kernel.
        Otherwise, we fallback.

        Note:
            RealQuantLinear.forward() is doing the same check inside and will fall back to use the super
            class forward(). This is not desired since _forward_impl introduces much more args and kwargs
            while the original forward only takes 1 positional argument. We must above the fallback path
            in RealQuantLinear.forward().
        """
        if self._should_run_real_quant_gemm and self.get_real_quant_gemm_impl(
            input, *args, **kwargs
        ):
            allreduce_dgrad = kwargs.get("allreduce_dgrad", False)
            tp_group = kwargs.get("tp_group")
            sequence_parallel = kwargs.get("sequence_parallel", False)

            tp_group = get_tensor_model_parallel_group_if_none(tp_group)

            if sequence_parallel:
                input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True, group=tp_group
                )
            else:
                input = input

            return RealQuantLinear.forward(
                self,
                input,
                allreduce_dgrad=allreduce_dgrad,
                tp_group=tp_group,
            )
        else:
            return super()._forward_impl(input, *args, **kwargs)


class _RealQuantMegatronColumnParallelLinear(
    _RealQuantMegatronParallelLinear, _MegatronColumnParallelLinear
):
    _scale_tensor_shard_axis = 0

    def forward(self, input, *args, **kwargs):
        return _MegatronColumnParallelLinear.forward(self, input, *args, **kwargs)


class _RealQuantMegatronRowParallelLinear(
    _RealQuantMegatronParallelLinear, _MegatronRowParallelLinear
):
    _scale_tensor_shard_axis = 1

    def forward(self, input, *args, **kwargs):
        return _MegatronRowParallelLinear.forward(self, input, *args, **kwargs)
