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
from contextlib import contextmanager
from typing import Any

import megatron.core.parallel_state as mcore_parallel
import megatron.core.tensor_parallel.layers as megatron_parallel
import megatron.core.transformer.mlp as megatron_mlp
import torch
import torch.nn as nn
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

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


def quant_module_get_extra_state(self) -> dict:
    """Populating the extra_state when state_dict() is called.

    quantizer_state is usually stored with in the modelopt_state
    metadata where the keys are the full module name. The issue
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

    return extra_state


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
    """
    if state is None:
        return

    quantizer_state = state.get("modelopt_quantizer_state", None)

    if quantizer_state is not None and self.allow_post_restore:
        for name, module in self.named_modules():
            if isinstance(module, TensorQuantizer):
                module.set_from_modelopt_state(quantizer_state[name])
        self.modelopt_post_restore()

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


class _RealQuantMegatronColumnParallelLinear(RealQuantLinear, _MegatronColumnParallelLinear):
    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        return any(k in key for k in self.list_of_scale_tensors)

    def _get_shard_axis_dict(self, state_dict):
        shard_axis_dict = super()._get_shard_axis_dict(state_dict)
        for k in state_dict:
            if (
                any(k.endswith(suffix) for suffix in self.list_of_scale_tensors)
                and state_dict[k].dim() > 1
            ):
                shard_axis_dict[k] = 0
        return shard_axis_dict

    def modelopt_post_restore(self, prefix: str = ""):
        # First follow the fake quant behavior to initialize tensor_quantizers
        with _view_as_fake_quant_module(self):
            super().modelopt_post_restore(prefix=prefix)

        # Restore dtype of real quant parameters in tensor_quanitzer
        _restore_real_quant_parameters(self)


class _RealQuantMegatronRowParallelLinear(RealQuantLinear, _MegatronRowParallelLinear):
    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        return any(k in key for k in self.list_of_scale_tensors)

    def _get_shard_axis_dict(self, state_dict):
        shard_axis_dict = super()._get_shard_axis_dict(state_dict)
        for k in state_dict:
            if (
                any(k.endswith(suffix) for suffix in self.list_of_scale_tensors)
                and state_dict[k].dim() > 1
            ):
                shard_axis_dict[k] = 1
        return shard_axis_dict

    def modelopt_post_restore(self, prefix: str = ""):
        # Fisrt follow the fake quant behavior to initialize tensor_quantizers
        with _view_as_fake_quant_module(self):
            super().modelopt_post_restore(prefix=prefix)

        # Restore dtype of real quant parameters in tensor_quanitzer
        _restore_real_quant_parameters(self)


@contextmanager
def _view_as_fake_quant_module(module: RealQuantLinear):
    """View the module as a fake quantized module."""
    # skip if the module is not a RealQuantLinear or QTensorWrapper
    if not isinstance(module, RealQuantLinear):
        yield
        return
    assert isinstance(module.weight, QTensorWrapper), "module.weight is not a QTensorWrapper"
    try:
        quantized_weight = module.weight
        dummy_dequantized_weight = torch.rand(
            module.weight.metadata["shape"],
            dtype=module.weight.metadata["dtype"],
            device=module.weight.device,
        )
        module.weight_quantizer._fake_quant = True
        module.weight_quantizer._dequantize = False
        module.weight = nn.Parameter(dummy_dequantized_weight)
        yield
    finally:
        module.weight_quantizer._fake_quant = False
        module.weight_quantizer._dequantize = True
        module.weight = quantized_weight


def _restore_real_quant_parameters(module: RealQuantLinear):
    """Restore the real quant parameters in the tensor_quanitzer by performing real weight quantization again."""
    dequantized_weight = module.weight_quantizer(module.weight)
    module.weight_quantizer._fake_quant = False
    module.weight_quantizer._dequantize = False
    for k in ["_scale", "double_scale", "_scale_zeros"]:
        if hasattr(module.weight_quantizer, k):
            delattr(module.weight_quantizer, k)
    module.weight = QTensorWrapper(module.weight_quantizer(dequantized_weight))
