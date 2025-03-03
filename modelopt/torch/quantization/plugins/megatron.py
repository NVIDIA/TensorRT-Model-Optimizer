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

import math
import warnings
from contextlib import contextmanager

import megatron.core.tensor_parallel.layers as megatron_parallel
import torch
import torch.nn.functional as F
from megatron.core.parallel_state import get_data_parallel_group, get_tensor_model_parallel_group
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from packaging.version import Version

from modelopt import __version__
from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from .custom import _ParallelLinear

__all__ = []


class _MegatronParallelLinear(_ParallelLinear):
    _functionals_to_replace = [
        (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
        (megatron_parallel, "linear_with_frozen_weight"),
    ]

    _QUANT_MIN_BLOCK_SIZE = 128

    def initialize_parallel_state(self):
        self._parallel_state = ParallelState(
            get_data_parallel_group(), get_tensor_model_parallel_group()
        )

    def _process_weight_quantizer_amax(self, k, v, quantizer_state_dict):
        quantizer_state_dict[k] = v.view(self.weight.shape[0], -1) if v.numel() > 1 else v.view(-1)

    def _process_activation_quantizer_amax(self, k, v, quantizer_state_dict):
        # if quantizer is not enabled, we need to honor the input dtype for consistency
        # TODO: This is a temporary solution and needs to be removed once megatron supports
        # non-homogeneous layers
        def _sanitize_amax_dtype_for_disabled_quantizer(k, v):
            quantizer_name = k.split(".")[0]
            if hasattr(self, quantizer_name) and not getattr(self, quantizer_name).is_enabled:
                input_dtype = getattr(self, k.split(".")[0])._input_dtype
                v = v.to(input_dtype)
            return v

        quantizer_state_dict[k] = _sanitize_amax_dtype_for_disabled_quantizer(k, v).view(-1)

    def _process_activation_quantizer_pre_quant_scale(self, k, v, quantizer_state_dict):
        quantizer_state_dict[k] = v

    def _get_shard_axis_dict(self):
        raise NotImplementedError

    @contextmanager
    def _get_quantizer_states_for_sharded_state_dict(self):
        """Get the quantizer states for sharded state_dict.

        This is a workaround to support MCore distributed checkpointing. MCore distributed checkpointing
        requires all the transformer blocks to be uniform in terms the parameters and buffers.

        However with partial quantization/mixed precision quantization, the quantizers of Linear layers are not uniform
        across the transformer blocks. For example, linear layer from a particular block might have INT4 quantization
        with `weight_quantizer.amax` of shape (Cout x Cin//block_size) while the linear layer from another block might
        be disabled and `weight_quantizer.amax` could be None.

        MCore uses `shared_state_dict` to get the state_dict of the Linear layers.
        So lets call `sharded_state_dict` from this context. We will temporarily insert dummy states for the quantizers
        with the maximum size. We will store the original quantizer state if it exists in the dummy states.
        After the `sharded_state_dict` is called, we will restore the original quantizer states.
        This will ensure that the quantizer states `shared_state_dict` are uniform across the entire model.
        """
        if self.adapt_for_old_ckpt:
            yield
            return

        original_quantizer_states = {}

        _is_original_weight_quantizer_sequential = isinstance(
            self.weight_quantizer, SequentialQuantizer
        )
        if not isinstance(self.weight_quantizer, SequentialQuantizer):
            self.weight_quantizer = SequentialQuantizer(self.weight_quantizer, TensorQuantizer())
        else:
            assert len(self.weight_quantizer) == 2, (
                f"Invalid weight_quantizer: {self.weight_quantizer}"
            )

        for name, module in self.named_modules():
            # lm_head or output_layer has no weight (reusing the embeddings)
            if self.weight is None:
                continue
            if not isinstance(module, TensorQuantizer):
                continue

            original_quantizer_states[module] = {
                "_amax": getattr(module, "_amax", None),
                "_pre_quant_scale": getattr(module, "_pre_quant_scale", None),
            }
            if "weight_quantizer" in name:
                weight_shape = self.weight.shape
                dummy_tensor_shape = (
                    weight_shape[0],
                    # if dim % block_size != 0, then we will pad the dim, so we need ceil here
                    math.ceil(weight_shape[1] / self._QUANT_MIN_BLOCK_SIZE),
                )
                if hasattr(module, "_amax"):
                    assert module._amax.ndim in [0, 2], "Invalid amax"
                    if module._amax.ndim == 2:
                        dummy_tensor = module._amax.view(weight_shape[0], -1)
                        if dummy_tensor.shape[-1] == 1:
                            # Per-Channel quantization
                            dummy_tensor = module._amax.repeat(1, dummy_tensor_shape[-1])
                        else:
                            # Per-block quantization
                            cur_amax_dim = dummy_tensor.shape[-1]
                            # if block_size > 128, padding the amax to max possible shape
                            if cur_amax_dim < dummy_tensor_shape[-1]:
                                dummy_tensor = F.pad(
                                    dummy_tensor, (0, dummy_tensor_shape[1] - cur_amax_dim)
                                )
                            elif cur_amax_dim > dummy_tensor_shape[-1]:
                                raise ValueError(
                                    "Expecting the block_size >= 128 for static block-wise quantization!"
                                )
                    elif module._amax.ndim == 0:
                        # Per-tensor quantization
                        dummy_tensor = (
                            torch.ones(
                                dummy_tensor_shape,
                                device=module._amax.device,
                                dtype=module._amax.dtype,
                            )
                            * module._amax
                        )
                    else:
                        raise ValueError("Invalid amax")
                    delattr(module, "_amax")
                else:
                    dummy_tensor = torch.zeros(
                        dummy_tensor_shape, device=self.weight.device, dtype=self.weight.dtype
                    )
            else:
                if hasattr(module, "_amax"):
                    assert module._amax.numel() == 1, "Invalid amax"
                    dummy_tensor = module._amax.view(-1)
                    delattr(module, "_amax")
                else:
                    dummy_tensor = torch.zeros(
                        1, device=self.weight.device, dtype=self.weight.dtype
                    )
            module.register_buffer("_amax", dummy_tensor)
            if "input_quantizer" in name:
                if hasattr(module, "_pre_quant_scale"):
                    dummy_tensor = module._pre_quant_scale.view(-1)
                    delattr(module, "_pre_quant_scale")
                else:
                    dummy_tensor = torch.zeros(
                        self.weight.shape[1], device=self.weight.device, dtype=self.weight.dtype
                    )
                module.register_buffer("_pre_quant_scale", dummy_tensor)

        yield

        for module, original_state in original_quantizer_states.items():
            for k, v in original_state.items():
                if hasattr(module, k):
                    delattr(module, k)
                if v is not None:
                    module.register_buffer(k, v)

        if not _is_original_weight_quantizer_sequential:
            self.weight_quantizer = self.weight_quantizer[0]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        with self._get_quantizer_states_for_sharded_state_dict():
            sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets)
            # TODO: Clean this up; We dont need all the custom handling here
            quantizer_state_dict, sharded_axis_dict = {}, self._get_shard_axis_dict()
            for k, v in self.state_dict(prefix="", keep_vars=True).items():
                if "weight_quantizer" in k and "_amax" in k:
                    self._process_weight_quantizer_amax(k, v, quantizer_state_dict)
                elif ("input_quantizer" in k or "output_quantizer" in k) and k.endswith("._amax"):
                    self._process_activation_quantizer_amax(k, v, quantizer_state_dict)
                elif k == "input_quantizer._pre_quant_scale":
                    self._process_activation_quantizer_pre_quant_scale(k, v, quantizer_state_dict)

            sharded_state_dict.update(
                **make_sharded_tensors_for_checkpoint(
                    quantizer_state_dict, prefix, sharded_axis_dict, sharded_offsets
                )
            )
        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if (
            not isinstance(self.weight_quantizer, SequentialQuantizer)
            and (prefix + "weight_quantizer.0._amax") in state_dict
        ):
            # state_dict generated from sharded_state_dict which temporarily SequentialQuantizer
            state_dict[prefix + "weight_quantizer._amax"] = state_dict.pop(
                prefix + "weight_quantizer.0._amax"
            )
            state_dict.pop(prefix + "weight_quantizer.1._amax")

        for k in list(state_dict.keys()):
            if not any(
                k.startswith(prefix + name)
                for name in ["weight_quantizer", "input_quantizer", "output_quantizer"]
            ):
                continue

            name = k.split(prefix)[-1]
            if name not in self.state_dict():
                state_dict.pop(k)
                continue

            if "weight_quantizer" in name and self.state_dict()[name].ndim == 2:
                num_cols_to_crop = self.state_dict()[name].view(self.weight.shape[0], -1).shape[-1]
                state_dict[k] = state_dict[k][..., :num_cols_to_crop].reshape(-1, 1)
            elif "_amax" in name:
                state_dict[k] = state_dict[k].view(-1)[0].view_as(self.state_dict()[name])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._modelopt_load_version = __version__  # Will get overwritten by `replace_quant_module`

    @property
    def adapt_for_old_ckpt(self) -> bool:
        if (
            self._modelopt_load_version
            and Version(self._modelopt_load_version) < Version("0.20")
            and Version(self._modelopt_load_version) != Version("0.0.0")
        ):  # version in NeMo container is 0.0.0 if installed from source without history
            warnings.warn("Old checkpoint detected. Please re-save model to avoid this warning.")
            return True
        return False


@QuantModuleRegistry.register(
    {megatron_parallel.ColumnParallelLinear: "megatron_ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    _is_column_parallel = True

    def _get_shard_axis_dict(self):
        shard_axis_dict = {}
        for k, v in self.state_dict().items():
            if "weight_quantizer" in k and "_amax" in k:
                shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_parallel.RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    _is_row_parallel = True

    def _get_shard_axis_dict(self):
        if self.adapt_for_old_ckpt:
            shard_weight_for_block_wise_quant_only = True
        else:
            shard_weight_for_block_wise_quant_only = False

        shard_axis_dict = {}
        for k, v in self.state_dict().items():
            if "weight_quantizer" in k and "_amax" in k:
                if shard_weight_for_block_wise_quant_only:
                    quantizer = self.get_submodule(k.split("._amax")[0])
                    if (
                        quantizer.block_sizes
                        and quantizer.block_sizes.get("type", None) != "dynamic"
                    ):
                        shard_axis_dict[k] = 1
                else:
                    shard_axis_dict[k] = 1
            if k == "input_quantizer._pre_quant_scale":
                shard_axis_dict[k] = 0
        return shard_axis_dict
