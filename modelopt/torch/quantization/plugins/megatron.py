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
from itertools import product

import megatron.core.tensor_parallel.layers as megatron_parallel
import megatron.core.transformer.mlp as megatron_mlp
import torch
from megatron.core.parallel_state import get_data_parallel_group, get_tensor_model_parallel_group
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from packaging.version import Version
from torch.nn import functional as F

from modelopt import __version__
from modelopt.torch.opt.plugins.megatron import _MegatronMLP
from modelopt.torch.utils.distributed import ParallelState

from ..nn import QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from .custom import _ParallelLinear

__all__ = []


class _MegatronParallelLinear(_ParallelLinear):
    _functionals_to_replace = [
        (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
        (megatron_parallel, "linear_with_frozen_weight"),
    ]

    _QUANT_MIN_BLOCK_SIZE = 16

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

    def _get_args_for_w_amax_checkpoint(self, amax: torch.Tensor, wq: TensorQuantizer):
        cout, cin = self.weight.shape

        if amax.numel() == 1:
            b0, b1 = cout, cin
        elif amax.ndim == 2 or amax.ndim == 4:
            if (
                wq.block_sizes is None
            ):  # Per-channel quantization: amax: [cout, 1]; b0 = 1, b1 = Cin
                b0, b1 = 1, cin
            else:
                # 1D Per-block quantization: amax: [cout * cin // b1, 1]; b0 = 1
                # 2D Per-block quantization: amax: [cout // b0, 1, cin // b1, 1];
                b1 = wq.block_sizes.get(-1, wq.block_sizes.get(1, 1))
                b0 = wq.block_sizes.get(-2, wq.block_sizes.get(0, 1))

            amax = amax.view(cout, -1) if amax.ndim == 2 else amax.view(amax.shape[0], -1)
            if cin % b1 != 0:
                cin = amax.shape[1] * b1
            if cout % b0 != 0:
                cout = amax.shape[0] * b0
        else:
            raise NotImplementedError(f"Invalid amax shape: {amax.shape}")
        return cout, b0, cin, b1

    @torch.no_grad()
    def _get_w_amax_for_save(self, amax: torch.Tensor, wq: TensorQuantizer):
        cout, b0, cin, b1 = self._get_args_for_w_amax_checkpoint(amax, wq)
        amax = (
            amax.reshape(cout // b0, 1, cin // b1, 1)
            .expand(cout // b0, b0, cin // b1, b1)
            .reshape(cout, cin)
        )

        if self.is_version_less_than("0.27"):
            assert wq.block_sizes is None or b1 >= 128
            shrink_factor = 128  # Backward compatibility
            if cin % shrink_factor != 0:
                amax = F.pad(
                    amax, (0, shrink_factor - cin % shrink_factor), mode="constant", value=0
                )
                warnings.warn(
                    "Padded block-quantization along input dimension. "
                    "Sharded checkpointing could fail if the tensor parallelism is changed!"
                )
        else:
            assert self.weight.shape == (cout, cin), (
                "Padded quantization! Sharded checkpointing does not support this! "
                "Please save and load the model with regular Pytorch `state_dict` or "
                "pad the weights manually and quantize again."
            )
            shrink_factor = (
                self._QUANT_MIN_BLOCK_SIZE if cin % self._QUANT_MIN_BLOCK_SIZE == 0 else 1
            )
            if wq.block_sizes is not None and wq.block_sizes.get("type", "static") == "static":
                assert shrink_factor >= self._QUANT_MIN_BLOCK_SIZE
        return amax.reshape(cout, -1, shrink_factor)[:, :, 0].clone()

    @torch.no_grad()
    def _get_w_amax_for_load(
        self, amax: torch.Tensor, broad_amax: torch.Tensor, wq: TensorQuantizer
    ):
        cout, b0, cin, b1 = self._get_args_for_w_amax_checkpoint(amax, wq)

        if self.is_version_less_than("0.27"):
            shrink_factor = 128
        else:
            shrink_factor = (
                self._QUANT_MIN_BLOCK_SIZE if cin % self._QUANT_MIN_BLOCK_SIZE == 0 else 1
            )

        broad_amax = (
            broad_amax.reshape(cout, -1, 1).expand(cout, -1, shrink_factor).reshape(cout, -1)
        )
        broad_amax = broad_amax[:, :cin]  # Slicing to remove padding
        reduced_amax = broad_amax.reshape(cout // b0, b0, cin // b1, b1)[:, 0, :, 0]
        return reduced_amax.reshape(amax.shape)

    def _process_real_quantizer_scale(self, k, v, quantizer_state_dict):
        wq = self.weight_quantizer[0]
        if (
            wq.block_sizes is not None
            and wq.block_sizes.get(-1, None)
            and wq.block_sizes.get(-2, None)
        ):
            v = v[:, None, :, None]  # Aligning scale shape with fake quantizer amax
        cout, b0, cin, b1 = self._get_args_for_w_amax_checkpoint(v, self.weight_quantizer[0])
        assert self.weight.shape == (cout, cin), (
            "Sharded checkpointing does not allow padded quantization!"
        )
        quantizer_state_dict[k] = v.reshape(cout // b0, cin // b1)

    @contextmanager
    def _quantizer_states_for_homogenous_sharded_state_dict(self):
        """Get the quantizer states for sharded state_dict.

        This is a workaround to support MCore distributed checkpointing. MCore distributed checkpointing
        requires all the transformer blocks to be uniform in terms the parameters and buffers.

        However with partial quantization/mixed precision quantization, the quantizers of Linear layers are not uniform
        across the transformer blocks. For example, linear layer from a particular block might have INT4 quantization
        with `weight_quantizer.amax` of shape (Cout x Cin//block_size) while the linear layer from another block might
        be disabled and `weight_quantizer.amax` could be None.

        MCore uses `shared_state_dict` to get the state_dict of the Linear layers.
        So lets call `sharded_state_dict` from this context.
        This context:
            1. Inserts amax, _pre_quant_scale buffers to the quantizers if they are not present.
            2. Creates a new amax by correctly broadcasting amax to weight.
        At exist, this context restores the original state of the quantizers.
        """
        if self.is_version_less_than("0.20") or self.weight is None:
            yield
            return

        original_qs = {}

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
            if not isinstance(module, TensorQuantizer):
                continue
            original_qs[module] = module.state_dict()

            amax = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
            if "weight_quantizer" in name:
                amax = self._get_w_amax_for_save(
                    module.amax if module.amax is not None else amax, module
                )
            elif module.amax is not None:
                assert module.amax.numel() == 1, (
                    f"Invalid amax shape: {module.amax.shape} for {name}"
                )
                amax = module.amax

            if hasattr(module, "_amax"):
                delattr(module, "_amax")
            module.register_buffer("_amax", amax)

            if "input_quantizer" in name:
                pqs = torch.zeros(
                    self.weight.shape[1], device=self.weight.device, dtype=self.weight.dtype
                )
                if hasattr(module, "_pre_quant_scale"):
                    pqs = module._pre_quant_scale.view(-1)
                    delattr(module, "_pre_quant_scale")
                module.register_buffer("_pre_quant_scale", pqs)

        yield

        for module, k in product(original_qs.keys(), ["_amax", "_pre_quant_scale"]):
            if hasattr(module, k):
                delattr(module, k)
            if k in original_qs[module]:
                module.register_buffer(k, original_qs[module][k])

        if not _is_original_weight_quantizer_sequential:
            self.weight_quantizer = self.weight_quantizer[0]

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

        with self._quantizer_states_for_homogenous_sharded_state_dict():
            sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets)
            # TODO: Clean this up; We dont need all the custom handling here
            quantizer_state_dict, sharded_axis_dict = {}, self._get_shard_axis_dict()
            for k, v in self.state_dict(prefix="", keep_vars=True).items():
                if "weight_quantizer" in k and "_amax" in k:
                    self._process_weight_quantizer_amax(k, v, quantizer_state_dict)
                elif "weight_quantizer" in k and "._scale" in k:
                    # real quantizer scale. homogenous checkpointing with real mixed-precision
                    # quantization is not supported and will throw an error.
                    self._process_real_quantizer_scale(k, v, quantizer_state_dict)
                elif ("input_quantizer" in k or "output_quantizer" in k) and k.endswith("._amax"):
                    self._process_activation_quantizer_amax(k, v, quantizer_state_dict)
                elif k == "input_quantizer._pre_quant_scale":
                    self._process_activation_quantizer_pre_quant_scale(k, v, quantizer_state_dict)
                elif "quantizer" in k:
                    raise NotImplementedError(f"Unsupported quantizer state: {k}")

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
            # state_dict generated for sharded_state_dict which temporarily inserted SequentialQuantizer
            for k, v in list(state_dict.items()):
                if "weight_quantizer.0" in k:
                    state_dict[k.replace("weight_quantizer.0", "weight_quantizer")] = v
                    state_dict.pop(k)
                    state_dict.pop(k.replace("weight_quantizer.0", "weight_quantizer.1"), None)

        for k in list(state_dict.keys()):
            if not any(qt + "_quantizer" in k for qt in ["weight", "input", "output"]):
                continue

            name = k.split(prefix)[-1] if prefix else k
            if name not in self.state_dict():
                state_dict.pop(k)
                continue

            if "weight_quantizer" in k:
                if "_amax" in k and state_dict[k].numel() != self.state_dict()[name].numel():
                    # Sharded state dict
                    state_dict[k] = self._get_w_amax_for_load(
                        self.state_dict()[name],
                        state_dict[k],
                        self.get_submodule(name.split("._amax")[0]),
                    )
                else:  # Regular state dict
                    state_dict[k] = state_dict[k].view_as(self.state_dict()[name])
            elif "_amax" in name:
                state_dict[k] = state_dict[k].view(-1)[0].view_as(self.state_dict()[name])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._modelopt_load_version = __version__  # Will get overwritten by `replace_quant_module`

    def is_version_less_than(self, version: str) -> bool:
        if (
            self._modelopt_load_version
            and Version(self._modelopt_load_version) < Version(version)
            and Version(self._modelopt_load_version) != Version("0.0.0")
        ):  # version in NeMo container is 0.0.0 if installed from source without history
            warnings.warn(
                f"Checkpoint version {self._modelopt_load_version} is less than {version}. "
                "Please re-save model to avoid this warning."
            )
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
            if "weight_quantizer" in k:
                shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_parallel.RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    _is_row_parallel = True

    def _get_shard_axis_dict(self):
        shard_axis_dict = {}
        for k, v in self.state_dict().items():
            if "weight_quantizer" in k:
                if self.is_version_less_than("0.20"):
                    assert "._amax" in k, f"Invalid quantizer state: {k}"
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


@QuantModuleRegistry.register({megatron_mlp.MLP: "megatron_MegatronMLP"})
class _QuantMegatronMLP(_MegatronMLP):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`."""

    _modelopt_state_keys = [
        r"weight_quantizer\.(\d+\.)*_amax$",
        r"weight_quantizer\.(\d+\.)*_scale$",
    ]
