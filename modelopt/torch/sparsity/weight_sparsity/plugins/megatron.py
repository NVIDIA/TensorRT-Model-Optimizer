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

"""Support sparsify and save/resore for Megatron."""

import megatron.core.transformer.mlp as megatron_mlp
from megatron.core.parallel_state import get_data_parallel_group
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from modelopt.torch.opt.plugins.megatron import _MegatronMLP

from ..config import SparseGPTConfig, SparseMagnitudeConfig
from ..module import SparseModule, SpDMRegistry


def ensure_metadata_has_dp_cp_group(metadata):
    """Ensure `metadata` is a dict containing `dp_cp_group` entry.

    If `metadata` is None, a new dict is returned with `dp_cp_group` set.
    If `metadata` is a dict and missing `dp_cp_group`, it is updated in-place.

    This function is adapted from megatron-lm's megatron.core.transformer.utils to avoid
    dependency on megatron-lm's specific version.

    Note:
        This is a temporary method and will be removed once this function is merged to
        megatron.core.transformer.utils in the main branch of megatron-lm.
    """
    if metadata is None:
        metadata = {}
    if "dp_cp_group" not in metadata:
        try:
            metadata["dp_cp_group"] = get_data_parallel_group(with_context_parallel=True)
        except (AssertionError, RuntimeError):
            # Fallback if context parallel is not initialized
            metadata["dp_cp_group"] = get_data_parallel_group()
    return metadata


class _MegatronParallelLinear(SparseModule):
    def _get_shard_axis_dict(self, state_dict):
        raise NotImplementedError

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Ensure metadata has dp_cp_group to avoid None subscript errors
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        sparse_state_dict = {
            k: v
            for k, v in self.state_dict(prefix="", keep_vars=True).items()
            if k == "_weight_mask"
        }

        sharded_axis_dict = self._get_shard_axis_dict(sparse_state_dict)

        if sparse_state_dict:
            sharded_state_dict.update(
                **make_sharded_tensors_for_checkpoint(
                    sparse_state_dict, prefix, sharded_axis_dict, sharded_offsets
                )
            )

        return sharded_state_dict


@SpDMRegistry.register(
    {ColumnParallelLinear: "megatron.core.tensor_parallel.layers.ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self, state_dict):
        return {"_weight_mask": 0}


@SpDMRegistry.register(
    {RowParallelLinear: "megatron.core.tensor_parallel.layers.RowParallelLinear"}
)
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    def _get_shard_axis_dict(self, state_dict):
        return {"_weight_mask": 1}


@SpDMRegistry.register({megatron_mlp.MLP: "megatron.core.transformer.mlp.MLP"})
class _SparseMegatronMLP(_MegatronMLP):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`."""

    _modelopt_state_keys = [r"\._weight_mask$"]


def _get_extra_rules():
    """Get the extra rules for megatron."""
    return {
        "megatron.core.tensor_parallel.layers.ColumnParallelLinear": {
            "*": {},
            "*output_layer*": None,
        },
        "megatron.core.tensor_parallel.layers.RowParallelLinear": {
            "*": {},
            "*output_layer*": None,
        },
        "megatron.core.transformer.mlp.MLP": {},
    }


# Update the default rules
SparseMagnitudeConfig.register_default(_get_extra_rules())
SparseGPTConfig.register_default(_get_extra_rules())
