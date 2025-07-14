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

"""Plugin to add NAS/Pruning support for megatron-core GPT model."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.hparam import HPType
from modelopt.torch.opt.searcher import ConstraintsDict
from modelopt.torch.opt.utils import named_hparams
from modelopt.torch.trace import Symbol
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import (
    get_module_device,
    make_divisible,
    param_num_from_forward,
    print_rank_0,
    random,
)

from ..algorithms import (
    MODULE_TYPE_TO_CONSTRAINTS_FUNC,
    ConstraintEvalFunc,
    ConstraintInterpolator,
    ConstraintsFunc,
    ConstraintsRes,
)
from ..modules import _DynamicLayerNorm
from ..modules.utils import get_sliced_tensor, get_sliced_tensor_by_slices
from ..registry import DMRegistry
from ..search_space import SampleFunc
from ..traced_hp import TracedHp

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    HAS_TE = True
except ImportError:
    HAS_TE = False

__all__ = ["_DynamicGPTModel", "drop_mcore_gpt_layers"]


class _DynamicParallelLinear(DynamicModule):
    """A parallel linear layer with dynamic hyperparams."""

    def _setup(self):
        # register hyperparameters
        self._register_hparam("input_size", TracedHp(list(range(1, self.input_size + 1))))
        self._register_hparam("output_size", TracedHp(list(range(1, self.output_size + 1))))

        # register dynamic attributes of the class
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

        # NOTE: No importance estimators are registered

    @staticmethod
    def _get_weight(mod: "_DynamicParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, weight, "output_size", "input_size")

    @staticmethod
    def _get_bias(mod: "_DynamicParallelLinear", bias: torch.Tensor | None) -> torch.Tensor | None:
        return get_sliced_tensor(mod, bias, "output_size")


@DMRegistry.register(
    {ColumnParallelLinear: "megatron.core.tensor_parallel.layers.ColumnParallelLinear"}
)
class _DynamicColumnParallelLinear(_DynamicParallelLinear):
    """A ``megatron.core.tensor_parallel.layers.ColumnParallelLinear`` layer with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_dynamic_attribute(
            "output_size_per_partition", lambda mod, val: mod.output_size
        )


@DMRegistry.register({RowParallelLinear: "megatron.core.tensor_parallel.layers.RowParallelLinear"})
class _DynamicRowParallelLinear(_DynamicParallelLinear):
    """A ``megatron.core.tensor_parallel.layers.RowParallelLinear`` layer with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_dynamic_attribute(
            "input_size_per_partition", lambda mod, val: mod.input_size
        )


@DMRegistry.register(
    {VocabParallelEmbedding: "megatron.core.tensor_parallel.layers.VocabParallelEmbedding"}
)
class _DynamicVocabParallelEmbedding(DynamicModule):
    """A ``megatron.core.tensor_parallel.layers.VocabParallelEmbedding`` layer with dynamic hyperparams."""

    def _setup(self):
        self._register_hparam("embedding_dim", TracedHp(list(range(1, self.embedding_dim + 1))))
        self._register_dynamic_attribute("weight", self._get_weight)

    def _get_weight(
        self, mod: "_DynamicVocabParallelEmbedding", weight: torch.Tensor
    ) -> torch.Tensor:
        """Return the weight tensor of the embedding layer."""
        return get_sliced_tensor(mod, weight, None, "embedding_dim")


@DMRegistry.register({FusedLayerNorm: "megatron.core.fusions.fused_layer_norm.FusedLayerNorm"})
class _DynamicFusedLayerNorm(_DynamicLayerNorm):
    """A ``megatron.core.fusions.fused_layer_norm.FusedLayerNorm`` layer with dynamic hyperparams."""

    def _setup(self):
        # construct hidden_size with Hparam as last dimension
        hidden_size = list(self.hidden_size)
        hidden_size[-1] = TracedHp(list(range(1, hidden_size[-1] + 1)))

        # register the hyperparameter with a new name
        self._register_hparam("num_features", hidden_size[-1])

        # register dynamic attributes
        self._register_dynamic_attribute("weight", self._cut_to_active_features)
        self._register_dynamic_attribute("bias", self._cut_to_active_features)
        self._register_dynamic_attribute("hidden_size", self._get_normalized_shape)


class RepeatedTracedHp(TracedHp):
    """An hparam repeated N number of times to form a longer hparam.

    One key difference from ConcatTracedHp is that in ConcatTracedHp, the input hparams are not configurable,
    and only the concatenated hparam is configurable. In RepeatedTracedHp, its the opposite.
    """

    def __init__(self, hparam: TracedHp, num_repeats: int) -> None:
        """Initialize the repeated hparam."""
        self._hparam = hparam
        self._num_repeats = num_repeats
        choices = [c * self._num_repeats for c in self._hparam.choices]
        original = self._hparam.original * self._num_repeats
        super().__init__(choices, original)
        self._is_configurable = False
        self._importance_estimators = None

    @property  # type: ignore[misc]
    def active(self) -> int:
        """Return the active value of the hparam."""
        assert isinstance(self._hparam.active, int)
        return self._hparam.active * self._num_repeats

    @property
    def active_slice(self) -> TracedHp.ActiveSlice:
        """Return the currently active sorted indices or slice corresponding to the active value."""
        hp_active_slice = self._hparam.active_slice
        if isinstance(hp_active_slice, slice):
            hp_active_slice = torch.LongTensor(range(hp_active_slice.stop))

        active_slice = torch.cat(
            [hp_active_slice + i * self._hparam.max for i in range(self._num_repeats)]
        )
        return active_slice

    @property  # type: ignore[misc]
    def choices(self) -> Sequence[HPType]:
        """Return available choices."""
        return [c * self._num_repeats for c in self._hparam.choices]

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> dict[Symbol, TracedHp]:
        raise NotImplementedError("RepeatedTracedHp does not support `_resolve_dependencies`!")


@DMRegistry.register({MLP: "megatron.core.transformer.mlp.MLP"})
class _DynamicMLP(DynamicModule):
    """A ``megatron.core.transformer.mlp.MLP`` layer with dynamic hyperparams."""

    def _setup(self):
        assert self.input_size == self.config.hidden_size, (
            "MLP input_size must be equal to hidden_size"
        )

        self.linear_fc1 = DMRegistry.convert(self.linear_fc1)
        self.linear_fc2 = DMRegistry.convert(self.linear_fc2)

        ffn_hidden_size = TracedHp(list(range(1, self.config.ffn_hidden_size + 1)))
        fc1_output_size = (
            RepeatedTracedHp(ffn_hidden_size, 2)
            if self.config.gated_linear_unit
            else ffn_hidden_size
        )

        self._register_hparam("ffn_hidden_size", ffn_hidden_size)
        self.linear_fc1.output_size = fc1_output_size
        self.linear_fc2.input_size = ffn_hidden_size

        self._register_dynamic_attribute("input_size", lambda mod, val: mod.linear_fc1.input_size)

        # register importance estimator for ffn_hidden_size
        # TODO: Ideally we want to set the forward hook right before search begins so previously collected activations
        # can be discarded.
        # This limitation might be fixed in OMNIML-180 (Flexible Importance Estimator)
        # where we separate the importance estimation from the dynamic module.
        self._register_temp_attribute("_activations", None)
        self.hook_handle = self.linear_fc2.register_forward_hook(self._linear_fc2_forward_hook)
        ffn_hidden_size.register_importance(self._estimate_importance)

    def _linear_fc2_forward_hook(self, module, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        If we take the square root of the sum, we get the L2 norm of the activations.
        """
        # Gather input [seq_len, batch_size, ffn_hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input = gather_from_tensor_model_parallel_region(input[0]).detach()

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if input.shape[-1] != self.get_hparam("ffn_hidden_size").max:
            return

        input = input.to(torch.float32)  # use full precision to avoid overflow
        activations = input.abs().mean(dim=0)  # [batch_size, ffn_hidden_size]
        activations = activations.pow(2).sum(dim=0)  # [ffn_hidden_size]
        if self._activations is None:
            self._activations = activations
        else:
            self._activations += activations

    def _estimate_importance(self) -> TracedHp.Importance:
        """Return the activation magnitude-based importance of the ffn_hidden_size."""
        assert self._activations is not None, "No activations collected for importance estimation."
        return self._activations

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.hook_handle.remove()
        self.linear_fc1.export()
        self.linear_fc2.export()
        super().export()
        return self


def expand_head_indices(heads: torch.LongTensor, hidden_size_per_head: int) -> torch.LongTensor:
    """Expand each head index to hidden_size_per_head indices and offset by head * hidden_size_per_head."""
    return (
        heads[:, None].repeat(1, hidden_size_per_head) * hidden_size_per_head
        + torch.arange(hidden_size_per_head, device=heads.device)[None, :]
    ).flatten()


# NOTE: We provide a parent class since we do not register to DMRegistry.
class _DynamicQKVColumnParallelLinear(DynamicModule, ColumnParallelLinear):
    """An mcore ColumnParallelLinear layer for linear_qkv with dynamic attributes."""

    def _setup(self):
        pass  # setup is done in convert

    @classmethod
    @torch.no_grad()
    def convert(
        cls, mod: ColumnParallelLinear, num_heads_per_group: TracedHp, num_query_groups: TracedHp
    ) -> "_DynamicQKVColumnParallelLinear":
        """Convert a ColumnParallelLinear to a _DynamicQKVColumnParallelLinear."""
        super().convert(mod)
        assert isinstance(mod, _DynamicQKVColumnParallelLinear)

        mod._register_hparam("input_size", TracedHp(list(range(1, mod.input_size + 1))))
        mod._register_dynamic_attribute(
            "output_size",
            lambda mod, val: (num_heads_per_group.active + 2)
            * num_query_groups.active
            * mod.config.kv_channels,
        )
        mod._register_dynamic_attribute(
            "output_size_per_partition", lambda mod, val: mod.output_size
        )
        mod._register_dynamic_attribute("weight", mod._get_weight)
        mod._register_dynamic_attribute("bias", mod._get_bias)

        mod._register_temp_attribute(
            "_parent_hparams_refs",
            {"num_heads_per_group": num_heads_per_group, "num_query_groups": num_query_groups},
        )

        return mod

    def _get_output_size_indices(self) -> torch.LongTensor:
        """Get the indices of the output size based on sorted + pruned heads and query groups."""
        num_heads_per_group_hp: TracedHp = self._parent_hparams_refs["num_heads_per_group"]
        num_query_groups_hp: TracedHp = self._parent_hparams_refs["num_query_groups"]
        num_heads_per_group: int = num_heads_per_group_hp.active
        num_query_groups: int = num_query_groups_hp.active
        max_num_heads_per_group: int = num_heads_per_group_hp.max
        max_num_query_groups: int = num_query_groups_hp.max

        if num_heads_per_group_hp._slice_order is not None:
            # Sort heads per group
            # NOTE: We use the importance instead of slice order which is sorted without the notion of groups
            attn_head_scores_per_group = num_heads_per_group_hp.importance.view(
                max_num_query_groups, max_num_heads_per_group
            )
            attn_head_ranking_per_group = attn_head_scores_per_group.argsort(descending=True)
            qkv_head_ranking_per_group = torch.hstack(
                (
                    attn_head_ranking_per_group,
                    torch.arange(
                        max_num_heads_per_group,
                        max_num_heads_per_group + 2,
                        device=attn_head_ranking_per_group.device,
                    ).repeat(max_num_query_groups, 1),
                )
            )
            qkv_head_ranking_global = (
                qkv_head_ranking_per_group
                + torch.arange(
                    0,
                    (max_num_heads_per_group + 2) * max_num_query_groups,
                    max_num_heads_per_group + 2,
                    device=attn_head_ranking_per_group.device,
                )[:, None]
            )
        else:
            qkv_head_ranking_global = torch.arange(
                max_num_query_groups * (max_num_heads_per_group + 2)
            ).view(max_num_query_groups, max_num_heads_per_group + 2)

        if num_query_groups_hp._slice_order is not None:
            # Sort query groups
            qkv_head_ranking_global = qkv_head_ranking_global[num_query_groups_hp._slice_order]

        selected_qkv_heads = qkv_head_ranking_global[
            :num_query_groups,  # Q groups
            torch.cat(
                (
                    torch.arange(num_heads_per_group),  # Q heads
                    torch.arange(max_num_heads_per_group, max_num_heads_per_group + 2),  # KV heads
                )
            ),
        ].flatten()
        selected_indices = expand_head_indices(selected_qkv_heads, self.config.kv_channels)

        return selected_indices.cpu()

    @staticmethod
    def _get_weight(mod: "_DynamicQKVColumnParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        """Return the weight tensor of the linear layer."""
        return get_sliced_tensor_by_slices(
            weight, [mod._get_output_size_indices(), mod.get_hparam("input_size").active_slice]
        )

    @staticmethod
    def _get_bias(
        mod: "_DynamicQKVColumnParallelLinear", bias: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return the bias tensor of the linear layer."""
        if bias is None:
            return bias
        return get_sliced_tensor_by_slices(bias, [mod._get_output_size_indices()])


# NOTE: We provide a parent class since we do not register to DMRegistry.
class _DynamicProjRowParallelLinear(DynamicModule, RowParallelLinear):
    """An mcore RowParallelLinear layer for linear_qkv with dynamic attributes."""

    def _setup(self):
        pass  # Rest of the setup is done in convert

    @classmethod
    @torch.no_grad()
    def convert(
        cls, mod: RowParallelLinear, num_heads_per_group: TracedHp, num_query_groups: TracedHp
    ) -> "_DynamicProjRowParallelLinear":
        """Convert a RowParallelLinear to a _DynamicProjRowParallelLinear."""
        super().convert(mod)
        assert isinstance(mod, _DynamicProjRowParallelLinear)

        mod._register_hparam("output_size", TracedHp(list(range(1, mod.output_size + 1))))
        mod._register_dynamic_attribute(
            "input_size",
            lambda mod, val: (num_heads_per_group.active)
            * num_query_groups.active
            * mod.config.kv_channels,
        )
        mod._register_dynamic_attribute("input_size_per_partition", lambda mod, val: mod.input_size)
        mod._register_dynamic_attribute("weight", mod._get_weight)
        mod._register_dynamic_attribute("bias", mod._get_bias)

        mod._register_temp_attribute(
            "_parent_hparams_refs",
            {"num_heads_per_group": num_heads_per_group, "num_query_groups": num_query_groups},
        )

        return mod

    def _get_input_size_indices(self) -> torch.LongTensor:
        """Get the indices of the input size based on sorted + pruned heads and query groups."""
        num_heads_per_group_hp: TracedHp = self._parent_hparams_refs["num_heads_per_group"]
        num_query_groups_hp: TracedHp = self._parent_hparams_refs["num_query_groups"]
        num_heads_per_group: int = num_heads_per_group_hp.active
        num_query_groups: int = num_query_groups_hp.active
        max_num_heads_per_group: int = num_heads_per_group_hp.max
        max_num_query_groups: int = num_query_groups_hp.max

        if num_heads_per_group_hp._slice_order is not None:
            # Sort heads per group
            # NOTE: We use the importance instead of slice order which is sorted without the notion of groups
            attn_head_scores_per_group = num_heads_per_group_hp.importance.view(
                max_num_query_groups, max_num_heads_per_group
            )
            attn_head_ranking_per_group = attn_head_scores_per_group.argsort(
                dim=-1, descending=True
            )
            attn_head_ranking_global = (
                attn_head_ranking_per_group
                + torch.arange(
                    0,
                    max_num_heads_per_group * max_num_query_groups,
                    max_num_heads_per_group,
                    device=attn_head_ranking_per_group.device,
                )[:, None]
            )
        else:
            attn_head_ranking_global = torch.arange(
                max_num_query_groups * max_num_heads_per_group
            ).view(max_num_query_groups, max_num_heads_per_group)

        if num_query_groups_hp._slice_order is not None:
            # Sort query groups
            attn_head_ranking_global = attn_head_ranking_global[num_query_groups_hp._slice_order]

        selected_attn_heads = attn_head_ranking_global[
            :num_query_groups,  # Q groups
            :num_heads_per_group,  # Q heads
        ].flatten()
        selected_indices = expand_head_indices(selected_attn_heads, self.config.kv_channels)

        return selected_indices.cpu()

    @staticmethod
    def _get_weight(mod: "_DynamicProjRowParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        """Return the weight tensor of the linear layer."""
        return get_sliced_tensor_by_slices(
            weight, [mod.get_hparam("output_size").active_slice, mod._get_input_size_indices()]
        )

    @staticmethod
    def _get_bias(
        mod: "_DynamicProjRowParallelLinear", bias: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return the bias tensor of the linear layer."""
        return get_sliced_tensor(mod, bias, "output_size")


@DMRegistry.register({SelfAttention: "megatron.core.transformer.attention.SelfAttention"})
class _DynamicSelfAttention(DynamicModule):
    """A ``megatron.core.transformer.attention.SelfAttention`` layer with dynamic hyperparams.

    NOTE: Layernorms apply on hidden_size_per_attention_head hence no need to convert to dynamic
    """

    def _setup(self):
        # Register hparams
        num_heads_per_group = TracedHp(
            list(
                range(
                    1,
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    + 1,
                )
            )
        )
        num_heads_per_group._strict_len = False  # allow for different length for imp and order
        num_query_groups = TracedHp(list(range(1, self.num_query_groups_per_partition + 1)))
        self._register_hparam("num_heads_per_group", num_heads_per_group)
        self._register_hparam("num_query_groups", num_query_groups)

        # Register dynamic attributes
        self._register_dynamic_attribute(
            "num_attention_heads_per_partition",
            lambda mod, val: self.num_heads_per_group * self.num_query_groups,
        )
        self._register_dynamic_attribute(
            "num_query_groups_per_partition", lambda mod, val: self.num_query_groups
        )

        # Convert the Dot Product Attention to dynamic module
        if isinstance(self.core_attention, DotProductAttention):
            _DynamicDotProductAttention: DynamicModule = type(  # noqa: N806
                "_DynamicDotProductAttention",
                (DynamicModule, DotProductAttention),
                {"_setup": lambda self: None},
            )

            self.core_attention = _DynamicDotProductAttention.convert(self.core_attention)
            self.core_attention._register_dynamic_attribute(
                "hidden_size_per_partition",
                lambda mod, val: self.config.kv_channels * self.num_attention_heads_per_partition,
            )
            self.core_attention._register_dynamic_attribute(
                "num_attention_heads_per_partition",
                lambda mod, val: self.num_attention_heads_per_partition,
            )
            self.core_attention._register_dynamic_attribute(
                "num_query_groups_per_partition",
                lambda mod, val: self.num_query_groups_per_partition,
            )
        else:
            assert HAS_TE and isinstance(self.core_attention, TEDotProductAttention)

            _DynamicTEDotProductAttention: DynamicModule = type(  # noqa: N806
                "_DynamicTEDotProductAttention",
                (DynamicModule, TEDotProductAttention),
                {"_setup": lambda self: None},
            )

            self.core_attention = _DynamicTEDotProductAttention.convert(self.core_attention)
            self.core_attention._register_dynamic_attribute(
                "num_attention_heads", lambda mod, val: self.num_attention_heads_per_partition
            )
            self.core_attention._register_dynamic_attribute(
                "num_gqa_groups", lambda mod, val: self.num_query_groups_per_partition
            )
            self.core_attention._register_dynamic_attribute(
                "num_gqa_groups_per_partition", lambda mod, val: self.num_query_groups_per_partition
            )

        # Convert the fused qkv and output projection linear layer to dynamic module
        self.linear_qkv = _DynamicQKVColumnParallelLinear.convert(
            self.linear_qkv, num_heads_per_group, num_query_groups
        )
        self.linear_proj = _DynamicProjRowParallelLinear.convert(
            self.linear_proj, num_heads_per_group, num_query_groups
        )

        # register importance estimator for linear_qkv.output_size and linear_proj.input_size
        self._register_temp_attribute("_activations", None)
        self.hook_handle = self.linear_proj.register_forward_hook(self._linear_proj_forward_hook)
        # NOTE: num_heads_per_group's slice_order will be of length num_attention_heads to be able to sort heads,
        # otherwise we would only have aggregated importance of heads per group.
        # While enforcing order during `sort_parameters`, we dont check the shape of the slice_order
        num_heads_per_group.register_importance(self._estimate_all_head_importance)
        num_query_groups.register_importance(self._estimate_query_group_importance)

    def _linear_proj_forward_hook(self, module, input, output):
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then squared and summed over batch_size.
        If we take the square root of the sum, we get the L2 norm of the activations.
        """
        # Gather input [seq_len, batch_size, query_projection_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input = gather_from_tensor_model_parallel_region(input[0]).detach()

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if (
            input.shape[-1]
            != self.get_hparam("num_heads_per_group").max
            * self.get_hparam("num_query_groups").max
            * self.config.kv_channels
        ):
            return

        input = input.to(torch.float32)  # use full precision to avoid overflow
        activations = input.abs().mean(dim=0)
        activations = activations.pow(2).sum(dim=0)  # [query_projection_size]
        if self._activations is None:
            self._activations = activations
        else:
            self._activations += activations

    def _estimate_all_head_importance(self) -> TracedHp.Importance:
        """Return the importance for num_attention_heads (num_heads_per_group * num_query_groups)."""
        assert self._activations is not None, "No activations collected for importance estimation."
        attn_head_importance = torch.linalg.vector_norm(
            self._activations.view(
                self.get_hparam("num_heads_per_group").max
                * self.get_hparam("num_query_groups").max,
                self.config.kv_channels,
            ),
            ord=2,
            dim=1,
        )
        return attn_head_importance

    def _estimate_query_group_importance(self) -> TracedHp.Importance:
        """Return the importance of the ``num_query_groups`` hparam."""
        assert self._activations is not None, "No activations collected for importance estimation."
        group_importance = torch.linalg.vector_norm(
            self._activations.view(
                self.get_hparam("num_heads_per_group").max,
                self.get_hparam("num_query_groups").max,
                self.config.kv_channels,
            ),
            ord=2,
            dim=(0, 2),
        )
        return group_importance

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.hook_handle.remove()
        self.core_attention.export()
        self.linear_qkv.export()
        self.linear_proj.export()
        super().export()
        return self


@DMRegistry.register(
    {TransformerLayer: "megatron.core.transformer.transformer_layer.TransformerLayer"}
)
class _DynamicTransformerLayer(DynamicModule):
    """A ``megatron.core.transformer.transformer_layer.TransformerLayer`` layer with dynamic hyperparams."""

    def _setup(self):
        # Convert the layernorms, self-attention, and mlp layers to dynamic modules
        self.input_layernorm = DMRegistry.convert(self.input_layernorm)
        self.self_attention = DMRegistry.convert(self.self_attention)
        self.pre_mlp_layernorm = DMRegistry.convert(self.pre_mlp_layernorm)
        self.mlp = DMRegistry.convert(self.mlp)

        # Register forward hook to collect activations for importance estimation
        self._register_temp_attribute("_scores", 0.0)
        self.hook_handle = self.register_forward_hook(
            self._layer_imp_forward_hook, with_kwargs=True
        )

    def set_hidden_size_hp(self, hidden_size: TracedHp) -> None:
        self.input_layernorm.num_features = hidden_size
        self.self_attention.linear_qkv.input_size = hidden_size
        self.self_attention.linear_proj.output_size = hidden_size
        self.pre_mlp_layernorm.num_features = hidden_size
        self.mlp.linear_fc1.input_size = hidden_size
        self.mlp.linear_fc2.output_size = hidden_size

    def _layer_imp_forward_hook(self, module, args, kwargs, output) -> None:
        """Hook to collect cosine similarity between input and output to rank layers for depth pruning."""
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

        output, _ = output  # [seq_len, batch_size, hidden_size]

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if hidden_states.shape[-1] != self.input_layernorm.get_hparam("num_features").max:
            return

        with torch.no_grad():
            # Lower cosine_similarity means higher importance hence use 1 - cosine_similarity
            score = 1 - F.cosine_similarity(hidden_states, output, dim=2).mean()
            # TODO: Check if we need to reduce over TP regions (seems like all TP have same scores anyway)
            global_score = reduce_from_tensor_model_parallel_region(score).item()
            self._scores += global_score  # aggregate sum instead of mean of scores for simplicity

    def modify(
        self,
        *,
        num_heads_per_group_divisor: int = 1,
        num_query_groups_divisor: int = 1,
        ffn_hidden_size_divisor: int = 1,
    ) -> None:
        # Modify SelfAttention hparams
        for hp_name, divisor in [
            ("num_heads_per_group", num_heads_per_group_divisor),
            ("num_query_groups", num_query_groups_divisor),
        ]:
            hp = self.self_attention.get_hparam(hp_name)
            choices = {int(make_divisible(c, divisor)) for c in hp.choices}  # type: ignore[arg-type]
            hp.choices = list(set(hp.choices) & choices | {hp.original})

        # Modify MLP hparams
        hp_mlp = self.mlp.get_hparam("ffn_hidden_size")
        choices = {int(make_divisible(c, ffn_hidden_size_divisor)) for c in hp_mlp.choices}  # type: ignore[arg-type]
        hp_mlp.choices = list(set(hp_mlp.choices) & choices | {hp_mlp.original})

    def export(self):
        """Export the dynamic module to a torch.nn.Module."""
        self.hook_handle.remove()
        self.input_layernorm.export()
        self.self_attention.export()
        self.pre_mlp_layernorm.export()
        self.mlp.export()
        super().export()
        return self

    def freeze(self):
        """Freeze the dynamic module."""
        super().freeze()
        self.input_layernorm.freeze()
        self.self_attention.freeze()
        self.pre_mlp_layernorm.freeze()
        self.mlp.freeze()


@DMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicGPTModel(DynamicModule):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _setup(self):
        assert self.config.tensor_model_parallel_size == 1, "Only TP=1 is supported."
        assert self.config.virtual_pipeline_model_parallel_size is None, (
            "Virtual pipeline parallel is not supported."
        )
        assert not self.config.sequence_parallel, "Sequence parallel is not supported."
        assert self.config.context_parallel_size == 1, "Context parallel is not supported."
        assert self.config.expert_model_parallel_size == 1, "Expert parallel is not supported."
        assert self.pre_process == is_pipeline_first_stage()
        assert self.post_process == is_pipeline_last_stage()
        assert self.position_embedding_type == "rope", "Only rope position embedding is supported."

        # Register num_layers hparam for depth pruning
        self._register_hparam("num_layers", TracedHp(list(range(1, self.config.num_layers + 1))))

        # Convert layers to dynamic modules and set the shared hidden_size hparam
        if is_pipeline_first_stage():
            self.embedding.word_embeddings = DMRegistry.convert(self.embedding.word_embeddings)
            hidden_size = self.embedding.word_embeddings.get_hparam("embedding_dim")
        else:
            hidden_size = None
        hidden_size = dist.broadcast(hidden_size, src=0)
        self._register_hparam("hidden_size", hidden_size)  # type: ignore[arg-type]

        for i in range(len(self.decoder.layers)):
            self.decoder.layers[i] = DMRegistry.convert(self.decoder.layers[i])
            self.decoder.layers[i].set_hidden_size_hp(hidden_size)

        if is_pipeline_last_stage():
            self.decoder.final_layernorm = DMRegistry.convert(self.decoder.final_layernorm)
            self.decoder.final_layernorm.num_features = hidden_size
            self.output_layer = DMRegistry.convert(self.output_layer)
            self.output_layer.input_size = hidden_size
            self.output_layer.get_hparam("output_size").choices = [self.output_layer.output_size]

        # register importance estimator for hidden_size per hook
        self._register_temp_attribute("_activations", {})
        self.hook_handles = []
        for layer in self.decoder.layers:
            self.hook_handles.append(
                layer.input_layernorm.register_forward_hook(self._emb_layernorm_forward_hook)
            )
            self.hook_handles.append(
                layer.pre_mlp_layernorm.register_forward_hook(self._emb_layernorm_forward_hook)
            )
        hidden_size.register_importance(self._estimate_importance)  # type: ignore[union-attr]

    def _emb_layernorm_forward_hook(self, module, input, output) -> None:
        """Hook to collect activations for importance estimation.

        Activations are computed as mean over seq_len and then L2 norm over batch_size.
        """
        # Gather output [seq_len, batch_size, hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        output = gather_from_tensor_model_parallel_region(output).detach()

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if output.shape[-1] != self.get_hparam("hidden_size").max:
            return

        output = output.to(torch.float32)  # use full precision to avoid overflow
        activations = output.abs().mean(dim=0)  # [batch_size, hidden_size]
        activations = activations.pow(2).sum(dim=0)  # [hidden_size]
        if module not in self._activations:
            self._activations[module] = activations
        else:
            self._activations[module] += activations

    def _estimate_importance(self) -> TracedHp.Importance:
        """Return the activation magnitude-based importance of the hidden_size."""
        assert self._activations, "No activations collected for importance estimation."
        aggregated_activations = [
            act.pow(0.5)  # L2 norm over global batch size per hook
            for act in self._activations.values()
        ]
        activations = torch.stack(aggregated_activations).sum(dim=0)  # [hidden_size]

        # Reduce over all PP ranks
        activations = activations.clone()
        torch.distributed.all_reduce(activations, op=torch.distributed.ReduceOp.SUM)  # average
        return activations

    def modify(
        self,
        *,
        hidden_size_divisor: int = 1,
        num_heads_per_group_divisor: int = 1,
        num_query_groups_divisor: int = 1,
        ffn_hidden_size_divisor: int = 1,
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            hidden_size_divisor: The divisor of the hidden_size.
            num_heads_per_group_divisor: The divisor of the self-attention num_heads_per_group.
            num_query_groups_divisor: The divisor of the self-attention num_query_groups.
            ffn_hidden_size_divisor: The divisor of the mlp ffn_hidden_size.
        """
        hp = self.get_hparam("hidden_size")
        choices = {int(make_divisible(c, hidden_size_divisor)) for c in hp.choices}  # type: ignore[arg-type]
        hp.choices = list(set(hp.choices) & choices | {hp.original})

        for layer in self.decoder.layers:
            layer.modify(
                num_heads_per_group_divisor=num_heads_per_group_divisor,
                num_query_groups_divisor=num_query_groups_divisor,
                ffn_hidden_size_divisor=ffn_hidden_size_divisor,
            )

    def _export_drop_layers(self) -> None:
        """Drop layers during export if num_layers hparam is set to a smaller value during pruning."""
        num_layers_hp = self.get_hparam("num_layers")
        if num_layers_hp.active == num_layers_hp.max:  # no depth pruning
            return

        for layer in self.decoder.layers:
            assert layer._scores > 0, "No scores collected for importance estimation."

        # gather layer scores from all TP regions
        layer_scores = {}
        for layer in self.decoder.layers:
            layer_scores[layer.layer_number] = layer._scores
        all_pp_layer_scores = [None] * get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(
            all_pp_layer_scores, layer_scores, group=get_pipeline_model_parallel_group()
        )
        layer_scores = {k: v for d in all_pp_layer_scores for k, v in d.items()}  # type: ignore[attr-defined]
        assert sorted(layer_scores.keys()) == list(range(1, num_layers_hp.max + 1))  # type: ignore[arg-type]

        # sort layers by scores and drop the lowest ones
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        layers_to_drop = [layer for layer, _ in sorted_layers[num_layers_hp.active :]]  # type: ignore[misc]
        drop_mcore_gpt_layers(self, layers_to_drop=layers_to_drop)

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        # TODO: Improve this!
        # Slice order needs to be reset before exporting since weights are already
        # force assigned and we dont want to sort them again (losing the correct order)
        for n, hp in named_hparams(self, configurable=True):
            hp.enforce_order(None)

        for handle in self.hook_handles:
            handle.remove()
        self._export_drop_layers()
        if is_pipeline_first_stage():
            self.embedding.word_embeddings.export()
        for layer in self.decoder.layers:
            layer.export()
        if is_pipeline_last_stage():
            self.decoder.final_layernorm.export()
            self.output_layer.export()
        super().export()
        return self

    def freeze(self) -> None:
        """Freeze the dynamic module."""
        super().freeze()
        for layer in self.decoder.layers:
            layer.freeze()


def drop_mcore_gpt_layers(model: nn.Module, *, layers_to_drop: list[int]) -> None:
    """Remove given layers (1-indexed) of the model (works with TP and/or PP).

    If model is a wrapper around GPTModel, we unwrap it to get the actual GPTModel.
    """
    # NOTE: If this function is invoked from _DynamicGPTModel during export, model.config.num_layers is already updated
    layers_to_drop = sorted(layers_to_drop)
    assert layers_to_drop[0] >= 1, (
        f"Layers to drop should be in range 1 to {model.config.num_layers}, got {layers_to_drop}."
    )

    for m in model.modules():
        if isinstance(m, GPTModel):
            model = m
            break
    assert isinstance(model, GPTModel), f"Model should have {GPTModel} submodule, got {model}"
    print_rank_0(f"Dropping layers {layers_to_drop} from {GPTModel}.")

    # get the number of layers remaining in each pp rank
    layers_remaining_per_pp = torch.zeros(
        get_pipeline_model_parallel_world_size(),
        dtype=torch.int,
        device=get_module_device(model),
    )
    layers_remaining = torch.tensor(
        sum(1 for layer in model.decoder.layers if layer.layer_number not in layers_to_drop),
        dtype=torch.int,
        device=get_module_device(model),
    )

    # Below distributed gather requires tensors to be on cuda
    layers_remaining_per_pp = layers_remaining_per_pp.cuda()
    layers_remaining = layers_remaining.cuda()
    torch.distributed.all_gather_into_tensor(
        layers_remaining_per_pp, layers_remaining, group=get_pipeline_model_parallel_group()
    )
    layers_remaining_per_pp = [i.item() for i in layers_remaining_per_pp]
    new_num_layers = sum(layers_remaining_per_pp)

    # reindex kept layers, exlude sharded state dict for dropped layers
    layer_offset = sum(layers_remaining_per_pp[: get_pipeline_model_parallel_rank()])
    layer_number = layer_offset + 1
    dropped_layers = []
    for layer in model.decoder.layers:
        if layer.layer_number in layers_to_drop:
            layer.layer_number = -1  # should not be used
            # layer.sharded_state_dict = lambda prefix, sharded_offsets, metadata: {}
            dropped_layers.append(layer)
        else:
            layer.layer_number = layer_number
            layer.get_transformer_layer_offset = lambda: layer_offset
            layer_number += 1

    # remove dropped layers from the modulelist
    model.decoder.layers = nn.ModuleList(
        [layer for layer in model.decoder.layers if layer.layer_number != -1]
    )
    for layer in dropped_layers:
        del layer

    model.config.num_layers = new_num_layers


class MegatronConstraintsFunc(ConstraintsFunc):
    """A Functor class to check if sub-net satisfied all provided constraints.

    We intentionally expose some attributes like `limits` s.t. we can modify it manually.
    """

    _sample_points_dict: dict[tuple[str, ...], dict[str, SampleFunc]] = {
        ("params",): {"min": min, "centroid": random.centroid, "max": max},
    }

    def __init__(
        self,
        model: MegatronModule,
        constraints: ConstraintsDict,
        dummy_input: Any | tuple[Any, ...],
        deployment: dict | None = None,
        fast_eval: bool = True,
    ):
        """Initialize with additional data parallel group info from megatron."""
        for key in constraints:
            if key != "params":
                raise ValueError("Only params constraints is supported for MegatronModule!")

        self.model = model
        self.dummy_input = dummy_input
        self.deployment = deployment
        self._fast_eval = fast_eval

        # Getting data parallel group for
        self.dp_group = get_data_parallel_group()

        # initialize latency interpolator
        keys_for_interpolation = ("params",)
        if ConstraintsFunc.is_configurable(self.model, "depth"):
            keys_for_interpolation += ("flops_min_depth",)
        self._latency_interpolator = ConstraintInterpolator(
            self.model,
            points_funcs={k: self.constraint_eval_funcs[k] for k in keys_for_interpolation},
            value_func=self._get_true_latency,
        )
        # set fast/regular mode for latency interpolator
        self._latency_interpolator.collect_mode = not self.fast_eval

        # set limit at the end with setter to use sanity checks on constraints
        self._limits = {}
        self.limits = constraints

    @property
    def constraint_eval_funcs(self) -> dict[str, ConstraintEvalFunc]:
        """Get constraint eval fns."""
        return {
            "params": self._get_params,
        }

    def _get_params(self, _: ConstraintsRes | None = None) -> float:
        """Get number of model parameters from forward pass."""
        params = param_num_from_forward(self.model, args=self.dummy_input, unit=1.0)
        reduced_params = torch.Tensor([params]).to(device=get_module_device(self.model))
        torch.distributed.all_reduce(reduced_params, group=get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(reduced_params, group=get_tensor_model_parallel_group())
        return reduced_params.item()

    def _get_flops(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs."""
        raise NotImplementedError

    def _get_flops_min_depth(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs with depth set to minimum."""
        raise NotImplementedError

    def _get_true_latency(self, _: ConstraintsRes | None = None) -> float:
        """Get true inference latency."""
        raise NotImplementedError

    def _get_latency(self, precomputed: ConstraintsRes | None = None) -> float:
        """Get inference latency from interpolator."""
        raise NotImplementedError


# Clear the mapping and reinsert.
MODULE_TYPE_TO_CONSTRAINTS_FUNC[MegatronModule] = MegatronConstraintsFunc
