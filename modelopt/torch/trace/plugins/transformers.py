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

"""Utilities to describe symbols in the dynamic attention module."""

from torch import nn
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention

from ..symbols import Symbol, SymInfo, SymMap

__all__ = ["SymAttentionHead"]


class SymAttentionHead(Symbol):
    """Just a special class to mark the attention head symbol."""


def get_hf_attn_sym_info(sortable_attn: bool = False) -> SymInfo:
    # embed_dim is registered as elastic incoming symbol (we don't support sorting for now!)
    embed_dim = Symbol(is_sortable=False, cl_type=Symbol.CLType.INCOMING, elastic_dims={-1})

    # num_attention_heads is registered as a special symbol
    num_attention_heads = SymAttentionHead(is_sortable=sortable_attn, is_searchable=True)

    # hidden_dim is linked to num_attention_heads. Correct handling of dependencies done in hps
    # NOTE: we assume hidden_dim is 1st dependency of num_attention_heads in hps!
    hidden_dim = Symbol(is_sortable=sortable_attn, elastic_dims={-1})
    hidden_dim.link_to(num_attention_heads)

    return SymInfo(
        is_shape_preserving=True,
        num_attention_heads=num_attention_heads,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )


@SymMap.register([BertAttention])
def get_hf_attn_sym_info_sortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)


@SymMap.register([GPTJAttention])
def get_hf_attn_sym_info_unsortable(mod: nn.Module) -> SymInfo:
    return get_hf_attn_sym_info(sortable_attn=True)
