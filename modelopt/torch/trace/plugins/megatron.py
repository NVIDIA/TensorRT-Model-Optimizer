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

"""Plugins for tracing Megatron modules."""

from megatron.core.models.gpt import GPTModel

from ..symbols import Symbol, SymInfo, SymMap


# NOTE: No need to register symbols for VocabParallelEmbedding, SelfAttention, MLP, LayerNorm, Row/Col Parallel Linear,
# etc. as they are not traced and manually handled in the _DynamicGPTModel class
@SymMap.register(GPTModel)
def get_megatron_gpt_model_sym_info(mod: GPTModel) -> SymInfo:
    """Get symbol information for ``GPTModel`` layers."""
    hidden_size = Symbol(is_searchable=True)
    return SymInfo(is_shape_preserving=True, hidden_size=hidden_size)
