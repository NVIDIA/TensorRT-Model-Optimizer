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

try:
    from megatron.core.models.mamba import MambaModel

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# NOTE: No need to register symbols for VocabParallelEmbedding, SelfAttention, MLP, LayerNorm, Row/Col Parallel Linear,
# etc. as they are not traced and manually handled in the _DynamicMCoreLanguageModel class
@SymMap.register([GPTModel] + ([MambaModel] if HAS_MAMBA else []))
def get_megatron_language_model_sym_info(mod) -> SymInfo:
    """Get symbol information for ``GPTModel`` and ``MambaModel`` layers."""
    hidden_size = Symbol(is_searchable=True)
    return SymInfo(is_shape_preserving=True, hidden_size=hidden_size)
