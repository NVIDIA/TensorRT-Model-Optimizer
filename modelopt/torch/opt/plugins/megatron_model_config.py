# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Megatron-Core model config (TransformerConfig+)."""

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Llama31Config8B(TransformerConfig):
    """Configuration class for GPT models.

    Extends TransformerConfig with additional parameters specific to GPT models
    and provides utility methods for model configuration.
    """

    # From megatron.core.models.gpt.gpt_model.GPTModel
    transformer_layer_spec = None
    vocab_size: int = None
    max_sequence_length: int = 8192
    position_embedding_type = "rope"
    rotary_percent: float = 1.0
    rotary_base: int = 500000
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0

    # Specific TransformerConfig
    seq_length: int = 8192
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    kv_channels: int = 128
    num_attention_heads: int = 32
    num_query_groups: int = 8
    init_method_std: float = 0.01
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1.0e-05
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Different from the default values in TransformerConfig
    attention_softmax_in_fp32: bool = False
    gradient_accumulation_fusion: bool = False
