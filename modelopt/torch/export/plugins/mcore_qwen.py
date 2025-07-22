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

"""Custom mapping from Qwen Hugging Face models to Megatron Core models."""

from .mcore_custom import COL_PARALLEL, ROW_PARALLEL, CustomModuleMapping

qwen3_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens.", COL_PARALLEL),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head.", COL_PARALLEL),
    # Attention
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping("qkv_merging", "model.layers.{}.self_attn.", COL_PARALLEL),
    "linear_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.o_proj.", ROW_PARALLEL
    ),
    "q_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.q_norm."),
    "k_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.k_norm."),
    # MLP
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_merging", "model.layers.{}.mlp.", COL_PARALLEL),
    "linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.down_proj.", ROW_PARALLEL
    ),
    # MoE
    "router": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.gate."),
    "local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_merging", "model.layers.{}.mlp.experts.{}.", COL_PARALLEL
    ),
    "local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping",
        "model.layers.{}.mlp.experts.{}.down_proj.",
        ROW_PARALLEL,
    ),
}


qwen3_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens."),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
    # Attention
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping("qkv_slicing", "model.layers.{}.self_attn."),
    "linear_proj": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.o_proj."),
    "q_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.q_norm."),
    "k_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.k_norm."),
    # MLP
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_slicing", "model.layers.{}.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.down_proj."),
    # MoE
    "router": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.gate."),
    "local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "model.layers.{}.mlp.experts.{}."
    ),
    "local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.experts.{}.down_proj."
    ),
}
