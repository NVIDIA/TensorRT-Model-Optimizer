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


"""Custom mapping from Llama Hugging Face models to Megatron Core models."""

from .mcore_custom import COL_PARALLEL, ROW_PARALLEL, CustomModuleMapping

llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens."),
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping(
        "qkv_slicing",
        "model.layers.{}.self_attn.",
    ),
    "linear_proj": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_slicing", "model.layers.{}.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.down_proj."),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
}

medusa_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    # MedusaForCausalLM support
    "lm_head": CustomModuleMapping(
        "name_remapping", "medusa_heads.{}.1."
    ),  # TODO: lm_head is hardcoded to .1 as currently only support using 1 layer in medusa head
    # needs a fix
    "linear": CustomModuleMapping("name_remapping", "medusa_heads.{}.{}.linear."),
}

eagle_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "embed_tokens."),
    "enorm": CustomModuleMapping("name_remapping", "enorm."),
    "hnorm": CustomModuleMapping("name_remapping", "hnorm."),
    "fc": CustomModuleMapping("name_remapping", "fc."),
    "input_layernorm": CustomModuleMapping("name_remapping", "layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping("qkv_slicing", "layers.{}.self_attn."),
    "linear_proj": CustomModuleMapping("name_remapping", "layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_slicing", "layers.{}.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "layers.{}.mlp.down_proj."),
    "final_layernorm": CustomModuleMapping("name_remapping", "norm."),
    "d2t": CustomModuleMapping("name_remapping", "d2t"),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
}

eagle3_llama_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "embed_tokens."),
    "enorm": CustomModuleMapping("name_remapping", "midlayer.input_layernorm."),
    "fc": CustomModuleMapping("name_remapping", "fc."),
    "input_layernorm": CustomModuleMapping("name_remapping", "midlayer.hidden_norm."),
    "linear_qkv": CustomModuleMapping("qkv_slicing", "midlayer.self_attn."),
    "linear_proj": CustomModuleMapping("name_remapping", "midlayer.self_attn.o_proj."),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "midlayer.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_slicing", "midlayer.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "midlayer.mlp.down_proj."),
    "final_layernorm": CustomModuleMapping("name_remapping", "norm."),
    "d2t": CustomModuleMapping("name_remapping", "d2t"),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
}


llama_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens.", COL_PARALLEL),
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping("qkv_merging", "model.layers.{}.self_attn.", COL_PARALLEL),
    "linear_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.o_proj.", ROW_PARALLEL
    ),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_merging", "model.layers.{}.mlp.", COL_PARALLEL),
    "linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.down_proj.", ROW_PARALLEL
    ),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head.", COL_PARALLEL),
}
