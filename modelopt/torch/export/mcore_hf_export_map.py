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


"""Custom mapping from Megatron Core models to their Hugging Face counter part."""

import copy
from typing import Any

COL_PARALLEL = {"sharding_dim": 0}
ROW_PARALLEL = {"sharding_dim": 1}


class CustomModuleMapping:
    """A custom module mapping from Megatron Core to its HF counter part."""

    def __init__(
        self, func_name: str = "", target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}
    ):
        """Create a custom module mapping."""
        self.func_name = func_name
        self.target_name_or_prefix = target_name_or_prefix
        self.func_kwargs = func_kwargs


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


# Example on adding a new CausalLM.
nemotron_causal_lm_export_delta: dict[str, CustomModuleMapping] = {
    # NemotronForCausalLM is using square-relu where no gated handle is needed.
    "linear_fc1": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.up_proj."),
    # EagleForCausalLM support
    "eagle_module.linear_fc1": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.mlp.up_proj."
    ),
}
# Copy the Llama rule book and overwrite the delta
nemotron_causal_lm_export = copy.deepcopy(llama_causal_lm_export)
nemotron_causal_lm_export.update(nemotron_causal_lm_export_delta)

deepseek_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens."),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
    # Multi-Latent Attention (V3 has lora on q as well)
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_q_proj": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.q_proj."),
    "linear_q_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_proj."
    ),
    "linear_q_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_layernorm."
    ),
    "linear_q_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_b_proj."
    ),
    "linear_kv_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_proj_with_mqa."
    ),
    "linear_kv_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_layernorm."
    ),
    "linear_kv_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_b_proj."
    ),
    "linear_proj": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    # MLP for dense layers
    "linear_fc1": CustomModuleMapping("gated_mlp_slicing", "model.layers.{}.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.down_proj."),
    # MoE shared experts
    "router": CustomModuleMapping(
        "name_remapping",
        "model.layers.{}.mlp.gate.",
        {"mapping": {"expert_bias": "e_score_correction_bias"}},
    ),
    "shared_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "model.layers.{}.mlp.shared_experts."
    ),
    "shared_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.shared_experts.down_proj."
    ),
    # MoE local experts
    "local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "model.layers.{}.mlp.experts.{}."
    ),
    "local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.experts.{}.down_proj."
    ),
    # MedusaForCausalLM support
    "medusa_heads.lm_head": CustomModuleMapping(
        "name_remapping", "medusa_heads.{}.1."
    ),  # TODO: lm_head is hardcoded to .1 as currently only support using 1 layer in medusa head
    # needs a fix
    "medusa_heads.medusa_layers.linear": CustomModuleMapping(
        "name_remapping", "medusa_heads.{}.{}.linear."
    ),
    # EagleForCausalLM support
    "eagle_module.fc": CustomModuleMapping("name_remapping", "eagle_module.fc."),
    "eagle_module.input_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.input_layernorm."
    ),
    "eagle_module.linear_q_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.q_proj."
    ),
    "eagle_module.linear_q_down_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.q_a_proj."
    ),
    "eagle_module.linear_q_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.q_a_layernorm."
    ),
    "eagle_module.linear_q_up_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.q_b_proj."
    ),
    "eagle_module.linear_kv_down_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.kv_a_proj_with_mqa."
    ),
    "eagle_module.linear_kv_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.kv_a_layernorm."
    ),
    "eagle_module.linear_kv_up_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.kv_b_proj."
    ),
    "eagle_module.linear_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.o_proj."
    ),
    "eagle_module.pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.post_attention_layernorm."
    ),
    "eagle_module.router": CustomModuleMapping(
        "name_remapping",
        "eagle_module.layers.{}.mlp.gate.",
        {"mapping": {"expert_bias": "e_score_correction_bias"}},
    ),
    "eagle_module.shared_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "eagle_module.layers.{}.mlp.shared_experts."
    ),
    "eagle_module.shared_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.mlp.shared_experts.down_proj."
    ),
    "eagle_module.local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "eagle_module.layers.{}.mlp.experts.{}."
    ),
    "eagle_module.local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.mlp.experts.{}.down_proj."
    ),
    "eagle_module.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "eagle_module.layers.{}.mlp."
    ),
    "eagle_module.linear_fc2": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.mlp.down_proj."
    ),
    # MTPCausalLM support
    "mtp.fc": CustomModuleMapping("name_remapping", "mtp.{}.fc."),
    "mtp.enorm": CustomModuleMapping("name_remapping", "mtp.{}.enorm."),
    "mtp.hnorm": CustomModuleMapping("name_remapping", "mtp.{}.hnorm."),
    "mtp.input_layernorm": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.input_layernorm."
    ),
    "mtp.linear_q_proj": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.q_proj."
    ),
    "mtp.linear_q_down_proj": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.q_a_proj."
    ),
    "mtp.linear_q_layernorm": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.q_a_layernorm."
    ),
    "mtp.linear_q_up_proj": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.q_b_proj."
    ),
    "mtp.linear_kv_down_proj": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.kv_a_proj_with_mqa."
    ),
    "mtp.linear_kv_layernorm": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.kv_a_layernorm."
    ),
    "mtp.linear_kv_up_proj": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.self_attn.kv_b_proj."
    ),
    "mtp.linear_proj": CustomModuleMapping("name_remapping", "mtp.{}.layers.{}.self_attn.o_proj."),
    "mtp.pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.post_attention_layernorm."
    ),
    "mtp.router": CustomModuleMapping(
        "name_remapping",
        "mtp.{}.layers.{}.mlp.gate.",
        {"mapping": {"expert_bias": "e_score_correction_bias"}},
    ),
    "mtp.shared_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "mtp.{}.layers.{}.mlp.shared_experts."
    ),
    "mtp.shared_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.mlp.shared_experts.down_proj."
    ),
    "mtp.local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "mtp.{}.layers.{}.mlp.experts.{}."
    ),
    "mtp.local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "mtp.{}.layers.{}.mlp.experts.{}.down_proj."
    ),
}


all_mcore_hf_export_mapping: dict[str, Any] = {
    "DeepseekV2ForCausalLM": deepseek_causal_lm_export,
    "DeepseekV3ForCausalLM": deepseek_causal_lm_export,
    "LlamaForCausalLM": llama_causal_lm_export,
    "Llama4ForConditionalGeneration": {},
    "NemotronForCausalLM": nemotron_causal_lm_export,
    "LlamaForCausalLMEagle": eagle_llama_causal_lm_export,
    "LlamaForCausalLMEagle3": eagle3_llama_causal_lm_export,
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

deepseek_v3_causal_lm_import = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens.", COL_PARALLEL),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head.", COL_PARALLEL),
    # Per-layer
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_q_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_proj.", COL_PARALLEL
    ),
    "linear_q_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_proj.", COL_PARALLEL
    ),
    "linear_q_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_layernorm."
    ),
    "linear_q_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_b_proj.", COL_PARALLEL
    ),
    "linear_kv_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_proj_with_mqa.", COL_PARALLEL
    ),
    "linear_kv_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_layernorm."
    ),
    "linear_kv_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_b_proj.", COL_PARALLEL
    ),
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
    # MoE shared experts
    "router": CustomModuleMapping(
        "name_remapping",
        "model.layers.{}.mlp.gate.",
        {"mapping": {"expert_bias": "e_score_correction_bias"}},
    ),
    "shared_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_merging", "model.layers.{}.mlp.shared_experts.", COL_PARALLEL
    ),
    "shared_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.shared_experts.down_proj.", ROW_PARALLEL
    ),
    # MoE local experts
    "local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_merging", "model.layers.{}.mlp.experts.{}.", COL_PARALLEL
    ),
    "local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.experts.{}.down_proj.", ROW_PARALLEL
    ),
    # MTP
    "mtp.fc": CustomModuleMapping("name_remapping", "model.layers.{}.eh_proj."),
    "mtp.enorm": CustomModuleMapping("name_remapping", "model.layers.{}.enorm."),
    "mtp.hnorm": CustomModuleMapping("name_remapping", "model.layers.{}.hnorm."),
    "mtp.input_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.input_layernorm."
    ),
    "mtp.linear_q_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_proj.", COL_PARALLEL
    ),
    "mtp.linear_q_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_proj.", COL_PARALLEL
    ),
    "mtp.linear_q_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_a_layernorm."
    ),
    "mtp.linear_q_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.q_b_proj.", COL_PARALLEL
    ),
    "mtp.linear_kv_down_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_proj_with_mqa.", COL_PARALLEL
    ),
    "mtp.linear_kv_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_a_layernorm."
    ),
    "mtp.linear_kv_up_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.kv_b_proj.", COL_PARALLEL
    ),
    "mtp.linear_proj": CustomModuleMapping(
        "name_remapping", "model.layers.{}.self_attn.o_proj.", ROW_PARALLEL
    ),
    "mtp.pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "mtp.router": CustomModuleMapping(
        "name_remapping",
        "model.layers.{}.mlp.gate.",
        {"mapping": {"expert_bias": "e_score_correction_bias"}},
    ),
    "mtp.shared_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_merging", "model.layers.{}.mlp.shared_experts.", COL_PARALLEL
    ),
    "mtp.shared_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.shared_experts.down_proj.", ROW_PARALLEL
    ),
    "mtp.local_experts.linear_fc1": CustomModuleMapping(
        "gated_mlp_merging", "model.layers.{}.mlp.experts.{}.", COL_PARALLEL
    ),
    "mtp.local_experts.linear_fc2": CustomModuleMapping(
        "name_remapping", "model.layers.{}.mlp.experts.{}.down_proj.", ROW_PARALLEL
    ),
}

all_mcore_hf_import_mapping: dict[str, Any] = {
    "LlamaForCausalLM": llama_causal_lm_import,
    "DeepseekV2ForCausalLM": deepseek_v3_causal_lm_import,
    "DeepseekV3ForCausalLM": deepseek_v3_causal_lm_import,
}
