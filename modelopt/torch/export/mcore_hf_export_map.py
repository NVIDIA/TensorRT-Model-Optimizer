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
from typing import Any, Dict


class CustomModuleMapping:
    """A custom module mapping from Megatron Core to its HF counter part."""

    def __init__(
        self, func_name: str = "", target_name_or_prefix: str = "", func_kwargs: Dict[str, Any] = {}
    ):
        """Create a custom module mapping."""
        self.func_name = func_name
        self.target_name_or_prefix = target_name_or_prefix
        self.func_kwargs = func_kwargs


gpt_model_export: Dict[str, CustomModuleMapping] = {
    # MCore GPTModel
    "word_embeddings": CustomModuleMapping("name_remapping", "embedding.word_embeddings."),
    "input_layernorm": CustomModuleMapping("name_remapping", "decoder.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping(
        "name_remapping",
        "decoder.layers.{}.self_attention.linear_qkv.",
        {"skip_output_scale": False},
    ),
    "linear_proj": CustomModuleMapping(
        "name_remapping", "decoder.layers.{}.self_attention.linear_proj."
    ),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "decoder.layers.{}.pre_mlp_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("name_remapping", "decoder.layers.{}.mlp.linear_fc1."),
    "linear_fc2": CustomModuleMapping("name_remapping", "decoder.layers.{}.mlp.linear_fc2."),
    "final_layernorm": CustomModuleMapping("name_remapping", "decoder.final_layernorm."),
    "output_layer": CustomModuleMapping("name_remapping", "output_layer."),
    # ModelOpt MedusaGPTModel support
    "medusa_heads.lm_head": CustomModuleMapping("name_remapping", "medusa_heads.{}.1."),
    "medusa_heads.medusa_layers.linear": CustomModuleMapping(
        "name_remapping", "medusa_heads.{}.{}.linear."
    ),
    # ModelOpt EagleGPTModel support
    "eagle_module.fc": CustomModuleMapping("name_remapping", "eagle_module.fc."),
    "eagle_module.input_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.decoder.layers.{}.input_layernorm."
    ),
    "eagle_module.linear_qkv": CustomModuleMapping(
        "name_remapping",
        "eagle_module.decoder.layers.{}.self_attention.linear_qkv.",
        {"skip_output_scale": False},
    ),
    "eagle_module.linear_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.decoder.layers.{}.self_attention.linear_proj."
    ),
    "eagle_module.pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.decoder.layers.{}.pre_mlp_layernorm."
    ),
    "eagle_module.linear_fc1": CustomModuleMapping(
        "name_remapping", "eagle_module.decoder.layers.{}.mlp.linear_fc1."
    ),
    "eagle_module.linear_fc2": CustomModuleMapping(
        "name_remapping", "eagle_module.decoder.layers.{}.mlp.linear_fc2."
    ),
}


llama_causal_lm_export: Dict[str, CustomModuleMapping] = {
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
    # MedusaForCausalLM support
    "medusa_heads.lm_head": CustomModuleMapping("name_remapping", "medusa_heads.{}.1."),
    "medusa_heads.medusa_layers.linear": CustomModuleMapping(
        "name_remapping", "medusa_heads.{}.{}.linear."
    ),
    # EagleForCausalLM support
    "eagle_module.fc": CustomModuleMapping("name_remapping", "eagle_module.fc."),
    "eagle_module.input_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.input_layernorm."
    ),
    "eagle_module.linear_qkv": CustomModuleMapping(
        "qkv_slicing",
        "eagle_module.layers.{}.self_attn.",
    ),
    "eagle_module.linear_proj": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.self_attn.o_proj."
    ),
    "eagle_module.pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.post_attention_layernorm."
    ),
    "eagle_module.linear_fc1": CustomModuleMapping(
        "gated_mlp_slicing", "eagle_module.layers.{}.mlp."
    ),
    "eagle_module.linear_fc2": CustomModuleMapping(
        "name_remapping", "eagle_module.layers.{}.mlp.down_proj."
    ),
}

# Example on adding a new CausalLM.
nemotron_causal_lm_export_delta: Dict[str, CustomModuleMapping] = {
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

all_mcore_hf_export_mapping: Dict[str, Any] = {
    "GPTModel": gpt_model_export,
    "LlamaForCausalLM": llama_causal_lm_export,
    "NemotronForCausalLM": nemotron_causal_lm_export,
}


llama_causal_lm_import: Dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "model.embed_tokens."),
    "input_layernorm": CustomModuleMapping("name_remapping", "model.layers.{}.input_layernorm."),
    "linear_qkv": CustomModuleMapping("qkv_merging", "model.layers.{}.self_attn."),
    "linear_proj": CustomModuleMapping("name_remapping", "model.layers.{}.self_attn.o_proj."),
    "pre_mlp_layernorm": CustomModuleMapping(
        "name_remapping", "model.layers.{}.post_attention_layernorm."
    ),
    "linear_fc1": CustomModuleMapping("gated_mlp_merging", "model.layers.{}.mlp."),
    "linear_fc2": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.down_proj."),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
}

all_mcore_hf_import_mapping: Dict[str, Any] = {
    "LlamaForCausalLM": llama_causal_lm_import,
}
