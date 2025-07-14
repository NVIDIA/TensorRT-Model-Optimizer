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


"""Custom mapping from Nemotron Hugging Face models to Megatron Core models."""

from .mcore_custom import COL_PARALLEL, ROW_PARALLEL, CustomModuleMapping

# Example on adding a new CausalLM.
nemotron_causal_lm_export: dict[str, CustomModuleMapping] = {
    # NemotronForCausalLM is using square-relu where no gated handle is needed.
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
    # NemotronForCausalLM is using square-relu where no gated handle is needed.
    "linear_fc1": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.up_proj."),
    "linear_fc2": CustomModuleMapping("name_remapping", "model.layers.{}.mlp.down_proj."),
    "final_layernorm": CustomModuleMapping("name_remapping", "model.norm."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
}


nemotron_h_causal_lm_import: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "backbone.embeddings.", COL_PARALLEL),
    "final_norm": CustomModuleMapping("name_remapping", "backbone.norm_f."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head.", COL_PARALLEL),
    # Mamba
    "norm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "mixer_norm": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.norm."),
    "A_log": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.A_log"),
    "D": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.D"),
    "dt_bias": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.dt_bias"),
    "conv1d": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.conv1d."),
    "in_proj": CustomModuleMapping(
        "name_remapping", "backbone.layers.{}.mixer.in_proj.", COL_PARALLEL
    ),
    "out_proj": CustomModuleMapping(
        "name_remapping", "backbone.layers.{}.mixer.out_proj.", ROW_PARALLEL
    ),
    # Attention
    "input_layernorm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "linear_qkv": CustomModuleMapping("qkv_merging", "backbone.layers.{}.mixer.", COL_PARALLEL),
    "linear_proj": CustomModuleMapping(
        "name_remapping", "backbone.layers.{}.mixer.o_proj.", ROW_PARALLEL
    ),
    # MLP
    "pre_mlp_layernorm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "linear_fc1": CustomModuleMapping(
        "name_remapping", "backbone.layers.{}.mixer.up_proj.", COL_PARALLEL
    ),
    "linear_fc2": CustomModuleMapping(
        "name_remapping", "backbone.layers.{}.mixer.down_proj.", ROW_PARALLEL
    ),
}


nemotron_h_causal_lm_export: dict[str, CustomModuleMapping] = {
    "word_embeddings": CustomModuleMapping("name_remapping", "backbone.embeddings."),
    "final_norm": CustomModuleMapping("name_remapping", "backbone.norm_f."),
    "output_layer": CustomModuleMapping("name_remapping", "lm_head."),
    # Mamba
    "norm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "mixer_norm": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.norm."),
    "A_log": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.A_log"),
    "D": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.D"),
    "dt_bias": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.dt_bias"),
    "conv1d": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.conv1d."),
    "in_proj": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.in_proj."),
    "out_proj": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.out_proj."),
    # Attention
    "input_layernorm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "linear_qkv": CustomModuleMapping("qkv_slicing", "backbone.layers.{}.mixer."),
    "linear_proj": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.o_proj."),
    # MLP
    "pre_mlp_layernorm": CustomModuleMapping("name_remapping", "backbone.layers.{}.norm."),
    "linear_fc1": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.up_proj."),
    "linear_fc2": CustomModuleMapping("name_remapping", "backbone.layers.{}.mixer.down_proj."),
}
