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

"""Configurations for speculative decoding modes."""

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

EAGLE1_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_num_layers": 1,
        "eagle_hidden_state_distillation": False,
        "eagle_disable_moe": True,
        "use_input_layernorm_in_first_layer": False,
        "use_last_layernorm": False,
        "use_mtp_layernorm": False,
    },
}

EAGLE3_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_num_layers": 1,
        "eagle_hidden_state_distillation": False,
        "eagle_disable_moe": True,
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": [],
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": True,
        "use_mtp_layernorm": False,
    },
}

EAGLE_MTP_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_num_layers": 1,
        "eagle_hidden_state_distillation": False,
        "eagle_disable_moe": False,
        "use_aux_hidden_state": False,
        "eagle_aux_hidden_state_layer_ids": [],
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": True,
        "use_mtp_layernorm": True,
    },
}

MTP_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_num_layers": 1,
        "eagle_hidden_state_distillation": False,
        "eagle_disable_moe": False,
        "use_input_layernorm_in_first_layer": True,
        "use_last_layernorm": False,
        "use_mtp_layernorm": True,
    },
}


class MedusaConfig(ModeloptBaseConfig):
    """Medusa config."""

    medusa_num_heads: int = ModeloptField(
        default=2,
        description=("The number of medusa heads added to the model."),
    )

    medusa_num_layers: int = ModeloptField(
        default=1,
        description=("The number of ResBlocks used in medusa head."),
    )


class EagleConfig(ModeloptBaseConfig):
    """Eagle config."""

    eagle_num_layers: int = ModeloptField(
        default=1,
        description=("The number of decoder used in the eagle model."),
    )

    use_input_layernorm_in_first_layer: bool = ModeloptField(
        default=True, description=("Whether to use input_layernorm in the first decoder layer.")
    )

    use_last_layernorm: bool = ModeloptField(
        default=False, description=("Whether to use a final layernorm before lm_head.")
    )

    eagle_hidden_state_distillation: bool = ModeloptField(
        default=False, description=("Whether to use feature hidden states distillation.")
    )

    use_aux_hidden_state: bool = ModeloptField(
        default=False, description=("Whether to use aux hidden state (EAGLE-3).")
    )

    eagle_aux_hidden_state_layer_ids: list = ModeloptField(
        default=[],
        description=("The list of aux hidden state layers used in EAGLE-3."),
    )

    eagle_disable_moe: bool = ModeloptField(
        default=False, description=("Whether to disable MoE in eagle module.")
    )

    draft_vocab_size: int = ModeloptField(
        default=0,
        description=("The vocab size of the eagle module. 0 means the same as base model."),
    )

    use_mtp_layernorm: bool = ModeloptField(
        default=False,
        description=(
            "Whether to use norms before input_hidden_states and embedding in eagle module."
        ),
    )

    ffn_hidden_size: int = ModeloptField(
        default=0,
        description=(
            "ffn_hidden_size of the eagle module. Using base model's ffn_hidden_size is set to 0."
        ),
    )


class MTPConfig(ModeloptBaseConfig):
    """MTP config."""

    mtp_num_layers: int = ModeloptField(
        default=1,
        description=("The number of decoder used in the mtp model."),
    )

    mtp_num_module: int = ModeloptField(
        default=1,
        description=("The number of mtp used in the model."),
    )

    mtp_freeze_list: list = ModeloptField(
        default=[],
        description=("The list of mtp module to freeze."),
    )

    use_last_layernorm: bool = ModeloptField(
        default=False, description=("Whether to use a final layernorm before lm_head.")
    )
