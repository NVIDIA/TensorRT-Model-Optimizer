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
