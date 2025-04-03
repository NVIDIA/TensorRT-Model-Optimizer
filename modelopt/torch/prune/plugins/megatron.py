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

"""Plugins for pruning for Megatron-Core / NeMo modules using Minitron algorithm."""

# import nas plugin to check if it is enabled else raises an Exception
from modelopt.torch.nas.plugins.megatron import *  # noqa: F403

from ..config import MCoreGPTMinitronConfig

MCoreGPTMinitronConfig.register_default(
    {
        "megatron.core.models.gpt.GPTModel": {
            "hidden_size_divisor": 64,
            "num_heads_per_group_divisor": 1,
            "num_query_groups_divisor": 1,
            "ffn_hidden_size_divisor": 64,
        },
    }
)
