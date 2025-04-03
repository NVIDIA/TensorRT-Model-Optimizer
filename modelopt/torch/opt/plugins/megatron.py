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
"""Support quantization and save/resore for Megatron."""

import megatron.core.transformer.mlp as megatron_mlp
import regex as re

from ..dynamic import DynamicModule


class _MegatronMLP(DynamicModule):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`.

    See https://github.com/NVIDIA/Megatron-LM/blob/0657a52d5a5bfea3f74bcc19eb620211b71f8671/megatron/core/transformer/mlp.py#L151.
    """

    _modelopt_state_keys = []

    def _setup(self):
        pass

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if not self.config.gated_linear_unit:
            return sharded_state_dict
        for k, v in sharded_state_dict.items():
            if "linear_fc1" in k and any(
                [re.compile(pattern).match(k) for pattern in self._modelopt_state_keys]
            ):
                sharded_state_dict[k] = megatron_mlp.apply_swiglu_sharded_factory(
                    v, sharded_offsets
                )
        return sharded_state_dict
