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

"""Redrafter model to support redrafter decoding."""

from modelopt.torch.opt.dynamic import DynamicModule


class RedrafterModel(DynamicModule):
    """Base Redrafter Model."""

    def _setup(self):
        self._register_temp_attribute("redrafter_predict_n_tokens", 0)
        self._register_temp_attribute("redrafter_num_layers", 0)
        self._register_temp_attribute("drafter", None)

    def modify(self, redrafter_predict_n_tokens=0, redrafter_num_layers=0):
        """Base Redrafter Model modify function. Child class should implement the details."""
        self.redrafter_predict_n_tokens = redrafter_predict_n_tokens
        self.redrafter_num_layers = redrafter_num_layers
