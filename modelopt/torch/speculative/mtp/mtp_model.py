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

"""MTP model to support mtp decoding."""

from modelopt.torch.opt.dynamic import DynamicModule


class MTPModel(DynamicModule):
    """Base MTP Model."""

    def _setup(self):
        self._register_temp_attribute("mtp_num_layers", 0)
        self._register_temp_attribute("mtp_num_module", 0)
        self._register_temp_attribute("mtp_freeze_list", [])

    def modify(self, mtp_num_layers, mtp_num_module, mtp_freeze_list, use_last_layernorm):
        """Base MTP Model modify function. Child class should implement the details."""
        self.mtp_num_layers = mtp_num_layers
        self.mtp_num_module = mtp_num_module
        self.mtp_freeze_list = mtp_freeze_list
        self.use_last_layernorm = use_last_layernorm
