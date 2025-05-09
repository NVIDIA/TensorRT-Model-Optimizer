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

import copy

from packaging.version import Version


def format_modelopt_checkpoint_by_version(modelopt_state: dict, version: str):
    if Version(version) >= Version("0.29"):
        return modelopt_state
    modelopt_state = copy.deepcopy(modelopt_state)
    modelopt_state["modelopt_version"] = version
    for mode, state in modelopt_state["modelopt_state_dict"]:
        if "quantizer_state" not in state["metadata"]:
            continue
        for quantizer_name, quantizer_state in state["metadata"]["quantizer_state"].items():
            quantizer_state["_mopt_ckpt_versn"] = version
            pyt_states = quantizer_state.pop("_pytorch_state_metadata", None)
            if pyt_states is None:
                continue
            for k, v in pyt_states["buffers"].items():
                quantizer_state["_has" + k] = True
    return modelopt_state
