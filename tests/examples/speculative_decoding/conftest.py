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

import os

import pytest
from _test_utils.examples.run_command import MODELOPT_ROOT, run_example_command


@pytest.fixture(scope="session", autouse=True)
def daring_anteater_path():
    dataset_path = MODELOPT_ROOT / "examples/speculative_decoding/Daring-Anteater"
    if not os.path.exists(dataset_path):
        run_example_command(
            ["git", "clone", "https://huggingface.co/datasets/nvidia/Daring-Anteater"],
            "speculative_decoding",
        )
    return dataset_path / "train.jsonl"
