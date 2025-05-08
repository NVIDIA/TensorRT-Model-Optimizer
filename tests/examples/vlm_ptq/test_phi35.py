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

import subprocess
from pathlib import Path

import pytest
from _test_utils.examples.run_command import run_vlm_ptq_command

PHI35_PATH = "microsoft/Phi-3.5-vision-instruct"


@pytest.fixture(scope="session", autouse=True)
def install_phi35_requirements():
    subprocess.run(
        ["pip", "install", "-r", "requirements-phi3.5.txt"],
        cwd=Path(__file__).parent.parent.parent.parent / "examples/vlm_ptq",
        check=True,
    )


@pytest.mark.parametrize("quant", ["fp16", "bf16", "int8_sq", "int4_awq"])
def test_phi35(quant):
    run_vlm_ptq_command(model=PHI35_PATH, type="phi", quant=quant, trust_remote_code=True)


@pytest.mark.parametrize("quant", ["fp8", "w4a8_awq"])
def test_phi35_sm89(require_sm89, quant):
    run_vlm_ptq_command(model=PHI35_PATH, type="phi", quant=quant, trust_remote_code=True)
