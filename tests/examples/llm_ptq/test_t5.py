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
from _test_utils.examples.run_command import run_llm_ptq_command
from _test_utils.model import T5_PATH
from _test_utils.torch_misc import minimum_sm


@pytest.fixture(scope="session", autouse=True)
def install_t5_requirements():
    subprocess.run(
        ["pip", "install", "-r", "requirements-t5.txt"],
        cwd=Path(__file__).parent.parent.parent.parent / "examples/llm_ptq",
        check=True,
    )


@pytest.mark.parametrize(("quant", "export_fmt"), [("fp16", "tensorrt_llm")])
def test_t5(quant, export_fmt):
    run_llm_ptq_command(model=T5_PATH, quant=quant, export_fmt=export_fmt)


@minimum_sm(89)
@pytest.mark.parametrize(("quant", "export_fmt"), [("fp8", "tensorrt_llm")])
def test_t5_sm89(quant, export_fmt):
    run_llm_ptq_command(model=T5_PATH, quant=quant, export_fmt=export_fmt)
