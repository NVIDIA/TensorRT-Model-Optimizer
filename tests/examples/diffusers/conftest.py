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

import pytest


@pytest.fixture(scope="session")
def int8_args():
    # fmt: off
    return [
        "--format", "int8",
        "--calib-size", "8",
        "--collect-method", "min-mean",
        "--percentile", "1.0",
        "--alpha", "0.8",
        "--quant-level", "3.0",
        "--n-steps", "20",
        "--batch-size", "2",
        "--quant-algo", "smoothquant",
    ]
    # fmt: on


@pytest.fixture(scope="session")
def fp8_args():
    # fmt: off
    return [
        "--format", "fp8",
        "--calib-size", "8",
        "--quant-level", "3.0",
        "--n-steps", "20",
        "--batch-size", "2",
    ]
    # fmt: on
