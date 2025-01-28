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
import torch
from torchvision.models import resnet18

from modelopt.torch._deploy.utils import get_onnx_bytes


@pytest.fixture(scope="session")
def resnet18_model():
    return resnet18()


@pytest.fixture(scope="session")
def resnet18_onnx_bytes(resnet18_model):
    return get_onnx_bytes(resnet18_model, torch.rand(1, 3, 224, 224))
