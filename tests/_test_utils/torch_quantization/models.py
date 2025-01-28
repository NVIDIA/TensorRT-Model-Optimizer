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

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization.nn import QuantConv2d, QuantLinear


class QuantConvLinear(nn.Module):
    def __init__(self):
        super(QuantConvLinear, self).__init__()
        self.conv1 = QuantConv2d(1, 8, kernel_size=3)
        self.conv2 = QuantConv2d(8, 4, kernel_size=3)
        self.fc1 = QuantLinear(64, 8)
        self.fc2 = QuantLinear(8, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_input(self):
        return torch.randn(2, 1, 8, 8)


class SimpleLinear(nn.Module):
    """Test Linear model for ONNX export."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16)
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def get_input(cls):
        return torch.randn(2, 16)


class SimpleConv(nn.Module):
    """Test Conv model for ONNX export."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def get_input(cls):
        return torch.randn(2, 16, 4, 4)


class SimpleConvLinear(nn.Module):
    """Test ConvLinear model for ONNX export."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def get_input(cls):
        return torch.randn(4, 16, 4, 4)


class RegularQuantModelForTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = QuantLinear(128, 256)
        self.activation = nn.ReLU()
        self.fc2 = QuantLinear(256, 128)

    def forward(self, x):
        for block in [self.fc1, self.activation, self.fc2]:
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
        return x
