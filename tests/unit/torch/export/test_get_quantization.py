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

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import get_quantization_format
from modelopt.torch.export.model_config import QUANTIZATION_FP8, QUANTIZATION_W4A8_AWQ


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        )

    def forward(self, x):
        return self.linears(x)


partial_fp8_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.1.input_quantizer": {"num_bits": (4, 3), "axis": None},
        "default": {"num_bits": 8, "enable": False},
    },
    "algorithm": "max",
}

partial_w4a8_config = {
    "quant_cfg": {
        "*.2.weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*.2.input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "default": {"num_bits": 8, "enable": False},
    },
    "algorithm": "awq_lite",
}


@pytest.mark.parametrize(
    "config, expected",
    [(partial_fp8_config, QUANTIZATION_FP8), (partial_w4a8_config, QUANTIZATION_W4A8_AWQ)],
)
def test_get_quantization_format(config, expected):
    model = ToyModel()
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10)))
    assert get_quantization_format(model) == expected
