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

from modelopt.torch.quantization.utils import reduce_block_amax


@pytest.mark.parametrize(
    "block_sizes, test_input, expected_scales",
    [
        (
            {-1: 2, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([5.0, 7.0], dtype=torch.bfloat16),
        ),
        (
            {-1: 4, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([7.0, 15.0], dtype=torch.bfloat16),
        ),
        (
            {-2: 4},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([[12.0, 13.0, 14.0, 15.0]], dtype=torch.bfloat16),
        ),
    ],
)
def test_reduce_block_amax(block_sizes, test_input, expected_scales):
    scales = reduce_block_amax(test_input, block_sizes)

    torch.allclose(scales, expected_scales)
