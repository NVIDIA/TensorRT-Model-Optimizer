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

from modelopt.torch.quantization.utils import (
    convert_quantization_axis_to_reduce_axis,
    reduce_block_amax,
)


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


@pytest.mark.parametrize(
    "shape, quant_axis, expected_reduce_axis",
    [
        ((2, 3, 4), None, None),  # Per-tensor quantization (None axis)
        ((2, 3, 4), 0, [1, 2]),  # Single axis cases
        ((2, 3, 4), -2, [0, 2]),  # Negative indices
        ((2, 3, 4, 5), [0, -1], [1, 2]),  # Multiple axes
        ((2, 3, 4), (0, 2), [1]),  # Tuple instead of list
    ],
)
def test_convert_quantization_axis_to_reduce_axis(shape, quant_axis, expected_reduce_axis):
    """Test converting quantization axes to reduction axes."""
    # Create a tensor with the specified shape
    input_tensor = torch.randn(shape)

    # Convert quantization axis to reduce axis
    result = convert_quantization_axis_to_reduce_axis(input_tensor, quant_axis)

    # Check if the result matches expected
    assert result == expected_reduce_axis, (
        f"For shape {shape} and quant_axis {quant_axis}, expected {expected_reduce_axis} but got {result}"
    )

    # Additional sanity check: if we sum-reduce along the returned axes,
    # we should get a tensor with the same shape as if we kept the quantization axes
    if result is not None and len(result) > 0:
        # Create a ones tensor for easy shape verification
        ones = torch.ones(shape)

        # Reduce sum across the returned axes
        reduced = ones
        for axis in sorted(
            result, reverse=True
        ):  # Reduce from highest dim to avoid changing indices
            reduced = reduced.sum(dim=axis, keepdim=True)

        # Build expected shape after reduction
        expected_shape = list(shape)
        for axis in sorted(result, reverse=True):
            expected_shape[axis] = 1

        assert reduced.shape == tuple(expected_shape), (
            f"Reduction result shape {reduced.shape} doesn't match expected {tuple(expected_shape)}"
        )
