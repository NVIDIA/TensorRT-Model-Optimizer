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

"""Unit tests for hooks."""

import torch
import torch.nn as nn

from modelopt.torch.nas.plugins.hooks import IterativeChannelContributionHook


def test_iterative_channel_contribution_hook():
    """Test IterativeChannelContributionHook returns correct scores after pruning."""
    # Create a simple linear layer
    torch.manual_seed(42)
    linear_layer = nn.Linear(in_features=6, out_features=4, bias=False)

    # Configure hook
    activation_hooks_kwargs = {
        "validation_full_iters": 3,
        "clear_gpu_memory": False,
        "calibration_method": None,
    }

    # Create and register hook
    hook = IterativeChannelContributionHook(linear_layer, activation_hooks_kwargs)
    linear_layer.register_forward_hook(hook)

    # Run forward passes for all pruning iterations
    torch.manual_seed(123)
    for _ in range(activation_hooks_kwargs["validation_full_iters"]):
        activations = torch.randn(2, 3, linear_layer.in_features)
        _ = linear_layer(activations)

    # Get results
    results = hook.to_dict()

    # Verify shapes
    assert results["score"].shape == (6,)
    assert results["channels_importance_ascending"].shape == (6,)

    # Verify exact score values
    expected_scores = torch.tensor([0, 5, 1, 3, 2, 4])
    assert torch.equal(results["score"], expected_scores)

    # Verify exact channels_importance_ascending values
    expected_channels_asc = torch.tensor([0, 2, 4, 3, 5, 1])
    assert torch.equal(results["channels_importance_ascending"], expected_channels_asc)
