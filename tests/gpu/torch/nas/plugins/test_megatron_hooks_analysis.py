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

"""Unit tests for megatron hooks analysis tools."""

import pytest
import torch
import torch.nn as nn

from modelopt.torch.nas.plugins.megatron_hooks_analysis import evaluate_importance_scores


def test_evaluate_importance_scores_basic():
    """Test basic functionality of importance score evaluation."""
    torch.manual_seed(42)

    # Create a simple linear layer
    layer = nn.Linear(in_features=100, out_features=50, bias=False)
    activations = torch.randn(4, 8, 100)  # batch=4, seq=8, hidden=100

    # Create importance scores: lower values = less important
    importance_scores = torch.arange(100, dtype=torch.float32)

    # Prune 20% (20 channels with lowest scores: 0-19)
    metrics = evaluate_importance_scores(layer, activations, importance_scores, prune_ratio=0.2)

    # Check values with deterministic seed
    assert metrics["num_pruned"] == 20
    assert metrics["rmse"] == pytest.approx(0.23916207253932953)
    assert metrics["cosine_similarity"] == pytest.approx(0.9114444255828857)
