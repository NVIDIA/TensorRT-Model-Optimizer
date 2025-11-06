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

"""Unit tests for sparse attention export configuration extraction."""

from _test_utils.torch_sparsity.sparse_attention_common import (
    FLASH_SOFTMAX_SKIP_DEFAULT_CFG,
    FLASH_SOFTMAX_SKIP_PHASE_AWARE_CFG,
    SimpleTransformerEncoderLayer,
    sparsify_model_and_forward,
)

import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.export.unified_export_hf import _get_sparse_attention_config


class TestSparseAttentionConfigExtraction:
    """Test sparse attention config extraction for export."""

    def test_extract_non_calibrated_config(self):
        """Test extraction of non-calibrated sparse attention config."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention without calibration
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Verify structure
        assert "config_groups" in extracted_config
        assert "producer" in extracted_config
        assert extracted_config["producer"]["name"] == "modelopt"

        # Should NOT have calibration parameters
        assert "threshold_scale_factor" not in extracted_config
        assert "target_sparsity" not in extracted_config

        # Check config_groups
        groups = extracted_config["config_groups"]
        assert len(groups) > 0

        # Verify first group has sparse_algo and threshold
        group_0 = groups["group_0"]
        assert "sparse_algo" in group_0
        assert group_0["sparse_algo"] == "softmax_skip"
        assert "threshold" in group_0
        assert group_0["threshold"] == 1e-4
        assert "targets" in group_0
        assert isinstance(group_0["targets"], list)
        assert len(group_0["targets"]) > 0

    def test_extract_phase_aware_config(self):
        """Test extraction of phase-aware threshold config."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention with phase-aware thresholds
        sparse_model = sparsify_model_and_forward(
            model, FLASH_SOFTMAX_SKIP_PHASE_AWARE_CFG, calib_data
        )

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Check config_groups have phase-aware threshold
        groups = extracted_config["config_groups"]
        group_0 = groups["group_0"]
        assert "threshold" in group_0
        threshold = group_0["threshold"]
        assert isinstance(threshold, dict)
        assert "prefill" in threshold
        assert "decode" in threshold
        assert threshold["prefill"] == 1e-3
        assert threshold["decode"] == 1e-5

    def test_extract_empty_config_no_sparse_modules(self):
        """Test extraction returns empty dict when no sparse modules."""
        model = SimpleTransformerEncoderLayer()

        # Don't apply sparse attention
        extracted_config = _get_sparse_attention_config(model)

        # Should return empty dict
        assert extracted_config == {}

    def test_extract_empty_config_all_disabled(self):
        """Test extraction returns empty dict when all modules disabled."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Disable all modules
        sparse_attn.disable_sparse_attention(sparse_model, "*")

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Should return empty dict
        assert extracted_config == {}

    def test_sparse_algo_name_extraction(self):
        """Test that sparse_algo is correctly extracted from method name."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Verify sparse_algo is "softmax_skip" (stripped "flash_" prefix)
        groups = extracted_config["config_groups"]
        for group_config in groups.values():
            assert group_config["sparse_algo"] == "softmax_skip"

    def test_mock_calibrated_config(self):
        """Test extraction with mock calibrated parameters."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Manually set calibration parameters on method instances
        # (simulating what calibration does)
        for module in sparse_model.modules():
            if hasattr(module, "_sparse_method_instance"):
                module._sparse_method_instance.threshold_scale_factor = 0.00123456
                module._sparse_method_instance.target_sparsity = 0.5

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Verify global calibration parameters
        assert "threshold_scale_factor" in extracted_config
        assert "target_sparsity" in extracted_config
        assert extracted_config["threshold_scale_factor"] == 0.00123456
        assert extracted_config["target_sparsity"] == 0.5

        # Verify config_groups do NOT have threshold field (calibrated)
        groups = extracted_config["config_groups"]
        for group_config in groups.values():
            assert "sparse_algo" in group_config
            assert "threshold" not in group_config
            assert "targets" in group_config

    def test_producer_metadata(self):
        """Test that producer metadata is correctly added."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Verify producer metadata
        assert "producer" in extracted_config
        producer = extracted_config["producer"]
        assert "name" in producer
        assert "version" in producer
        assert producer["name"] == "modelopt"
        assert isinstance(producer["version"], str)
        assert len(producer["version"]) > 0

    def test_targets_in_config(self):
        """Test that targets field contains module class names."""
        model = SimpleTransformerEncoderLayer()
        calib_data = [model.get_input() for _ in range(2)]

        # Apply sparse attention
        sparse_model = sparsify_model_and_forward(model, FLASH_SOFTMAX_SKIP_DEFAULT_CFG, calib_data)

        # Extract config
        extracted_config = _get_sparse_attention_config(sparse_model)

        # Verify targets contain class names
        groups = extracted_config["config_groups"]
        for group_config in groups.values():
            assert "targets" in group_config
            targets = group_config["targets"]
            assert isinstance(targets, list)
            assert len(targets) > 0
            # Should contain attention class names
            assert all(isinstance(t, str) for t in targets)
