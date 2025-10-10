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

"""Test sparse attention configuration validation."""

import pytest
from pydantic import ValidationError

from modelopt.torch.sparsity.attention_sparsity.config import (
    SKIP_SOFTMAX_DEFAULT,
    FlashSoftmaxSkipConfig,
    SparseAttentionAttributeConfig,
    SparseAttentionConfig,
)


class TestSparseAttentionAttributeConfig:
    """Test SparseAttentionAttributeConfig validators."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = SparseAttentionAttributeConfig(
            method="flash_softmax_skip",
            threshold=1e-4,
            br=128,
            bc=128,
            enable=True,
        )
        assert config.method == "flash_softmax_skip"
        assert config.threshold == 1e-4
        assert config.br == 128
        assert config.bc == 128

    def test_method_validation(self):
        """Test method must be string."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            SparseAttentionAttributeConfig(method=123)

    def test_block_size_validation_negative(self):
        """Test block sizes must be positive."""
        with pytest.raises(ValidationError, match="Block size must be positive"):
            SparseAttentionAttributeConfig(br=-1)

        with pytest.raises(ValidationError, match="Block size must be positive"):
            SparseAttentionAttributeConfig(bc=0)

    def test_block_size_validation_large(self):
        """Test that large block sizes are accepted."""
        # Large block sizes are allowed (warning removed for simplicity)
        config = SparseAttentionAttributeConfig(br=2048)
        assert config.br == 2048

    def test_threshold_validation_range(self):
        """Test threshold must be in range (0, 1)."""
        with pytest.raises(ValidationError, match="Threshold must be in range"):
            SparseAttentionAttributeConfig(threshold=0)

        with pytest.raises(ValidationError, match="Threshold must be in range"):
            SparseAttentionAttributeConfig(threshold=-0.1)

        with pytest.raises(ValidationError, match="Threshold must be in range"):
            SparseAttentionAttributeConfig(threshold=1.0)

        with pytest.raises(ValidationError, match="Threshold must be in range"):
            SparseAttentionAttributeConfig(threshold=1.5)

    def test_threshold_validation_dict(self):
        """Test threshold dict validation."""
        # Valid phase-aware threshold
        config = SparseAttentionAttributeConfig(threshold={"prefill": 1e-3, "decode": 1e-5})
        assert config.threshold == {"prefill": 1e-3, "decode": 1e-5}

        # Invalid phase key
        with pytest.raises(ValidationError, match="Invalid threshold phases"):
            SparseAttentionAttributeConfig(threshold={"invalid_phase": 1e-3})

        # Invalid threshold value in dict (negative)
        with pytest.raises(ValidationError, match="must be in range"):
            SparseAttentionAttributeConfig(threshold={"prefill": -1e-3})

        # Invalid threshold value in dict (>= 1.0)
        with pytest.raises(ValidationError, match="must be in range"):
            SparseAttentionAttributeConfig(threshold={"prefill": 1.0})

    def test_threshold_validation_type(self):
        """Test threshold type validation."""
        with pytest.raises(ValidationError, match="Input should be a valid"):
            SparseAttentionAttributeConfig(threshold="invalid")


class TestSparseAttentionConfig:
    """Test SparseAttentionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SparseAttentionConfig()
        assert config.method == "flash_softmax_skip"
        assert config.collect_stats is False

    def test_predefined_config(self):
        """Test pre-defined configuration."""
        assert "method" in SKIP_SOFTMAX_DEFAULT
        assert "sparse_cfg" in SKIP_SOFTMAX_DEFAULT
        assert "*attn*" in SKIP_SOFTMAX_DEFAULT["sparse_cfg"]


class TestFlashSoftmaxSkipConfig:
    """Test FlashSoftmaxSkipConfig."""

    def test_default_values(self):
        """Test default values for flash_softmax_skip config."""
        config = FlashSoftmaxSkipConfig()
        assert config.method == "flash_softmax_skip"
        assert "*attention*" in config.sparse_cfg
