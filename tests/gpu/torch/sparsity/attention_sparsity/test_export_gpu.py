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

"""GPU tests for sparse attention HF checkpoint export.

Note: These tests use HuggingFace models created with create_tiny_llama_dir() rather than
the simple test models from sparse_attention_common.py because export_hf_checkpoint()
requires HF-specific features (model.save_pretrained(), model.config, etc.).
"""

import json

import pytest
import torch
from _test_utils.torch_model.transformers_models import create_tiny_llama_dir
from _test_utils.torch_sparsity.sparse_attention_common import FLASH_SOFTMAX_SKIP_CALIBRATION_CFG
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionConfig

# Skip all tests if GPU is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """Create tiny Llama model directory for testing.

    Uses create_tiny_llama_dir() to create a minimal local HF model
    without downloading, which is faster and doesn't require network access.
    """
    tmp_path = tmp_path_factory.mktemp("models")
    return create_tiny_llama_dir(tmp_path, with_tokenizer=True, num_hidden_layers=2)


@pytest.fixture(scope="module")
def tinyllama_model(tiny_llama_dir):
    """Load tiny Llama model for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    return model


@pytest.fixture(scope="module")
def tinyllama_tokenizer(tiny_llama_dir):
    """Load tiny Llama tokenizer for testing."""
    return AutoTokenizer.from_pretrained(tiny_llama_dir)


class TestSparseAttentionExport:
    """Test sparse attention model export to HF unified checkpoint."""

    def test_export_non_calibrated_model(self, tinyllama_model, tmp_path):
        """Test export of non-calibrated sparse attention model."""
        model = tinyllama_model

        # Apply sparse attention without calibration
        config = SparseAttentionConfig(
            method="flash_softmax_skip",
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-4,
                    "br": 128,
                    "bc": 128,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Export to temporary directory
        export_dir = tmp_path / "non_calibrated_export"
        export_hf_checkpoint(sparse_model, export_dir=export_dir)

        # Verify config.json was created
        config_path = export_dir / "config.json"
        assert config_path.exists(), "config.json not found"

        # Load and verify sparse_attention_config
        with open(config_path) as f:
            exported_config = json.load(f)

        assert "sparse_attention_config" in exported_config, "sparse_attention_config not found"
        sparse_config = exported_config["sparse_attention_config"]

        # Verify structure for non-calibrated model
        assert "config_groups" in sparse_config
        assert "producer" in sparse_config
        assert sparse_config["producer"]["name"] == "modelopt"

        # Should NOT have global calibration parameters
        assert "threshold_scale_factor" not in sparse_config
        assert "target_sparsity" not in sparse_config

        # Verify config_groups has threshold
        groups = sparse_config["config_groups"]
        assert len(groups) > 0, "No groups found in sparse_attention_config"

        # Check first group
        group_0 = groups["group_0"]

        assert "sparse_algo" in group_0
        assert group_0["sparse_algo"] == "softmax_skip"
        assert "threshold" in group_0
        assert isinstance(group_0["threshold"], float)
        assert group_0["threshold"] == 1e-4
        assert "targets" in group_0
        assert isinstance(group_0["targets"], list)
        assert len(group_0["targets"]) > 0

    def test_export_non_calibrated_phase_aware_model(self, tinyllama_model, tmp_path):
        """Test export of non-calibrated model with phase-aware thresholds."""
        model = tinyllama_model

        # Apply sparse attention with phase-aware thresholds
        config = SparseAttentionConfig(
            method="flash_softmax_skip",
            sparse_cfg={
                "*attn*": {
                    "threshold": {"prefill": 1e-3, "decode": 1e-5},
                    "br": 128,
                    "bc": 128,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Export to temporary directory
        export_dir = tmp_path / "phase_aware_export"
        export_hf_checkpoint(sparse_model, export_dir=export_dir)

        # Load and verify config
        config_path = export_dir / "config.json"
        with open(config_path) as f:
            exported_config = json.load(f)

        sparse_config = exported_config["sparse_attention_config"]
        groups = sparse_config["config_groups"]

        # Check first group has phase-aware threshold dict
        group_0 = groups["group_0"]

        assert "threshold" in group_0
        threshold = group_0["threshold"]
        assert isinstance(threshold, dict)
        assert "prefill" in threshold
        assert "decode" in threshold
        assert threshold["prefill"] == 1e-3
        assert threshold["decode"] == 1e-5
        assert "targets" in group_0

    def test_export_calibrated_model(self, tiny_llama_dir, tmp_path):
        """Test export of calibrated sparse attention model."""
        # Load a fresh model instance for calibration
        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Apply sparse attention with calibration (use common config)
        config = SparseAttentionConfig(**FLASH_SOFTMAX_SKIP_CALIBRATION_CFG)

        # Create a simple forward loop for calibration
        def forward_loop(model):
            """Simple forward loop for calibration."""
            device = next(model.parameters()).device
            # Create a few samples of different lengths
            for seq_len in [512, 768, 1024]:
                input_ids = torch.randint(0, 100, (1, seq_len), device=device)
                with torch.no_grad():
                    model(input_ids)

        # Sparsify with calibration
        sparse_model = sparse_attn.sparsify(model, config, forward_loop=forward_loop)

        # Export to temporary directory
        export_dir = tmp_path / "calibrated_export"
        export_hf_checkpoint(sparse_model, export_dir=export_dir)

        # Verify config.json
        config_path = export_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            exported_config = json.load(f)

        assert "sparse_attention_config" in exported_config
        sparse_config = exported_config["sparse_attention_config"]

        # Verify structure for calibrated model
        assert "config_groups" in sparse_config
        assert "producer" in sparse_config

        # SHOULD have global calibration parameters
        assert "threshold_scale_factor" in sparse_config
        assert "target_sparsity" in sparse_config

        # Verify calibration values
        assert isinstance(sparse_config["threshold_scale_factor"], float)
        assert sparse_config["threshold_scale_factor"] > 0
        assert sparse_config["target_sparsity"] == 0.5

        # Verify config_groups do NOT have threshold (calibrated)
        groups = sparse_config["config_groups"]
        assert len(groups) > 0

        group_0 = groups["group_0"]

        assert "sparse_algo" in group_0
        assert group_0["sparse_algo"] == "softmax_skip"
        # Should NOT have threshold field for calibrated model
        assert "threshold" not in group_0
        assert "targets" in group_0

    def test_export_model_without_sparse_attention(self, tiny_llama_dir, tmp_path):
        """Test export of model without sparse attention."""
        # Load a fresh model instance (not the shared fixture)
        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Export without applying sparse attention
        export_dir = tmp_path / "no_sparse_export"
        export_hf_checkpoint(model, export_dir=export_dir)

        # Verify config.json
        config_path = export_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            exported_config = json.load(f)

        # Should NOT have sparse_attention_config
        assert "sparse_attention_config" not in exported_config

    def test_export_disabled_sparse_attention(self, tinyllama_model, tmp_path):
        """Test export of model with disabled sparse attention modules."""
        model = tinyllama_model

        # Apply sparse attention
        config = SparseAttentionConfig(
            method="flash_softmax_skip",
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-4,
                    "br": 128,
                    "bc": 128,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Disable all sparse attention modules
        sparse_attn.disable_sparse_attention(sparse_model, "*")

        # Export to temporary directory
        export_dir = tmp_path / "disabled_export"
        export_hf_checkpoint(sparse_model, export_dir=export_dir)

        # Verify config.json
        config_path = export_dir / "config.json"
        with open(config_path) as f:
            exported_config = json.load(f)

        # Should NOT have sparse_attention_config (all modules disabled)
        assert "sparse_attention_config" not in exported_config

    def test_export_all_layers_have_same_config(self, tinyllama_model, tmp_path):
        """Test that all layers in exported config have consistent sparse_algo."""
        model = tinyllama_model

        config = SparseAttentionConfig(
            method="flash_softmax_skip",
            sparse_cfg={
                "*attn*": {
                    "threshold": 1e-4,
                    "br": 128,
                    "bc": 128,
                    "backend": "pytorch",
                    "enable": True,
                }
            },
        )

        sparse_model = sparse_attn.sparsify(model, config)

        # Export
        export_dir = tmp_path / "consistent_config_export"
        export_hf_checkpoint(sparse_model, export_dir=export_dir)

        # Load config
        config_path = export_dir / "config.json"
        with open(config_path) as f:
            exported_config = json.load(f)

        groups = exported_config["sparse_attention_config"]["config_groups"]

        # Verify there's only one config group (all layers have same config)
        assert len(groups) == 1, f"Expected 1 group (all layers same config), got: {len(groups)}"

        # Verify the single group has correct algo
        group_0 = groups["group_0"]
        assert group_0["sparse_algo"] == "softmax_skip"
        assert "targets" in group_0
