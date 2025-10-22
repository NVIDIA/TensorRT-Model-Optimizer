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

"""Tests for rotation preprocessing."""

import pytest
import torch
from _test_utils.torch_model.transformers_models import get_tiny_llama

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.rotation import apply_rotation, build_rotation_config_from_yaml
from modelopt.torch.quantization.rotation.rotate_utils import extract_layer_index


@pytest.fixture
def tiny_llama():
    """Create a tiny LlamaForCausalLM model for testing."""
    return get_tiny_llama()


def test_r1_global_rotation(tiny_llama):
    """Test R1 global rotation application."""
    orig_qproj = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()

    config = {
        "rotation_matrices": {"r1": {"mode": "hadamard", "dim": 32}},
        "rotation_config": {
            "*layers.*.self_attn.q_proj": ["r1", None],
        },
    }

    apply_rotation(tiny_llama, config)

    # Check weights changed
    assert not torch.allclose(orig_qproj, tiny_llama.model.layers[0].self_attn.q_proj.weight.data)


def test_yaml_config_loading(tmp_path):
    """Test YAML config loading."""
    yaml_content = """
rotation_matrices:
  r1:
    mode: hadamard

rotation_config:
  "*.embed_tokens": [r1, null]
"""

    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = build_rotation_config_from_yaml(str(yaml_file))

    # Check that config was loaded correctly
    assert "rotation_matrices" in config
    assert "r1" in config["rotation_matrices"]
    assert config["rotation_matrices"]["r1"]["mode"] == "hadamard"
    assert "rotation_config" in config


def test_extract_layer_index():
    """Test layer index extraction from module names."""
    assert extract_layer_index("model.layers.0.self_attn.q_proj") == 0
    assert extract_layer_index("model.layers.12.self_attn.o_proj") == 12
    assert extract_layer_index("model.layers.99.mlp.down_proj") == 99
    assert extract_layer_index("model.embed_tokens") is None


def test_r2_rotation(tiny_llama):
    """Test R2 per-layer rotation application."""
    orig_vproj = tiny_llama.model.layers[0].self_attn.v_proj.weight.data.clone()

    # Head dimension is hidden_size / num_attention_heads = 32 / 16 = 2
    config = {
        "rotation_matrices": {"r2": {"mode": "hadamard", "dim": 2, "per_layer": True}},
        "rotation_config": {
            "*v_proj": [None, "r2"],  # R2 on output
        },
    }

    apply_rotation(tiny_llama, config)

    # Check weights changed
    assert not torch.allclose(orig_vproj, tiny_llama.model.layers[0].self_attn.v_proj.weight.data)


def test_r1_and_r2_combined(tiny_llama):
    """Test combined R1 and R2 rotation application."""
    orig_vproj = tiny_llama.model.layers[0].self_attn.v_proj.weight.data.clone()
    orig_oproj = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()

    config = {
        "rotation_matrices": {
            "r1": {"mode": "hadamard", "dim": 32},
            "r2": {"mode": "hadamard", "dim": 2, "per_layer": True},
        },
        "rotation_config": {
            "*v_proj": ["r1", "r2"],  # R1 on input, R2 on output
            "*o_proj": ["r2", "r1"],  # R2 on input, R1 on output
        },
    }

    apply_rotation(tiny_llama, config)

    # Check both weights changed
    assert not torch.allclose(orig_vproj, tiny_llama.model.layers[0].self_attn.v_proj.weight.data)
    assert not torch.allclose(orig_oproj, tiny_llama.model.layers[0].self_attn.o_proj.weight.data)


def test_r1_with_layernorm_fusion(tiny_llama):
    """Test R1 rotation with layernorm fusion."""
    orig_qproj = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()

    vocab_size = tiny_llama.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 10))

    # Apply R1-only rotations WITH layernorm fusion
    config = {
        "rotation_matrices": {"r1": {"mode": "hadamard", "dim": 32}},
        "rotation_config": {
            "*embed_tokens": ["r1", None],
            "*lm_head": ["r1", None],
            "*q_proj|*k_proj|*v_proj": ["r1", None],
            "*o_proj": [None, "r1"],
            "*up_proj|*gate_proj": ["r1", None],
            "*down_proj": [None, "r1"],
        },
        "norm_fuse_config": {
            "decoder_layer_fuse": [
                ["input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]],
                ["post_attention_layernorm", ["mlp.up_proj", "mlp.gate_proj"]],
            ],
            "lm_head_fuse": [["model.norm", "lm_head"]],
        },
    }
    apply_rotation(tiny_llama, config)

    # Verify weights were modified
    assert not torch.allclose(orig_qproj, tiny_llama.model.layers[0].self_attn.q_proj.weight.data)

    # Get output after rotation and verify it's valid (no NaN/Inf)
    with torch.no_grad():
        output_after = tiny_llama.model(input_ids).last_hidden_state

    # Verify outputs are valid (no NaN/Inf)
    assert not torch.isnan(output_after).any(), "Output contains NaN after rotation"
    assert not torch.isinf(output_after).any(), "Output contains Inf after rotation"


def test_mtq_rotate_api(tiny_llama):
    """Test mtq.rotate() API (as used in main.py)."""
    orig_embed = tiny_llama.model.embed_tokens.weight.data.clone()

    config = {
        "rotation_matrices": {"r1": {"mode": "hadamard", "dim": 32}},
        "rotation_config": {
            "*embed_tokens": ["r1", None],
        },
    }

    # Use mtq.rotate() API which returns the model
    rotated_model = mtq.rotate(tiny_llama, config)

    # Verify it's the same model object (in-place modification)
    assert rotated_model is tiny_llama

    # Verify weights changed
    assert not torch.allclose(orig_embed, rotated_model.model.embed_tokens.weight.data)


def test_yaml_rotation_application(tiny_llama):
    """Test rotation application from YAML config."""
    vocab_size = tiny_llama.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 10))

    # Get output before rotation
    with torch.no_grad():
        output_before = tiny_llama.model(input_ids).last_hidden_state

    # Apply rotation from universal config using path string (can use either API)
    yaml_path = "modelopt/torch/quantization/rotation/configs/transformer_universal.yaml"
    mtq.rotate(tiny_llama, yaml_path)

    # Get output after rotation
    with torch.no_grad():
        output_after = tiny_llama.model(input_ids).last_hidden_state

    # Verify outputs are valid (no NaN/Inf)
    assert not torch.isnan(output_after).any(), "Output contains NaN after rotation"
    assert not torch.isinf(output_after).any(), "Output contains Inf after rotation"

    # Compare outputs - they will differ but should be in similar range
    max_diff = (output_before - output_after).abs().max().item()
    mean_before = output_before.abs().mean().item()
    mean_after = output_after.abs().mean().item()

    print("\nYAML rotation test:")
    print(f"  Output max diff: {max_diff:.4f}")
    print(f"  Output mean magnitude before/after: {mean_before:.4f} / {mean_after:.4f}")

    # Verify outputs are reasonable (not completely garbled)
    assert mean_after < mean_before * 5, (
        f"Output magnitude too large: {mean_after} vs {mean_before}"
    )
    assert mean_after > mean_before / 5, (
        f"Output magnitude too small: {mean_after} vs {mean_before}"
    )
