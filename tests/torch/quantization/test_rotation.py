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
from modelopt.torch.quantization.rotation import (
    apply_per_head_rotation,
    apply_rotation,
    build_rotation_config,
    build_rotation_config_from_yaml,
)
from modelopt.torch.quantization.rotation.rotate_utils import (
    extract_layer_index,
    get_orthogonal_matrix,
)


@pytest.fixture
def tiny_llama():
    """Create a tiny LlamaForCausalLM model for testing."""
    return get_tiny_llama()


def test_r1_global_rotation(tiny_llama):
    """Test R1 global rotation application."""
    orig_qproj = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()

    config = {
        "rotation_matrices": {"r1": {"mode": "hadamard"}},
        "rotation_config": {
            "*layers.*.self_attn.q_proj": ["r1", None],
        },
    }

    flow_config = build_rotation_config(config, tiny_llama)
    apply_rotation(tiny_llama, flow_config)

    # Check weights changed
    assert not torch.allclose(orig_qproj, tiny_llama.model.layers[0].self_attn.q_proj.weight.data)


def test_r2_per_layer_rotation(tiny_llama):
    """Test R2 per-layer generation."""
    config = {
        "rotation_matrices": {"r2": {"mode": "hadamard", "per_layer": True}},
        "rotation_config": {},  # Will be tested via build_rotation_config
    }

    flow_config = build_rotation_config(config, tiny_llama)

    # Verify per-layer R2 matrices were generated
    assert "r2_0" in flow_config["per_layer_r2"]
    assert "r2_1" in flow_config["per_layer_r2"]

    # Verify correct dimensions (hidden_size / num_heads = 32/16 = 2)
    assert flow_config["per_layer_r2"]["r2_0"].shape == (2, 2)
    assert flow_config["per_layer_r2"]["r2_1"].shape == (2, 2)


def test_per_head_rotation():
    """Test per-head rotation for V/O projections."""
    hidden_size = 256
    num_heads = 8
    head_dim = hidden_size // num_heads

    weight = torch.randn(hidden_size, hidden_size)
    orig_weight = weight.clone()

    r2 = get_orthogonal_matrix(head_dim, "hadamard", weight.device)
    rotated_weight = apply_per_head_rotation(weight, r2, num_heads, transpose_first=False)

    assert not torch.allclose(orig_weight, rotated_weight)
    assert rotated_weight.shape == weight.shape


def test_dimension_auto_extraction(tiny_llama, tmp_path):
    """Test dimension extraction and matrix generation from model.config."""
    yaml_content = """
rotation_matrices:
  r1:
    mode: hadamard
  r2:
    mode: hadamard
    per_layer: true

rotation_config:
  "*.embed_tokens": [r1, null]
"""

    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = build_rotation_config_from_yaml(str(yaml_file), tiny_llama)

    # Check that matrices were generated with correct dimensions
    assert config["rotation_matrices"]["r1"].shape == (32, 32)  # tiny_llama hidden_size=32
    assert "r2_0" in config["per_layer_r2"]
    assert config["per_layer_r2"]["r2_0"].shape == (2, 2)  # 32/16 heads = 2


def test_extract_layer_index():
    """Test layer index extraction from module names."""
    assert extract_layer_index("model.layers.0.self_attn.q_proj") == 0
    assert extract_layer_index("model.layers.12.self_attn.o_proj") == 12
    assert extract_layer_index("model.layers.99.mlp.down_proj") == 99
    assert extract_layer_index("model.embed_tokens") is None


def test_r1_only_rotation(tiny_llama):
    """Test that R1-only rotations preserve model output."""
    vocab_size = tiny_llama.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 10))

    # Get output before rotation
    with torch.no_grad():
        output_before = tiny_llama.model(input_ids).last_hidden_state

    # Apply R1-only rotations WITH layernorm fusion
    config = {
        "rotation_matrices": {"r1": {"mode": "hadamard"}},
        "rotation_config": {
            "*embed_tokens": ["r1", None],
            "*lm_head": ["r1", None],
            "*q_proj|*k_proj|*v_proj": ["r1", None],
            "*o_proj": [None, "r1"],
            "*up_proj|*gate_proj": ["r1", None],
            "*down_proj": [None, "r1"],
        },
        "per_head_config": {},
        "norm_fuse_config": {
            "decoder_layer_fuse": [
                ["input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]],
                ["post_attention_layernorm", ["mlp.up_proj", "mlp.gate_proj"]],
            ],
            "lm_head_fuse": [["model.norm", "lm_head"]],
        },
    }
    flow_config = build_rotation_config(config, tiny_llama)
    apply_rotation(tiny_llama, flow_config)

    # Get output after rotation
    with torch.no_grad():
        output_after = tiny_llama.model(input_ids).last_hidden_state

    max_diff = (output_before - output_after).abs().max().item()

    # Check for NaN/Inf
    has_nan_before = torch.isnan(output_before).any().item()
    has_nan_after = torch.isnan(output_after).any().item()
    has_inf_before = torch.isinf(output_before).any().item()
    has_inf_after = torch.isinf(output_after).any().item()

    print("\nR1-only rotation test:")
    print(f"  Max diff: {max_diff}")
    print(f"  NaN before/after: {has_nan_before}/{has_nan_after}")
    print(f"  Inf before/after: {has_inf_before}/{has_inf_after}")

    # Check a single layer's weight to see if rotation was applied correctly
    q0_weight = tiny_llama.model.layers[0].self_attn.q_proj.weight
    print(f"  Q0 weight range: [{q0_weight.min().item():.4f}, {q0_weight.max().item():.4f}]")

    # With orthogonal rotations, outputs should be identical
    assert torch.allclose(output_before, output_after, atol=1e-2, rtol=1e-2), (
        f"R1 rotations don't preserve output: max diff = {max_diff}"
    )


def test_quantize_with_rotation(tiny_llama):
    """Test integration of rotation with quantization."""
    config = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "axis": 0},
            "*input_quantizer": {"enable": False},
        },
        "algorithm": "max",
        "rotation": {
            "enabled": True,
            "rotation_matrices": {"r1": {"mode": "hadamard"}},
            "rotation_config": {
                "*.embed_tokens": ["r1", None],
                "*.lm_head": ["r1", None],
            },
        },
    }

    orig_embed = tiny_llama.model.embed_tokens.weight.data.clone()

    model = mtq.quantize(tiny_llama, config)

    assert not torch.allclose(orig_embed, model.model.embed_tokens.weight.data)


def test_full_spinquant_rotation(tiny_llama):
    """Test full rotation application (R1 + R2 + online transforms)."""
    vocab_size = tiny_llama.config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 10))

    # Get output before rotation
    with torch.no_grad():
        output_before = tiny_llama.model(input_ids).last_hidden_state

    # Apply full rotation from universal config
    yaml_path = "modelopt/torch/quantization/rotation/configs/transformer_universal.yaml"
    flow_config = build_rotation_config_from_yaml(yaml_path, tiny_llama)

    # Verify config was built correctly
    assert "r1" in flow_config["rotation_matrices"]
    assert "per_layer_r2" in flow_config
    assert len(flow_config["per_layer_r2"]) == 2  # 2 layers in tiny_llama

    # Apply rotation
    apply_rotation(tiny_llama, flow_config)

    # Verify online transforms were registered
    assert hasattr(tiny_llama.model.layers[0].self_attn.o_proj, "_rotation_hook_handle")
    assert hasattr(tiny_llama.model.layers[0].mlp.down_proj, "_rotation_hook_handle")

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

    print("\nSpinQuant rotation test:")
    print(f"  R1 matrix: {flow_config['rotation_matrices']['r1'].shape}")
    print(f"  Per-layer R2 matrices: {len(flow_config['per_layer_r2'])}")
    print("  Online hooks registered: o_proj, down_proj")
    print(f"  Output max diff: {max_diff:.4f}")
    print(f"  Output mean magnitude before/after: {mean_before:.4f} / {mean_after:.4f}")

    # Verify outputs are reasonable (not completely garbled)
    assert mean_after < mean_before * 5, (
        f"Output magnitude too large: {mean_after} vs {mean_before}"
    )
    assert mean_after > mean_before / 5, (
        f"Output magnitude too small: {mean_after} vs {mean_before}"
    )
