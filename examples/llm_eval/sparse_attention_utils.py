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

"""Utilities for sparse attention integration with llm_eval."""

import torch

import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.calibration import RulerDatasetBuilder
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

# Custom sparse attention configurations
CUSTOM_SPARSE_CONFIG = {
    "SPARSE_CONSERVATIVE": {
        "sparse_cfg": {
            "*attn*": {
                "method": "flash_skip_softmax",
                "threshold": {"prefill": 5e-4, "decode": 1e-5},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "enable": True,
            },
            "default": {"enable": False},
        },
    },
    "SPARSE_AGGRESSIVE": {
        "sparse_cfg": {
            "*attn*": {
                "method": "flash_skip_softmax",
                "threshold": {"prefill": 5e-3, "decode": 5e-4},
                "br": 128,
                "bc": 128,
                "backend": "pytorch",
                "enable": True,
            },
            "default": {"enable": False},
        },
    },
}


def _extract_model(model_obj):
    """Extract actual model from wrapper (HFLM or EvalModel)."""
    if hasattr(model_obj, "gpt2"):
        return model_obj.gpt2
    elif hasattr(model_obj, "model"):
        return model_obj.model
    else:
        return model_obj


def print_sparse_attention_stats(model):
    """Print sparse attention statistics.

    Args:
        model: Model with sparse attention applied
    """
    sparse_modules = []
    total_sparsity = 0
    enabled_count = 0

    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule):
            sparse_modules.append((name, module))
            if module.is_enabled:
                enabled_count += 1
                # Get stats if available
                stats = module.get_stats()
                if stats and "sparsity" in stats:
                    total_sparsity += stats["sparsity"]

    print("\n" + "=" * 60)
    print("SPARSE ATTENTION SUMMARY")
    print("=" * 60)
    print(f"Total sparse attention modules: {len(sparse_modules)}")
    print(f"Enabled modules: {enabled_count}")

    if enabled_count > 0:
        avg_sparsity = total_sparsity / enabled_count if total_sparsity > 0 else 0
        print(f"Average sparsity ratio: {avg_sparsity:.2%}")

        # Print per-layer info
        print("\nPer-layer configuration:")
        for name, module in sparse_modules[:5]:  # Show first 5
            method = getattr(module, "_method", "unknown")
            threshold = getattr(module, "_threshold", "N/A")
            status = "✓" if module.is_enabled else "✗"
            print(f"  {status} {name}: method={method}, threshold={threshold}")

        if len(sparse_modules) > 5:
            print(f"  ... and {len(sparse_modules) - 5} more modules")

    print("=" * 60 + "\n")


def _sparsify_model_with_ruler(
    model,
    sparse_cfg,
    tokenizer,
    ruler_samples=12,
    ruler_max_seqlen=8192,
):
    """Apply sparse attention with RULER calibration.

    Args:
        model: Model to sparsify
        sparse_cfg: Sparse attention configuration
        tokenizer: Tokenizer for RULER dataset
        ruler_samples: Number of RULER samples
        ruler_max_seqlen: Maximum sequence length
    """
    # Check if config requires calibration
    requires_calibration = False
    if isinstance(sparse_cfg, dict):
        sparse_cfg_dict = sparse_cfg.get("sparse_cfg", {})
        for pattern_cfg in sparse_cfg_dict.values():
            if isinstance(pattern_cfg, dict) and "calibration" in pattern_cfg:
                requires_calibration = True
                break

    if requires_calibration:
        print("\nBuilding RULER calibration dataset...")
        print(f"  Samples: {ruler_samples}")
        print(f"  Max sequence length: {ruler_max_seqlen}")

        # Build RULER dataset
        ruler_builder = RulerDatasetBuilder(
            samples=ruler_samples,
            max_seqlen=ruler_max_seqlen,
            tokenizer_name_or_path=tokenizer,
        )

        # Create forward loop
        def forward_loop(model):
            """Forward loop for RULER calibration."""
            # Generate RULER samples
            for sample_dict in ruler_builder.generate_samples(num_samples=ruler_samples):
                input_ids = sample_dict["input_ids"].to(model.device)
                with torch.no_grad():
                    model(input_ids)

        # Apply sparsification with calibration
        mtsa.sparsify(model, sparse_cfg, forward_loop=forward_loop)
    else:
        # No calibration, use static thresholds
        mtsa.sparsify(model, sparse_cfg)


def sparsify_model(
    model,
    sparse_cfg: str,
    tokenizer,
    ruler_samples=12,
    ruler_max_seqlen=8192,
    test_generated=True,
):
    """Apply sparse attention to model with optional RULER calibration.

    Args:
        model: Model wrapper (HFLM or EvalModel) or raw model
        sparse_cfg: Sparse attention config name or dict
        tokenizer: Tokenizer for RULER dataset
        ruler_samples: Number of RULER calibration samples
        ruler_max_seqlen: Maximum sequence length for calibration
        test_generated: If True, test generation before/after sparsification
    """
    # Extract actual model
    net = _extract_model(model)

    # Resolve config
    if isinstance(sparse_cfg, str):
        # Try custom configs first
        mtsa_cfg = CUSTOM_SPARSE_CONFIG.get(sparse_cfg)
        if mtsa_cfg is None:
            # Try predefined configs
            mtsa_cfg = getattr(mtsa, sparse_cfg, None)
        if mtsa_cfg is None:
            raise ValueError(f"Unknown sparse_cfg: {sparse_cfg}")
    else:
        mtsa_cfg = sparse_cfg

    # Test generation before sparsification
    if test_generated and hasattr(model, "run"):
        test_input = tokenizer.decode(torch.randint(0, 100, (10,)))
        generated_before = model.run(test_input)

    # Apply sparsification
    print(f"\nApplying sparse attention with config: {sparse_cfg}")
    _sparsify_model_with_ruler(
        net,
        mtsa_cfg,
        tokenizer,
        ruler_samples,
        ruler_max_seqlen,
    )

    # Print statistics
    print_sparse_attention_stats(net)

    # Test generation after sparsification
    if test_generated and hasattr(model, "run"):
        generated_after = model.run(test_input)
        print("--------")
        print(f"Test input: {test_input}")
        print("--------")
        print(f"Output before sparse attention: {generated_before}")
        print("--------")
        print(f"Output after sparse attention: {generated_after}")
        print("--------")
