#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import re
from collections import defaultdict

import torch


def convert_amax_hf2vllm(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert amax values from HuggingFace format to vLLM format.

    This function merges:
    - q_proj, k_proj, v_proj amax values into qkv_proj (taking max)
    - gate_proj, up_proj amax values into gate_up_proj (taking max)

    Args:
        hf_state_dict: HuggingFace state dict containing amax values

    Returns:
        vLLM format state dict with merged amax values
    """
    vllm_state_dict = {}

    # Group keys by their base pattern (without the specific projection name)
    merge_groups = defaultdict(list)

    for key, value in hf_state_dict.items():
        if "_amax" not in key:
            # Copy non-amax keys as-is
            vllm_state_dict[key] = value
            continue

        # Check if this is a q/k/v projection that needs merging
        qkv_match = re.search(r"(.*\.)([qkv])_proj(\..+_amax)$", key)
        if qkv_match:
            base_pattern = qkv_match.group(1) + "qkv_proj" + qkv_match.group(3)
            merge_groups[base_pattern].append((key, value))
            continue

        # Check if this is a gate/up projection that needs merging
        gate_up_match = re.search(r"(.*\.)(gate|up)_proj(\..+_amax)$", key)
        if gate_up_match:
            base_pattern = gate_up_match.group(1) + "gate_up_proj" + gate_up_match.group(3)
            merge_groups[base_pattern].append((key, value))
            continue

        # Copy other amax keys as-is (like o_proj, down_proj)
        vllm_state_dict[key] = value

    # Merge grouped amax values by taking the maximum
    for merged_key, key_value_pairs in merge_groups.items():
        if len(key_value_pairs) > 1:
            # Take the maximum across all values for this merged key
            values = [value for _, value in key_value_pairs]
            merged_value = torch.stack(values).max(dim=0)[0]
            vllm_state_dict[merged_key] = merged_value
            print(f"Merged {len(key_value_pairs)} keys into {merged_key}")
            for orig_key, _ in key_value_pairs:
                print(f"  - {orig_key}")
        else:
            # Single key, just rename it
            _, value = key_value_pairs[0]
            vllm_state_dict[merged_key] = value

    return vllm_state_dict


def test_conversion():
    """Test the conversion logic with sample keys"""
    import torch

    # Create sample HF state dict
    sample_hf_keys = [
        "model.layers.0.self_attn.q_proj.input_quantizer._amax",
        "model.layers.0.self_attn.k_proj.input_quantizer._amax",
        "model.layers.0.self_attn.v_proj.input_quantizer._amax",
        "model.layers.0.self_attn.q_proj.weight_quantizer._amax",
        "model.layers.0.self_attn.k_proj.weight_quantizer._amax",
        "model.layers.0.self_attn.v_proj.weight_quantizer._amax",
        "model.layers.0.self_attn.o_proj.input_quantizer._amax",
        "model.layers.0.self_attn.o_proj.weight_quantizer._amax",
        "model.layers.0.mlp.gate_proj.input_quantizer._amax",
        "model.layers.0.mlp.up_proj.input_quantizer._amax",
        "model.layers.0.mlp.gate_proj.weight_quantizer._amax",
        "model.layers.0.mlp.up_proj.weight_quantizer._amax",
        "model.layers.0.mlp.down_proj.input_quantizer._amax",
        "model.layers.0.mlp.down_proj.weight_quantizer._amax",
    ]

    hf_state_dict = {}
    for key in sample_hf_keys:
        hf_state_dict[key] = torch.tensor([1.0, 2.0, 3.0])  # Sample values

    print("Testing conversion with sample keys...")
    print(f"Input keys: {len(sample_hf_keys)}")

    vllm_state_dict = convert_amax_hf2vllm(hf_state_dict)
    vllm_amax_keys = [k for k in vllm_state_dict if "_amax" in k]

    print(f"Output keys: {len(vllm_amax_keys)}")
    print("\nExpected vLLM keys:")
    expected_keys = [
        "model.layers.0.self_attn.qkv_proj.input_quantizer._amax",
        "model.layers.0.self_attn.qkv_proj.weight_quantizer._amax",
        "model.layers.0.self_attn.o_proj.input_quantizer._amax",
        "model.layers.0.self_attn.o_proj.weight_quantizer._amax",
        "model.layers.0.mlp.gate_up_proj.input_quantizer._amax",
        "model.layers.0.mlp.gate_up_proj.weight_quantizer._amax",
        "model.layers.0.mlp.down_proj.input_quantizer._amax",
        "model.layers.0.mlp.down_proj.weight_quantizer._amax",
    ]

    for key in expected_keys:
        print(f"  {key}")

    print("\nActual vLLM keys:")
    for key in sorted(vllm_amax_keys):
        print(f"  {key}")

    # Check if all expected keys are present
    missing_keys = set(expected_keys) - set(vllm_amax_keys)
    extra_keys = set(vllm_amax_keys) - set(expected_keys)

    if missing_keys:
        print(f"\nMissing keys: {missing_keys}")
    if extra_keys:
        print(f"\nExtra keys: {extra_keys}")

    if not missing_keys and not extra_keys:
        print("\n✓ Test passed! All keys converted correctly.")
    else:
        print("\n✗ Test failed! Key mismatch detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert amax values from HuggingFace to vLLM format"
    )
    parser.add_argument("--input", "-i", help="Input HuggingFace checkpoint path")
    parser.add_argument("--output", "-o", help="Output vLLM checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Show conversion without saving")
    parser.add_argument("--test", action="store_true", help="Run test with sample data")

    args = parser.parse_args()

    if args.test:
        test_conversion()
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless using --test")

    # Load HuggingFace checkpoint
    print(f"Loading HuggingFace checkpoint from: {args.input}")
    if os.path.isfile(args.input):
        hf_state_dict = torch.load(args.input, map_location="cpu")
    else:
        raise Exception(f"File not found: {args.input}")

    print(f"Loaded {len(hf_state_dict)} keys from HuggingFace checkpoint")

    # Filter to only amax keys for analysis
    amax_keys = [k for k in hf_state_dict if "_amax" in k]
    print(f"Found {len(amax_keys)} amax keys")

    if args.dry_run:
        print("\nAmax keys in HuggingFace format:")
        for key in sorted(amax_keys):
            print(f"  {key}")

    # Convert to vLLM format
    print("\nConverting to vLLM format...")
    vllm_state_dict = convert_amax_hf2vllm(hf_state_dict)

    vllm_amax_keys = [k for k in vllm_state_dict if "_amax" in k]
    print(f"Result: {len(vllm_amax_keys)} amax keys in vLLM format")

    if args.dry_run:
        print("\nAmax keys in vLLM format:")
        for key in sorted(vllm_amax_keys):
            print(f"  {key}")
        print("\nDry run complete. No files saved.")
        return

    # Save vLLM checkpoint
    print(f"Saving vLLM checkpoint to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(vllm_state_dict, args.output)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
