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

r"""Compare module output tensors from different model variants.

This module provides:
1. OutputSaveHook - A PyTorch hook to capture module outputs during forward pass
2. Comparison utilities - Compute RMSE and cosine similarity between saved outputs

Usage Example:
--------------

Step 1: Capture outputs using OutputSaveHook in your training/inference script:

    from compare_module_outputs import OutputSaveHook

    # Create hook instance
    hook = OutputSaveHook()

    # Register on target module
    model.decoder.layers[0].mlp.linear_fc2.register_forward_hook(hook)

    # Run inference/training
    model(input_data)

    # Save captured outputs
    hook.save("output_unpruned.pt")

Step 2: Compare outputs from different model variants:

    python compare_module_outputs.py \
        --reference output_unpruned.pt \
        --compare output_l2norm.pt

"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputSaveHook:
    """Hook to capture and save module outputs during forward pass."""

    def __init__(self) -> None:
        """Initialize the output save hook."""
        self.saved_outputs: list[torch.Tensor] = []

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Capture and save module output during forward pass.

        Args:
            module: The PyTorch module being hooked.
            args: Input arguments to the module's forward pass.
            output: Output tensor(s) from the module's forward pass.
        """
        self.saved_outputs.append(output.detach().cpu())

    def save(self, path: str) -> None:
        """Save collected outputs to disk."""
        output_tensor = torch.stack(self.saved_outputs)
        torch.save(output_tensor, path)
        print(f"Saved {len(self.saved_outputs)} outputs with shape {output_tensor.shape} to {path}")


def compute_rmse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Root Mean Square Error between two tensors."""
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def compute_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> dict:
    """Compute average cosine similarity between two tensors."""
    # Flatten to 2D for cosine similarity computation
    t1_flat = tensor1.reshape(-1, tensor1.shape[-1])
    t2_flat = tensor2.reshape(-1, tensor2.shape[-1])

    # Compute cosine similarity per position
    cos_sim = F.cosine_similarity(t1_flat, t2_flat, dim=-1)

    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "std": cos_sim.std().item(),
    }


def main():
    """Compare module output tensors from different model variants."""
    parser = argparse.ArgumentParser(
        description="Compare module output tensors from different model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference output tensor (e.g., unpruned model)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        required=True,
        help="Path to output tensor to compare against reference",
    )
    args = parser.parse_args()

    # Load reference tensor
    print(f"\nLoading reference: {args.reference}")
    ref_tensor = torch.load(args.reference, map_location="cpu")
    print(f"Reference shape: {ref_tensor.shape}")

    # Load comparison tensor
    print(f"\nLoading compare: {args.compare}")
    comp_tensor = torch.load(args.compare, map_location="cpu")
    print(f"Compare shape: {comp_tensor.shape}")

    if ref_tensor.shape != comp_tensor.shape:
        print("\nERROR: Shape mismatch! Cannot compare.")
        return

    # Compute metrics
    rmse = compute_rmse(ref_tensor, comp_tensor)
    cos_sim = compute_cosine_similarity(ref_tensor, comp_tensor)

    print(f"\n{'=' * 70}")
    print("Comparison Results")
    print(f"{'=' * 70}")
    print(f"\nRMSE: {rmse:.6f}")
    print("\nCosine Similarity:")
    print(f"  Mean: {cos_sim['mean']:.6f}")
    print(f"  Min:  {cos_sim['min']:.6f}")
    print(f"  Max:  {cos_sim['max']:.6f}")
    print(f"  Std:  {cos_sim['std']:.6f}")

    # Compute per-step metrics
    print("\nPer-step RMSE:")
    for i in range(ref_tensor.shape[0]):
        step_rmse = compute_rmse(ref_tensor[i], comp_tensor[i])
        print(f"  Step {i}: {step_rmse:.6f}")


if __name__ == "__main__":
    main()
