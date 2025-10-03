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

"""Calibration framework for sparse attention methods."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from ..nn.sparse_attention import SparseAttentionModule


class DynamicThresholdCalibrator:
    """Dynamic threshold calibrator using length-based linear relationship.

    Implements calibration algorithm:
    1. Find hyperparameter 'a' where threshold λ = a / context_length
    2. Use dataset with different lengths and test multiple thresholds
    3. For each sample, find optimal threshold closest to target sparsity
    4. Use linear regression to fit: threshold = a * (1/length)
    """

    def __init__(
        self,
        target_sparse_ratio: float = 0.5,
        threshold_trials: list[float] | None = None,
        phase: str | None = None,
        max_new_tokens: int = 50,
        tolerance: float = 0.05,
        max_iterations: int = 50,
    ):
        """Initialize dynamic threshold calibrator.

        Args:
            target_sparse_ratio: Target sparsity ratio (0.0 to 1.0)
            threshold_trials: List of thresholds to try
            phase: Calibration phase ("prefill" or "decode")
            max_new_tokens: Max tokens for generation during calibration
            tolerance: Acceptable tolerance from target ratio
            max_iterations: Maximum calibration iterations
        """
        self.target_sparse_ratio = target_sparse_ratio
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.phase = phase
        self.max_new_tokens = max_new_tokens

        # Default threshold trials if not provided
        self.threshold_trials = threshold_trials or [
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
        ]

        # Statistics tracking
        self.sparsity_results = []

    def calibrate(self, model: nn.Module, forward_loop: Callable) -> dict[str, Any]:
        """Find optimal 'a' parameter for length-based threshold.

        Algorithm:
            1. Test all threshold trials by running forward_loop multiple times
            2. For each sample, find optimal threshold closest to target sparsity
            3. Use regression to find 'a' in: threshold = a / length

        Args:
            model: The model with sparse attention modules
            forward_loop: Callable that takes model and forwards calibration data
        """
        # Extract attention modules
        attention_modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]

        if not attention_modules:
            return {"error": "No sparse attention modules found"}

        print("Starting dynamic threshold calibration")
        print(f"Target sparsity: {self.target_sparse_ratio}")
        print(f"Threshold trials: {len(self.threshold_trials)}")

        # Stage 1: Collect sparsity for all sample-threshold pairs
        print("\nStage 1: Collecting sparsity data...")
        self.sparsity_results = []

        # For each threshold, run forward_loop and collect per-sample statistics
        for threshold_idx, threshold in enumerate(
            tqdm(self.threshold_trials, desc="Testing thresholds")
        ):
            # Set threshold and enable calibration mode
            self._set_threshold(attention_modules, threshold)
            self._enable_calibration_mode(attention_modules, reset_history=True)

            # Run forward loop - modules will collect per-sample stats via hooks
            import torch

            with torch.no_grad():
                forward_loop(model)

            # Extract collected statistics from modules
            per_sample_stats = self._extract_calibration_stats(attention_modules)

            # Store results indexed by threshold
            for sample_idx, sample_stat in enumerate(per_sample_stats):
                # Find or create entry for this sample
                if threshold_idx == 0:
                    # First threshold - create new sample entry
                    sample_length = sample_stat.get("sample_length", 0)
                    if sample_length <= 0:
                        continue

                    self.sparsity_results.append(
                        {
                            "sample_index": sample_idx,
                            "length": sample_length,
                            "threshold_sparsities": {},
                        }
                    )

                # Add sparsity for this threshold
                if sample_idx < len(self.sparsity_results):
                    sparsity = sample_stat.get("sparsity", 0.0)
                    self.sparsity_results[sample_idx]["threshold_sparsities"][threshold] = sparsity

            # Disable calibration mode
            self._disable_calibration_mode(attention_modules)

        if not self.sparsity_results:
            return {"error": "No valid sparsity measurements collected"}

        print(f"Collected statistics for {len(self.sparsity_results)} samples")

        # Stage 2: Find optimal threshold for each sample and compute 'a'
        print(
            f"\nStage 2: Finding 'a' parameter for target sparsity {self.target_sparse_ratio:.2f}"
        )

        optimal_pairs = []

        for sample_result in self.sparsity_results:
            length = sample_result["length"]
            best_threshold = None
            min_diff = float("inf")
            achieved_sparsity = 0.0

            # Find threshold closest to target sparsity
            for threshold, sparsity in sample_result["threshold_sparsities"].items():
                diff = abs(sparsity - self.target_sparse_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_threshold = threshold
                    achieved_sparsity = sparsity

            if best_threshold is not None:
                optimal_pairs.append(
                    {
                        "length": length,
                        "optimal_threshold": best_threshold,
                        "achieved_sparsity": achieved_sparsity,
                        "target_sparsity": self.target_sparse_ratio,
                    }
                )

        if not optimal_pairs:
            print(f"Warning: No optimal pairs found for target sparsity {self.target_sparse_ratio}")
            print(f"  Collected {len(self.sparsity_results)} samples but none had valid thresholds")
            return {
                "error": f"No optimal pairs found for target sparsity {self.target_sparse_ratio}"
            }

        # Linear regression: threshold = a * (1/length)
        lengths = np.array([p["length"] for p in optimal_pairs])
        thresholds = np.array([p["optimal_threshold"] for p in optimal_pairs])

        # X = 1/length, Y = threshold
        x = 1.0 / lengths
        y = thresholds

        # Least squares: a = sum(x*y) / sum(x^2)
        a_parameter = np.sum(x * y) / np.sum(x**2)

        # Calculate statistics
        a_per_sample = y * lengths  # Individual 'a' values
        a_std_dev = np.std(a_per_sample)

        # Calculate R-squared for quality metric
        y_pred = a_parameter * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate average achieved sparsity during calibration
        avg_achieved_sparsity = np.mean([p["achieved_sparsity"] for p in optimal_pairs])

        print("\nCalibration Results:")
        print(f"  a parameter: {a_parameter:.6f} (std: {a_std_dev:.6f})")
        print(f"  R-squared: {r_squared:.4f}")
        print(
            f"  Average achieved sparsity: {avg_achieved_sparsity:.2%} (target: {self.target_sparse_ratio:.2%})"
        )
        print(f"\nExample thresholds with λ = {a_parameter:.6f} / length:")
        for length in [1024, 2048, 4096, 8192, 16384]:
            print(f"  Length {length:5d}: threshold = {a_parameter / length:.2e}")

        # Apply the calibrated parameter to modules
        self._apply_length_based_calibration(attention_modules, a_parameter)

        return {
            "a_parameter": a_parameter,
            "a_std_dev": a_std_dev,
            "r_squared": r_squared,
            "num_samples": len(optimal_pairs),
            "target_sparsity": self.target_sparse_ratio,
            "avg_achieved_sparsity": avg_achieved_sparsity,
            "phase": self.phase,
            "optimal_pairs": optimal_pairs,
            "calibration_type": "length_based_dynamic",
        }

    def _apply_length_based_calibration(self, modules: list[nn.Module], a_parameter: float):
        """Apply calibrated 'a' parameter to modules for length-based thresholds.

        Args:
            modules: List of attention modules
            a_parameter: Calibrated 'a' value for λ = a / length
        """
        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                method = module._sparse_method_instance
                # Store the calibrated parameter
                if hasattr(method, "set_length_based_threshold"):
                    method.set_length_based_threshold(a_parameter)
                else:
                    # Store as attributes for methods to use
                    method.use_length_based_threshold = True
                    method.length_based_a = a_parameter

    def _enable_calibration_mode(self, modules: list[nn.Module], reset_history: bool = True):
        """Enable calibration mode on sparse attention modules.

        Args:
            modules: List of attention modules
            reset_history: Whether to reset stats history
        """
        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                method = module._sparse_method_instance
                if hasattr(method, "set_calibration_mode"):
                    method.set_calibration_mode(enabled=True, reset_history=reset_history)

    def _disable_calibration_mode(self, modules: list[nn.Module]):
        """Disable calibration mode on sparse attention modules.

        Args:
            modules: List of attention modules
        """
        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                method = module._sparse_method_instance
                if hasattr(method, "set_calibration_mode"):
                    method.set_calibration_mode(enabled=False, reset_history=False)

    def _extract_calibration_stats(self, modules: list[nn.Module]) -> list[dict]:
        """Extract per-sample calibration statistics from modules.

        Args:
            modules: List of attention modules

        Returns:
            List of per-sample statistics across all modules
        """
        # Collect stats from all modules and aggregate
        all_module_stats = []
        enabled_count = 0

        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                method = module._sparse_method_instance
                if hasattr(method, "get_calibration_stats"):
                    module_stats = method.get_calibration_stats()

                    if module_stats:  # Only count if we got stats
                        enabled_count += 1
                        all_module_stats.append(module_stats)

        # Average sparsity across all modules for each sample
        if not all_module_stats:
            return []

        num_samples = len(all_module_stats[0])
        aggregated_stats = []

        for sample_idx in range(num_samples):
            # Collect sparsity from all modules for this sample
            sparsities = []
            sample_length = 0

            for module_stats in all_module_stats:
                if sample_idx < len(module_stats):
                    sample_stat = module_stats[sample_idx]
                    sparsities.append(sample_stat.get("sparsity", 0.0))
                    if not sample_length and "sample_length" in sample_stat:
                        sample_length = sample_stat["sample_length"]

            # Calculate average sparsity across all modules
            avg_sparsity = np.mean(sparsities) if sparsities else 0.0

            aggregated_stats.append(
                {
                    "sparsity": avg_sparsity,
                    "sample_length": sample_length,
                }
            )

        return aggregated_stats

    def _set_threshold(self, modules: list[nn.Module], threshold: float):
        """Set threshold on sparse attention modules."""
        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                if hasattr(module._sparse_method_instance, "threshold"):
                    module._sparse_method_instance.threshold = threshold

    def _reset_stats(self, modules: list[nn.Module]):
        """Reset statistics in sparse attention modules."""
        for module in modules:
            if hasattr(module, "_sparse_method_instance"):
                # Reset any internal stats if present
                if hasattr(module._sparse_method_instance, "stats"):
                    module._sparse_method_instance.stats = {
                        "total_blocks": 0,
                        "sparse_blocks": 0,
                        "correction_factors": [],
                        "phase_counts": {"prefill": 0, "decode": 0},
                    }

    def _calculate_sparsity(self, modules: list[nn.Module]) -> float:
        """Calculate average sparsity across modules."""
        total_sparsity = 0.0
        count = 0

        for module in modules:
            if hasattr(module, "_sparse_method_instance") and hasattr(
                module._sparse_method_instance, "stats"
            ):
                stats = module._sparse_method_instance.stats
                if stats.get("total_blocks", 0) > 0:
                    sparsity = stats["sparse_blocks"] / stats["total_blocks"]
                    total_sparsity += sparsity
                    count += 1

        return total_sparsity / count if count > 0 else 0.0
