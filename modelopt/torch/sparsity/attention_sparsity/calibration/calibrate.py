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

"""Calibration functions for sparse attention."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from transformers import AutoTokenizer

from ..config import CalibrationConfig, SparseAttentionConfig
from ..nn.sparse_attention import SparseAttentionModule
from .calibrator import DynamicThresholdCalibrator
from .dataset import RulerDatasetBuilder


def _extract_tokenizer_from_model(model: nn.Module) -> str:
    """Extract tokenizer name/path from model config.

    Args:
        model: Model to extract tokenizer from

    Returns:
        Tokenizer name or path

    Raises:
        ValueError: If tokenizer cannot be determined
    """
    # Try to get from model config
    if hasattr(model, "config"):
        config = model.config

        # Check for _name_or_path
        if hasattr(config, "_name_or_path") and config._name_or_path:
            return config._name_or_path

        # Check for model_type
        if hasattr(config, "model_type"):
            # Map common model types to tokenizer names
            model_type_mapping = {
                "llama": "meta-llama/Llama-3-8B",
                "gpt2": "gpt2",
                "gpt_neo": "EleutherAI/gpt-neo-1.3B",
                "gptj": "EleutherAI/gpt-j-6B",
                "opt": "facebook/opt-1.3b",
                "bloom": "bigscience/bloom-1b7",
                "mistral": "mistralai/Mistral-7B-v0.1",
            }
            if config.model_type in model_type_mapping:
                return model_type_mapping[config.model_type]

    # Fallback: use GPT2 as universal tokenizer
    import warnings

    warnings.warn(
        "Could not determine model tokenizer. Using 'gpt2' as fallback. "
        "For best results, ensure model has proper config._name_or_path."
    )
    return "gpt2"


def _extract_calibration_config(
    config: SparseAttentionConfig | dict[str, Any],
) -> CalibrationConfig | None:
    """Extract and validate calibration config from sparse attention config.

    Args:
        config: Sparse attention configuration (SparseAttentionConfig or dict)

    Returns:
        Validated CalibrationConfig or None if not found
    """
    # Extract sparse_cfg
    sparse_cfg = (
        config.sparse_cfg
        if hasattr(config, "sparse_cfg")
        else config.get("sparse_cfg", {})
        if isinstance(config, dict)
        else {}
    )

    # Find calibration in pattern configurations
    calib_dict = next(
        (
            cfg["calibration"]
            for cfg in sparse_cfg.values()
            if isinstance(cfg, dict) and "calibration" in cfg
        ),
        None,
    )

    # Validate through Pydantic if found
    return CalibrationConfig(**calib_dict) if calib_dict else None


def create_calibration_forward_loop(
    calibration_data: list[dict[str, Any]],
    tokenizer_name_or_path: str,
    batch_size: int = 1,
) -> Callable:
    """Create a forward loop function from calibration dataset.

    Similar to create_forward_loop() in quantization but for RULER data.

    Args:
        calibration_data: List of calibration samples with 'input' (raw text) and 'length'
        tokenizer_name_or_path: Path or name of HuggingFace tokenizer
        batch_size: Batch size (default 1 for calibration)

    Returns:
        A forward loop function that takes model as argument
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def forward_loop(model):
        """Forward calibration data through model."""
        device = next(model.parameters()).device

        for sample in calibration_data:
            # RULER dataset always provides 'input' field with raw text
            if "input" not in sample:
                continue

            tokenized = tokenizer(
                sample["input"],
                return_tensors="pt",
                truncation=True,
                max_length=sample["length"],
            )
            forward_batch = {k: v.to(device) for k, v in tokenized.items()}

            # Forward pass
            model(**forward_batch)

    return forward_loop


def calibrate_sparse_attention(
    model: nn.Module,
    config: SparseAttentionConfig | dict[str, Any],
    forward_loop: Callable | None = None,
) -> dict[str, Any]:
    """Calibrate sparse attention parameters for optimal sparsity.

    Args:
        model: Model with sparse attention modules
        config: Sparse attention configuration
        forward_loop: Callable that forwards calibration data through model.
                     If None, will auto-generate RULER dataset and create forward_loop.

    Returns:
        Dictionary with calibration results
    """
    # Extract and validate calibration config
    calib_config = _extract_calibration_config(config)
    if not calib_config:
        return {}

    # Generate forward_loop if not provided
    if forward_loop is None:
        # Extract tokenizer from model
        tokenizer = _extract_tokenizer_from_model(model)

        # Build RULER dataset
        builder = RulerDatasetBuilder(
            samples=calib_config.samples,
            max_seqlen=calib_config.max_seqlen,
            tokenizer_name_or_path=tokenizer,
            num_length_bins=calib_config.num_length_bins,
            max_length_filter=int(calib_config.max_seqlen * 1.2),
        )
        calibration_data = builder.build_calibration_dataset()
        print(f"Generated {len(calibration_data)} calibration samples")

        # Create forward loop from calibration data with tokenization
        forward_loop = create_calibration_forward_loop(calibration_data, tokenizer, batch_size=1)

    if forward_loop is None:
        raise ValueError("No forward_loop provided and no dataset configuration found")

    # Extract sparse attention modules
    sparse_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SparseAttentionModule)
    ]

    if not sparse_modules:
        print("No sparse attention modules found for calibration")
        return {}

    print(f"Calibrating {len(sparse_modules)} sparse attention modules together...")

    # Ensure pytorch backend is used for calibration (required for hook-based stats collection)
    # Triton backend uses fused kernels that bypass our patching
    original_backends = {}
    for module_name, module in sparse_modules:
        if hasattr(module, "_sparse_method_instance"):
            method = module._sparse_method_instance
            if hasattr(method, "backend"):
                original_backends[module_name] = method.backend
                if method.backend != "pytorch":
                    print(
                        f"  Switching {module_name} from {method.backend} to pytorch backend for calibration"
                    )
                    method.backend = "pytorch"

    # Create ONE calibrator to calibrate all modules at once
    calibrator = DynamicThresholdCalibrator(
        target_sparse_ratio=calib_config.target_sparse_ratio,
        max_iterations=50,
        tolerance=0.05,
        phase=calib_config.phase,
    )

    # Run calibration once for all modules
    calibration_result = calibrator.calibrate(model, forward_loop)

    # Apply calibration results to all modules
    if "a_parameter" in calibration_result:
        print(
            f"\nApplying calibrated parameter a={calibration_result['a_parameter']:.6f} "
            f"to {len(sparse_modules)} modules"
        )
    elif "optimal_threshold" in calibration_result:
        print(
            f"\nApplying threshold={calibration_result['optimal_threshold']:.2e} "
            f"to {len(sparse_modules)} modules"
        )

    results = {}
    for module_name, module in sparse_modules:
        if "a_parameter" in calibration_result:
            # Length-based calibration
            method = module._sparse_method_instance
            method.use_length_based_threshold = True
            method.length_based_a = calibration_result["a_parameter"]
        elif "optimal_threshold" in calibration_result:
            # Fixed threshold calibration (fallback)
            module._sparse_method_instance.threshold = calibration_result["optimal_threshold"]
        if "optimal_k" in calibration_result:
            module._sparse_method_instance.k = calibration_result["optimal_k"]
        if "block_pattern" in calibration_result:
            module._sparse_method_instance._pattern_mask = calibration_result["block_pattern"]

        # Enable stats collection for inference
        method = module._sparse_method_instance
        if hasattr(method, "collect_stats"):
            method.collect_stats = True

        # Populate stats with calibration results for immediate availability
        if hasattr(method, "stats") and "avg_achieved_sparsity" in calibration_result:
            method.stats = {
                "sparsity": calibration_result["avg_achieved_sparsity"],
                "phase": "calibration",
                "total_blocks": calibration_result.get("num_samples", 0),
            }

        results[module_name] = calibration_result

    # Restore original backends after calibration
    for module_name, original_backend in original_backends.items():
        for name, module in sparse_modules:
            if name == module_name and hasattr(module, "_sparse_method_instance"):
                method = module._sparse_method_instance
                if hasattr(method, "backend") and method.backend != original_backend:
                    print(f"  Restoring {module_name} to {original_backend} backend")
                    method.backend = original_backend

    return {"calibration_results": results}
