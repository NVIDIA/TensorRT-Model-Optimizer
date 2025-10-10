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

"""Configuration classes for sparse attention optimization."""

from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

# Type definitions for sparse configuration
SparseAttributeConfig = dict[str, Any]  # Configuration for a specific pattern

SparseAttentionCfgType = dict[
    str | Callable,  # Pattern or callable for matching modules
    SparseAttributeConfig,  # Configuration dict with threshold, enable, etc.
]


class SparseAttentionAttributeConfig(ModeloptBaseConfig):
    """Sparse attention attribute configuration for pattern-based module config."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable sparse attention.",
        description="If True, enables sparse attention. If False, bypasses sparsity.",
    )

    method: str = ModeloptField(
        default="flash_softmax_skip",
        title="Sparse attention method.",
        description="The sparse attention method to use (e.g., 'flash_softmax_skip').",
    )

    threshold: float | dict[str, float] = ModeloptField(
        default=1e-3,
        title="Sparsity threshold.",
        description=(
            "Threshold for determining which attention values to skip. "
            "Can be a float or dict with phase-specific values."
        ),
    )

    br: int = ModeloptField(
        default=128,
        title="Block row size.",
        description="Block row size for block-wise sparsity in Flash Attention.",
    )

    bc: int = ModeloptField(
        default=128,
        title="Block column size.",
        description="Block column size for block-wise sparsity in Flash Attention.",
    )

    collect_stats: bool = ModeloptField(
        default=False,
        title="Collect statistics.",
        description="Whether to collect sparsity statistics during forward pass.",
    )

    backend: str = ModeloptField(
        default="pytorch",
        title="Backend implementation.",
        description=(
            "Backend to use for sparse attention computation. "
            "Only 'pytorch' is supported, which uses softmax patching with F.softmax. "
            "Requires model to be loaded with attn_implementation='eager'."
        ),
    )

    is_causal: bool = ModeloptField(
        default=True,
        title="Causal attention flag.",
        description=(
            "Whether the model uses causal (autoregressive) attention. "
            "If True, sparsity statistics are calculated over the lower triangle only. "
            "Defaults to True for decoder-only models like GPT, LLaMA, etc."
        ),
    )

    calibration: dict | None = ModeloptField(
        default=None,
        title="Calibration configuration",
        description=(
            "Calibration settings for this pattern. "
            "If provided, enables automatic threshold calibration. "
            "Only one pattern should have calibration enabled."
        ),
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate method is a string."""
        if not isinstance(v, str):
            raise ValueError("method must be a string")
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        """Validate backend is pytorch."""
        if v != "pytorch":
            raise ValueError(
                f"Invalid backend: {v}. Only 'pytorch' backend is supported. "
                f"Model must be loaded with attn_implementation='eager'."
            )
        return v

    @field_validator("br", "bc")
    @classmethod
    def validate_block_size(cls, v):
        """Validate block sizes are positive integers."""
        if v <= 0:
            raise ValueError(f"Block size must be positive, got {v}")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v):
        """Validate threshold is in valid range (0, 1) or dict with valid phases."""
        if isinstance(v, dict):
            # Validate phase keys
            valid_phases = {"prefill", "decode", "default"}
            invalid_keys = set(v.keys()) - valid_phases
            if invalid_keys:
                raise ValueError(
                    f"Invalid threshold phases: {invalid_keys}. Valid phases: {valid_phases}"
                )
            # Validate all values are in range (0, 1)
            for phase, threshold in v.items():
                if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
                    raise ValueError(
                        f"Threshold for phase '{phase}' must be in range (0, 1), got {threshold}"
                    )
        elif isinstance(v, (int, float)):
            if v <= 0 or v >= 1:
                raise ValueError(f"Threshold must be in range (0, 1), got {v}")
        else:
            raise ValueError(f"Threshold must be a number in range (0, 1) or dict, got {type(v)}")
        return v


class CalibrationConfig(ModeloptBaseConfig):
    """Configuration for automatic threshold calibration using RULER dataset.

    Calibration learns a dynamic threshold λ = scale_factor / sequence_length that
    achieves target sparsity. Only supports prefill phase (seq_len > 1).
    """

    target_sparse_ratio: float = ModeloptField(
        default=0.5,
        title="Target sparsity ratio",
        description="Target ratio of sparse attention blocks (0.0 to 1.0).",
    )

    samples: int = ModeloptField(
        default=24,
        title="Calibration samples",
        description="Total number of RULER samples for calibration (distributed across length bins).",
    )

    max_seqlen: int = ModeloptField(
        default=32768,
        title="Maximum sequence length",
        description="Maximum sequence length for calibration (length bins auto-generated as powers of 2).",
    )

    num_length_bins: int = ModeloptField(
        default=4,
        title="Number of length bins",
        description="Number of length bins to generate (hidden parameter, default: 4).",
    )

    threshold_trials: list[float] | None = ModeloptField(
        default=None,
        title="Threshold trials",
        description=(
            "List of threshold values to test during calibration. "
            "If None, uses default: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]"
        ),
    )

    @field_validator("threshold_trials")
    @classmethod
    def validate_threshold_trials(cls, v):
        """Validate threshold_trials are in valid range."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(f"threshold_trials must be a list, got {type(v)}")
            if len(v) == 0:
                raise ValueError("threshold_trials must not be empty")
            for threshold in v:
                if not isinstance(threshold, (int, float)):
                    raise ValueError(f"All threshold_trials must be numbers, got {type(threshold)}")
                if threshold <= 0 or threshold >= 1:
                    raise ValueError(
                        f"All threshold_trials must be in range (0, 1), got {threshold}"
                    )
        return v

    @field_validator("target_sparse_ratio")
    @classmethod
    def validate_target_sparse_ratio(cls, v):
        """Validate target sparsity ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"target_sparse_ratio must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v):
        """Validate samples is positive."""
        if v <= 0:
            raise ValueError(f"samples must be positive, got {v}")
        return v

    @field_validator("max_seqlen")
    @classmethod
    def validate_max_seqlen(cls, v):
        """Validate max_seqlen is at least 1024."""
        if v < 1024:
            raise ValueError(f"max_seqlen must be >= 1024, got {v}")
        return v

    @field_validator("num_length_bins")
    @classmethod
    def validate_num_length_bins(cls, v):
        """Validate num_length_bins is positive."""
        if v <= 0:
            raise ValueError(f"num_length_bins must be positive, got {v}")
        return v


# Pre-defined Sparse Attention Configuration
# Default configuration with block-wise sparsity optimized for Flash Attention
SKIP_SOFTMAX_DEFAULT = {
    "method": "flash_softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "threshold": {
                "prefill": 1e-3,  # More aggressive during prefill
                "decode": 1e-4,  # Conservative during decode
            },
            "br": 128,  # Flash Attention block rows
            "bc": 128,  # Flash Attention block columns
            "backend": "pytorch",  # Only pytorch backend supported
            "enable": True,
        },
        "default": {"enable": False},
    },
}


# Configuration with RULER calibration
# Note: threshold field is omitted - calibration determines dynamic threshold λ = a / length
# The calibrated threshold adapts to sequence length for optimal sparsity
SKIP_SOFTMAX_CALIB = {
    "method": "flash_softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "br": 128,
            "bc": 128,
            "backend": "pytorch",  # Only pytorch backend supported
            "enable": True,
            "calibration": {
                "target_sparse_ratio": 0.5,
                "samples": 120,
                "max_seqlen": 8192,
            },
        },
        "default": {"enable": False},
    },
}


class SparseAttentionConfig(ModeloptBaseConfig):
    """Base configuration for sparse attention optimization.

    This base configuration provides the common structure for all sparse
    attention methods and supports pattern-based layer configuration.
    """

    # Method selection
    method: str = Field("flash_softmax_skip", description="Sparse attention method to use")

    # Statistics collection
    collect_stats: bool = Field(
        False, description="Whether to collect sparsity statistics during forward pass"
    )

    # Pattern-based sparse configuration (similar to quant_cfg in quantization)
    sparse_cfg: SparseAttentionCfgType = ModeloptField(
        default={"*attention*": {"enable": True}, "default": {"enable": False}},
        title="Sparse attention configuration",
        description="Pattern-based configuration for sparse attention. Keys are patterns to match module names, "
        "values are configuration dicts with parameters like 'threshold', 'enable', and 'calibration'.",
        validate_default=True,
    )

    # Export configuration
    export_format: str | None = Field(
        None, description="Export format for sparse attention (e.g., 'onnx', 'tensorrt')"
    )


class FlashSoftmaxSkipConfig(SparseAttentionConfig):
    """Configuration for Flash Attention-aware softmax skip sparse attention."""

    # Override method to default to flash_softmax_skip
    method: str = Field(
        "flash_softmax_skip", description="Sparse attention method (fixed to flash_softmax_skip)"
    )

    # Override sparse_cfg with flash_softmax_skip specific defaults
    sparse_cfg: SparseAttentionCfgType = ModeloptField(
        default={
            "*attention*": {
                "threshold": {"prefill": 1e-3, "decode": 1e-5},
                "br": 128,  # Flash Attention block rows
                "bc": 128,  # Flash Attention block columns
                "backend": "pytorch",  # Only pytorch backend supported
                "enable": True,
            },
            "default": {"enable": False},
        },
        title="Flash softmax skip sparse configuration",
        description="Pattern-based configuration with flash_softmax_skip specific defaults. "
        "Includes FA block sizes (br, bc) and correction factor settings.",
        validate_default=True,
    )


__all__ = [
    "SKIP_SOFTMAX_CALIB",
    "SKIP_SOFTMAX_DEFAULT",
    "CalibrationConfig",
    "FlashSoftmaxSkipConfig",
    "SparseAttentionAttributeConfig",
    "SparseAttentionCfgType",
    "SparseAttentionConfig",
    "SparseAttributeConfig",
]
