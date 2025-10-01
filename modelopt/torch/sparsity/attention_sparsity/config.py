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
    """Sparse attention attribute configuration.

    Similar to QuantizerAttributeConfig, this defines the attributes for
    sparse attention modules.
    """

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
            "'pytorch' uses mask-based approach with softmax patching (compatible everywhere). "
            "'triton' uses fused skip kernel (faster, requires GPU and Triton)."
        ),
    )

    calibration: dict | None = ModeloptField(
        default=None,
        title="Calibration configuration.",
        description=(
            "Optional calibration configuration for this pattern. "
            "If provided, enables automatic threshold calibration."
        ),
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate that method is a registered sparse attention method."""
        if not isinstance(v, str):
            raise ValueError("method must be a string")
        # Note: We don't validate against the registry here because it's populated
        # at import time. Runtime validation happens via get_sparse_method().
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        """Validate backend is a supported option."""
        valid_backends = {"pytorch", "triton"}
        if v not in valid_backends:
            raise ValueError(f"Invalid backend: {v}. Valid backends: {valid_backends}")
        return v

    @field_validator("br", "bc")
    @classmethod
    def validate_block_size(cls, v):
        """Validate block sizes are positive integers."""
        if v <= 0:
            raise ValueError(f"Block size must be positive, got {v}")
        # Block sizes should typically be reasonable (not too large)
        if v > 1024:
            import warnings

            warnings.warn(
                f"Block size {v} is unusually large. "
                f"Typical Flash Attention block sizes are 64, 128, or 256."
            )
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v):
        """Validate threshold is positive number or dict with valid phases."""
        if isinstance(v, dict):
            # Validate phase keys
            valid_phases = {"prefill", "decode", "default"}
            invalid_keys = set(v.keys()) - valid_phases
            if invalid_keys:
                raise ValueError(
                    f"Invalid threshold phases: {invalid_keys}. Valid phases: {valid_phases}"
                )
            # Validate all values are positive floats
            for phase, threshold in v.items():
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    raise ValueError(
                        f"Threshold for phase '{phase}' must be positive, got {threshold}"
                    )
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError(f"Threshold must be positive, got {v}")
        else:
            raise ValueError(f"Threshold must be a positive number or dict, got {type(v)}")
        return v


class CalibrationConfig(ModeloptBaseConfig):
    """Configuration for sparse attention calibration.

    Simplified configuration for automatic threshold calibration using RULER dataset.
    Calibration is enabled by the presence of this config in sparse_cfg patterns.

    Examples:
        # Use all defaults
        calibration = {}

        # Override specific values
        calibration = {
            "target_sparse_ratio": 0.7,
            "samples": 96,
            "max_seqlen": 65536,
        }
    """

    target_sparse_ratio: float = ModeloptField(
        default=0.5,
        title="Target sparsity ratio",
        description="Target ratio of sparse attention blocks (0.0 to 1.0).",
    )

    samples: int = ModeloptField(
        default=48,
        title="Calibration samples",
        description="Total number of RULER samples for calibration (distributed across length bins).",
    )

    max_seqlen: int = ModeloptField(
        default=32768,
        title="Maximum sequence length",
        description="Maximum sequence length for calibration (length bins auto-generated as powers of 2).",
    )

    phase: str | None = ModeloptField(
        default=None,
        title="Calibration phase",
        description="Phase to calibrate for: None (both), 'prefill', or 'decode'.",
    )

    num_length_bins: int = ModeloptField(
        default=4,
        title="Number of length bins",
        description="Number of length bins to generate (hidden parameter, default: 4).",
    )

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

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v):
        """Validate phase is supported."""
        if v is not None and v not in {"prefill", "decode"}:
            raise ValueError(f"Invalid phase: {v}. Valid phases: None, 'prefill', 'decode'")
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
            "backend": "pytorch",
            "enable": True,
        },
        "default": {"enable": False},
    },
}


# Configuration with RULER calibration
SKIP_SOFTMAX_CALIB = {
    "method": "flash_softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "threshold": {
                "prefill": 1e-3,  # More aggressive during prefill
                "decode": 1e-4,  # Conservative during decode
            },
            "br": 128,
            "bc": 128,
            "backend": "pytorch",
            "enable": True,
            "calibration": {
                "target_sparse_ratio": 0.5,
                "samples": 24,
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
        "values are configuration dicts with parameters like 'threshold' and 'enable'.",
        validate_default=True,
    )

    # Export configuration
    export_format: str | None = Field(
        None, description="Export format for sparse attention (e.g., 'onnx', 'tensorrt')"
    )


class FlashSoftmaxSkipConfig(SparseAttentionConfig):
    """Configuration for Flash Attention-aware softmax skip sparse attention.

    This configuration uses flash_softmax_skip as the default method and includes
    Flash Attention specific parameters like block sizes and correction factor.
    """

    # Override method to default to flash_softmax_skip
    method: str = Field(
        "flash_softmax_skip", description="Sparse attention method (fixed to flash_softmax_skip)"
    )

    # Override sparse_cfg with flash_softmax_skip specific defaults
    sparse_cfg: SparseAttentionCfgType = ModeloptField(
        default={
            "*attention*": {
                "threshold": {"prefill": 1e-3, "decode": 1e-5},
                "br": 64,  # Flash Attention block rows
                "bc": 64,  # Flash Attention block columns
                "backend": "pytorch",
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
