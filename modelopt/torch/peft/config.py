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

"""Configuration classes for PEFT methods."""

import math
import pickle  # nosec B403 - Only checking picklability
from collections.abc import Callable

import torch.nn.init as init
from pydantic import field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["ExportPEFTConfig", "PEFTAttributeConfig", "PEFTConfig"]


def kaiming_init(weight):
    """Default initialization for LoRA A matrix using Kaiming uniform."""
    return init.kaiming_uniform_(weight, a=math.sqrt(5))


def zero_init(weight):
    """Default initialization for LoRA B matrix using zeros."""
    return init.zeros_(weight)


class PEFTAttributeConfig(ModeloptBaseConfig):
    """Configuration for PEFT adapter attributes."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable adapter",
        description="If True, enables the adapter. If False, by-passes the adapter.",
    )

    rank: int = ModeloptField(
        default=64,
        title="LoRA rank",
        description=(
            "The rank (dimension) of the LoRA matrices. "
            "Higher rank allows more expressiveness but uses more memory."
        ),
    )

    scale: float = ModeloptField(
        default=1.0,
        title="LoRA scaling factor",
        description="Scaling factor for the LoRA output. Controls the magnitude of the adaptation.",
    )

    lora_a_init: Callable[[object], None] | None = ModeloptField(
        default=kaiming_init,
        title="LoRA A matrix initializer",
        description="Custom initialization function for LoRA A matrix. Default to Kaiming uniform initialization.",
    )

    lora_b_init: Callable[[object], None] | None = ModeloptField(
        default=zero_init,
        title="LoRA B matrix initializer",
        description="Custom initialization function for LoRA B matrix. Default to zero initialization.",
    )

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v):
        """Validate rank is positive."""
        if v < 1:
            raise ValueError("rank must be a positive integer")
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate scale is positive."""
        if v <= 0:
            raise ValueError("scale must be a positive number")
        return v

    @model_validator(mode="after")
    def validate_init_functions(self):
        """Validate initialization functions are callable and picklable."""
        if self.lora_a_init is not None and not callable(self.lora_a_init):
            raise ValueError("lora_a_init must be callable")
        if self.lora_b_init is not None and not callable(self.lora_b_init):
            raise ValueError("lora_b_init must be callable")
        if self.lora_a_init is not None:
            try:
                _del = pickle.dumps(self.lora_a_init)
                del _del
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                raise ValueError(
                    f"lora_a_init cannot be pickled: {e}. "
                    "Please use a module-level function instead of a lambda or nested function."
                )
        if self.lora_b_init is not None:
            try:
                _del = pickle.dumps(self.lora_b_init)
                del _del
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                raise ValueError(
                    f"lora_b_init cannot be pickled: {e}. "
                    "Please use a module-level function instead of a lambda or nested function."
                )
        return self


# Type alias for adapter configuration
PEFTAdapterCfgType = dict[str | Callable, PEFTAttributeConfig | dict]


class PEFTConfig(ModeloptBaseConfig):
    """Default configuration for ``peft`` mode."""

    adapter_name: str = ModeloptField(
        default="default",
        title="Adapter name",
        description="Name of the adapter to create or update.",
        validate_default=True,
    )

    adapter_cfg: PEFTAdapterCfgType = ModeloptField(
        default={"default": {"rank": 128}},
        title="Adapter configuration",
        description="Configuration for adapters. Maps module patterns to PEFTAttributeConfig or dict.",
        validate_default=True,
    )

    adapter_type: str = ModeloptField(
        default="lora",
        title="Adapter type",
        description="Type of PEFT adapter to use. Currently only 'lora' is supported.",
        validate_default=True,
    )

    freeze_base_model: bool = ModeloptField(
        default=True,
        title="Freeze base weights during training",
        description="Whether to freeze the base model weights; in most cases, this should be set to True.",
        validate_default=True,
    )

    freeze_lora_weights: bool = ModeloptField(
        default=True,
        title="Placeholder",
        description="Placeholder",
        validate_default=True,
    )

    @field_validator("adapter_type")
    @classmethod
    def validate_adapter_type(cls, v):
        """Validate adapter type."""
        if v not in ["lora"]:
            raise ValueError(f"Unsupported adapter type: {v}. Only 'lora' is currently supported.")
        return v

    @field_validator("adapter_cfg")
    @classmethod
    def validate_adapter_cfg(cls, v):
        """Validate and convert adapter configurations."""
        validated_cfg = {}
        for key, value in v.items():
            if isinstance(value, dict) and not isinstance(value, PEFTAttributeConfig):
                # Convert dict to PEFTAttributeConfig to trigger validation
                try:
                    validated_cfg[key] = PEFTAttributeConfig(**value)
                except Exception as e:
                    raise ValueError(f"Invalid adapter configuration for '{key}': {e}")
            else:
                validated_cfg[key] = value
        return validated_cfg


class ExportPEFTConfig(ModeloptBaseConfig):
    """An empty config."""
