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

"""User-facing PEFT API for LoRA module conversion and adapter management."""

from typing import Any

import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.peft.config import PEFTConfig
from modelopt.torch.peft.conversion import add_adapter

from .lora.layer import LoRAModule
from .mode import PEFTModeRegistry


def update_model(
    model: nn.Module,
    config: dict[str, Any] | PEFTConfig,
):
    """Update model with PEFT/LoRA adapters.

    This function handles both initial PEFT conversion and adding additional adapters:
    - First call: Converts modules to LoRAModules and adds the first adapter
    - Subsequent calls: Adds new adapters to existing LoRAModules

    Args:
        model: The model to update
        config: PEFT configuration dict or PEFTConfig instance

    Returns:
        The updated model with LoRA adapters
    """
    # Check if model is already in PEFT mode by looking for LoRA modules
    if not is_peft_model(model):
        apply_mode(model, mode=[("peft", config)], registry=PEFTModeRegistry)
    else:
        if not isinstance(config, PEFTConfig):
            config = PEFTConfig(**config)
        add_adapter(model, config)
    return model


def is_peft_model(model: nn.Module) -> bool:
    """Check if the model has been converted to PEFT/LoRA model.

    This function checks if any modules in the model are LoRAModule instances,
    which indicates the model has already been converted to PEFT mode.

    Args:
        model: The model to check

    Returns:
        True if the model contains LoRA modules, False otherwise
    """
    return any(isinstance(module, LoRAModule) for _, module in model.named_modules())
