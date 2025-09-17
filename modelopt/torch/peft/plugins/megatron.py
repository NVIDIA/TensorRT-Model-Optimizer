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

"""Megatron-Core specific PEFT/LoRA plugins."""

from typing import Any

import torch

from modelopt.torch.opt.plugins.megatron import register_modelopt_extra_state_callbacks
from modelopt.torch.peft.lora.layer import LoRAModuleRegistry

# Import MegatronModule if available
try:
    from megatron.core.transformer.module import MegatronModule

    MEGATRON_AVAILABLE = True
except ImportError:
    MegatronModule = None
    MEGATRON_AVAILABLE = False

from ..custom import CUSTOM_MODEL_PLUGINS

__all__ = []


def lora_module_get_extra_state(self) -> dict:
    """Get extra state for LoRA modules.

    This is called by the modelopt extra state framework to gather
    PEFT/LoRA state for distributed checkpointing.
    """
    # LoRAModule already has get_extra_state method
    return self.get_extra_state()


def lora_module_set_extra_state(self, state: Any):
    """Set extra state for LoRA modules.

    This is called by the modelopt extra state framework to restore
    PEFT/LoRA state from distributed checkpoints.
    """
    # LoRAModule already has set_extra_state method
    self.set_extra_state(state)


def megatron_replace_lora_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model PEFT/LoRA support.

    This callback is called before the LoRAModule replacement to configure
    distributed checkpointing support. For each MegatronModule:
    1. We enable heterogeneous distributed checkpointing
    2. We register extra_state callbacks for all LoRAModule submodules
    """
    if not MEGATRON_AVAILABLE:
        return

    def _register_extra_state_callbacks(model: torch.nn.Module):
        """Register extra state callbacks for LoRA modules."""
        for name, module in model.named_modules():
            if type(module) in LoRAModuleRegistry:
                # This module will be replaced as a LoRAModule
                register_modelopt_extra_state_callbacks(
                    module,
                    lora_module_get_extra_state,
                    lora_module_set_extra_state,
                )

    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            # Enable heterogeneous distributed checkpointing
            if hasattr(module, "config") and hasattr(
                module.config, "hetereogenous_dist_checkpoint"
            ):
                module.config.hetereogenous_dist_checkpoint = True
            _register_extra_state_callbacks(module)


# Register the hook
CUSTOM_MODEL_PLUGINS.add(megatron_replace_lora_module_hook)
