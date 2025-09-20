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

import torch

# Import MegatronModule if available
try:
    from megatron.core.transformer.module import MegatronModule

    MEGATRON_AVAILABLE = True
except ImportError:
    MegatronModule = None
    MEGATRON_AVAILABLE = False

from ..custom import CUSTOM_MODEL_PLUGINS

__all__ = []


def megatron_replace_lora_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model PEFT/LoRA support.

    This callback is called before the LoRAModule replacement to configure
    distributed checkpointing support. For each MegatronModule:
    1. We enable heterogeneous distributed checkpointing

    Note: LoRAModule already has built-in get_extra_state and set_extra_state methods,
    so we don't need to register callbacks for them.
    """
    if not MEGATRON_AVAILABLE:
        return

    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            # Enable heterogeneous distributed checkpointing
            if hasattr(module, "config") and hasattr(
                module.config, "hetereogenous_dist_checkpoint"
            ):
                module.config.hetereogenous_dist_checkpoint = True


# Register the hook
CUSTOM_MODEL_PLUGINS.add(megatron_replace_lora_module_hook)
