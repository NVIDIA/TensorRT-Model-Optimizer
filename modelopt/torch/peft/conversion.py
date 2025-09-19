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

"""PEFT conversion and restore utilities for LoRA modules."""

import fnmatch
from typing import Any

import torch.nn as nn

from modelopt.torch.opt.conversion import ApplyModeError, ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils import get_unwrapped_name

from .config import PEFTConfig
from .lora.layer import LoRAModule, LoRAModuleRegistry

__all__ = [
    "replace_lora_module",
    "update_peft_metadata_in_model",
]


def convert_to_peft_model(model: ModelLikeModule, config: PEFTConfig) -> ConvertReturnType:
    """Convert the model to a peft one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # TODO: Replace to LoRA module
    replace_lora_module(model, version=ModeloptStateManager(model).state_version, config=config)

    metadata = {}
    add_adapter(model, config)
    # Should return adapaters, active_adapters
    update_peft_metadata(model, config, metadata)

    return model, metadata


def restore_peft_model(
    model: ModelLikeModule, config: PEFTConfig, metadata: MetadataDict
) -> nn.Module:
    convert_to_peft_model(model, config)
    return restore_peft_state(model, metadata)


def restore_peft_state(model: ModelLikeModule, metadata: MetadataDict):
    """Restore PEFT state from metadata or extra_state.

    For backward compatibility, we check metadata first. For distributed
    checkpoints (NeMo-MCore), the state will be in extra_state of each LoRAModule
    and will be restored automatically via set_extra_state() during load_state_dict().

    Args:
        model: Model with LoRA modules to restore
        metadata: Metadata dictionary that may contain peft_state
    Returns:
        The model with restored PEFT state
    """
    if "peft_state" not in metadata:
        # For distributed checkpoints (NeMo-MCore), peft_state is stored
        # in each LoRAModule's extra_state and will be restored via
        # set_extra_state() during load_state_dict()
        return model

    # Legacy path: restore from metadata
    peft_state_dict = metadata["peft_state"]
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            unwrapped_name = get_unwrapped_name(name)
            if unwrapped_name in peft_state_dict:
                try:
                    module.set_from_peft_state(peft_state_dict[unwrapped_name])
                except Exception as e:
                    raise ApplyModeError(f"Failed to restore PEFT state for module {name}: {e}")

    return model


def update_peft_metadata(model: nn.Module, config: PEFTConfig, metadata: MetadataDict) -> None:
    """Update the PEFT/LoRA state in the metadata dict."""
    metadata["peft_state"] = peft_state(model)


def peft_state(model: nn.Module) -> dict[str, Any]:
    return {
        get_unwrapped_name(n): m.get_peft_state()
        for n, m in model.named_modules()
        if isinstance(m, LoRAModule)
    }


def replace_lora_module(
    model: nn.Module, version=None, config: PEFTConfig = None, registry=LoRAModuleRegistry
):
    """Recursively replace the module with LoRA module."""
    # Register custom plugins (e.g., for Megatron distributed checkpointing)
    from .custom import register_custom_model_plugins_on_the_fly

    register_custom_model_plugins_on_the_fly(model)

    if type(model) in registry:
        model = registry.convert(model)
    _replace_lora_module(model, version=version, registry=registry)


def export_peft_model(model: nn.Module, config):
    raise NotImplementedError("Exporting a peft model is not supported yet.")


def restore_export_peft_model(model: nn.Module, config, metadata: MetadataDict):
    raise NotImplementedError("Restoring a peft & exported model is not supported yet.")


def _replace_lora_module(model: nn.Module, version=None, registry=LoRAModuleRegistry):
    for name, child in model.named_children():
        if type(child) in registry:
            lora_module = registry.convert(child)
            setattr(model, name, lora_module)

        _replace_lora_module(getattr(model, name), version=version, registry=registry)


def update_peft_metadata_in_model(model: nn.Module) -> None:
    """Update the PEFT metadata in the model's ModeloptStateManager.

    This function should be called after manually modifying LoRA adapters to ensure
    the metadata stored in the ModeloptStateManager reflects the current state.

    Args:
        model: Model with LoRA modules whose metadata needs updating
    Example:
        >>> # After manually adding/modifying adapters
        >>> for module in model.modules():
        ...     if isinstance(module, LoRAModule):
        ...         module.update_layer_lora("custom_adapter", rank=32)
        >>> # Update metadata to reflect changes
        >>> update_peft_metadata_in_model(model)
    """
    # Check if model has ModeloptStateManager (has been converted with peft mode)
    if not ModeloptStateManager.is_converted(model):
        return

    # Get the state manager
    manager = ModeloptStateManager(model)

    # Update the metadata with current PEFT state
    if manager._state and manager._last_metadata is not None:
        manager._last_metadata["peft_state"] = peft_state(model)


def add_adapter(model, config: PEFTConfig):
    """Add a new LoRA adapter to the model.

    Args:
        model: Model with LoRA modules to add adapters to
        config: PEFTConfig instance containing adapter_cfg and adapter_name

    Returns:
        The model with the new adapter added
    """
    adapter_cfg = config.adapter_cfg
    adapter_name = config.adapter_name

    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            for wildcard_or_filter_func, adapter_setting in adapter_cfg.items():
                if isinstance(wildcard_or_filter_func, str):
                    if not fnmatch.fnmatch(name, wildcard_or_filter_func):
                        continue
                elif callable(wildcard_or_filter_func):
                    if not wildcard_or_filter_func(name):
                        continue
                else:
                    raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")
                module.update_layer_lora(
                    adapter_name,
                    adapter_setting,
                )

    return model


def _update_peft_metadata_in_state(model: nn.Module) -> None:
    """Update the PEFT metadata in the ModeloptStateManager.

    This function updates the metadata to reflect the current state of LoRA adapters
    after they have been added or modified.
    """
    if not ModeloptStateManager.is_converted(model):
        return

    manager = ModeloptStateManager(model)

    current_peft_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            from modelopt.torch.utils import get_unwrapped_name

            unwrapped_name = get_unwrapped_name(name)
            current_peft_state[unwrapped_name] = module.get_peft_state()

    if manager._state and manager._last_metadata is not None:
        manager._last_metadata["peft_state"] = current_peft_state
