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
from collections.abc import Callable, Iterable

import torch.nn as nn

from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from .config import PEFTConfig
from .lora.layer import LoRAModule, LoRAModuleRegistry

__all__ = [
    "freeze_base_weights",
    "freeze_lora_weights",
    "replace_lora_module",
    "unfreeze_base_weights",
    "unfreeze_lora_weights",
]


def convert_to_peft_model(model: ModelLikeModule, config: PEFTConfig) -> ConvertReturnType:
    """Convert the model to a peft one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    replace_lora_module(model, version=ModeloptStateManager(model).state_version, config=config)

    metadata = {}
    add_adapter(model, config)
    update_grads(model, config)

    return model, metadata


def restore_peft_model(
    model: ModelLikeModule, config: PEFTConfig, metadata: MetadataDict
) -> nn.Module:
    model, _ = convert_to_peft_model(model, config)
    return model


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


def update_peft_metadata(model: nn.Module, config: PEFTConfig, metadata: MetadataDict) -> None:
    """Placeholder for the metadata-related function; not needed in this mode."""


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
                if _matches(name, wildcard_or_filter_func):
                    module.update_layer_lora(
                        adapter_name,
                        adapter_setting,
                    )

    return model


def _matches(
    name: str,
    patterns: str | Callable[[str], bool] | Iterable[str | Callable[[str], bool]] | None,
    *,
    allow_callable: bool = True,
) -> bool:
    if patterns is None:
        return True

    if isinstance(patterns, (str, bytes)):
        patterns_iter: Iterable[str | Callable[[str], bool]] = (patterns,)
    elif callable(patterns):
        if not allow_callable:
            raise TypeError("Callable patterns are not supported in this context.")
        patterns_iter = (patterns,)
    elif isinstance(patterns, Iterable):
        patterns_iter = tuple(patterns)
    else:
        raise TypeError(f"Unsupported pattern type: {type(patterns)}")

    for pattern in patterns_iter:
        if isinstance(pattern, (str, bytes)):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif callable(pattern):
            if not allow_callable:
                raise TypeError("Callable patterns are not supported in this context.")
            if pattern(name):
                return True
        else:
            raise TypeError(f"Unsupported pattern type: {type(pattern)}")

    return False


def _iter_lora_modules(model, layer_patterns=None):
    for module_name, module in model.named_modules():
        if isinstance(module, LoRAModule) and _matches(module_name, layer_patterns):
            yield module_name, module


def _set_base_requires_grad(model, *, requires_grad: bool, layer_patterns=None):
    for _, module in _iter_lora_modules(model, layer_patterns):
        lora_param_ids = {
            id(param)
            for adapter in module._lora_adapters.values()
            for submodule in ("lora_a", "lora_b")
            for _, param in adapter[submodule].named_parameters()
        }
        for _, param in module.named_parameters():
            if id(param) in lora_param_ids:
                continue
            param.requires_grad = requires_grad


def _iter_adapter_names(module, adapter_patterns=None):
    for adapter_name in module._lora_adapters:
        if _matches(adapter_name, adapter_patterns, allow_callable=False):
            yield adapter_name


def _set_lora_requires_grad(
    model, *, requires_grad: bool, layer_patterns=None, adapter_patterns=None
):
    for _, module in _iter_lora_modules(model, layer_patterns):
        for adapter_name in _iter_adapter_names(module, adapter_patterns):
            adapter = module._lora_adapters[adapter_name]
            for submodule in (adapter["lora_a"], adapter["lora_b"]):
                for _, param in submodule.named_parameters():
                    param.requires_grad = requires_grad


def freeze_base_weights(model, *, layer_patterns=None):
    """Freeze base model weights to prevent gradient updates during training.

    This function sets requires_grad=False for all base model parameters in LoRA modules,
    while keeping LoRA adapter parameters trainable. Useful for LoRA fine-tuning where
    only adapter weights should be updated.

    Args:
        model: Model containing LoRA modules whose base weights should be frozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
    """
    _set_base_requires_grad(model, requires_grad=False, layer_patterns=layer_patterns)


def unfreeze_base_weights(model, *, layer_patterns=None):
    """Unfreeze base model weights to allow gradient updates during training.

    This function sets requires_grad=True for all base model parameters in LoRA modules.
    Useful when you want to fine-tune both base model and LoRA adapter weights together.

    Args:
        model: Model containing LoRA modules whose base weights should be unfrozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
    """
    _set_base_requires_grad(model, requires_grad=True, layer_patterns=layer_patterns)


def freeze_lora_weights(model, *, layer_patterns=None, adapter_patterns=None):
    """Freeze LoRA adapter weights to prevent gradient updates during training.

    This function sets requires_grad=False for LoRA adapter parameters (lora_a and lora_b).
    Useful when you want to train only the base model weights or evaluate the model
    without updating LoRA adapters.

    Args:
        model: Model containing LoRA modules whose adapter weights should be frozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
        adapter_patterns: Optional patterns (str or Iterable) to match specific adapter
            names. If provided, only adapters matching these patterns will be affected.
            Supports Unix-style wildcards
    """
    _set_lora_requires_grad(
        model,
        requires_grad=False,
        layer_patterns=layer_patterns,
        adapter_patterns=adapter_patterns,
    )


def unfreeze_lora_weights(model, *, layer_patterns=None, adapter_patterns=None):
    """Unfreeze LoRA adapter weights to allow gradient updates during training.

    This function sets requires_grad=True for LoRA adapter parameters (lora_a and lora_b).
    This is the typical setting for LoRA fine-tuning where adapter weights are trained.

    Args:
        model: Model containing LoRA modules whose adapter weights should be unfrozen
        layer_patterns: Optional patterns (str, bytes, or Iterable) to match specific
            layer names. If provided, only layers matching these patterns will be affected.
            Supports Unix-style wildcards (e.g., "*.linear", "transformer.*")
        adapter_patterns: Optional patterns (str or Iterable) to match specific adapter
            names. If provided, only adapters matching these patterns will be affected.
            Supports Unix-style wildcards
    """
    _set_lora_requires_grad(
        model,
        requires_grad=True,
        layer_patterns=layer_patterns,
        adapter_patterns=adapter_patterns,
    )


def update_grads(model, config: PEFTConfig):
    """Update gradient computation settings based on PEFTConfig.

    This function configures which model parameters should have gradients computed
    based on the freeze settings in the PEFTConfig. It's typically called during
    model initialization or when switching training configurations.

    Args:
        model: Model containing LoRA modules to configure
        config: PEFTConfig instance with freeze_base_model and freeze_lora_weights settings
            - If config.freeze_base_model is True, base weights will have requires_grad=False
            - If config.freeze_lora_weights is True, LoRA weights will have requires_grad=False
    """
    _set_base_requires_grad(model, requires_grad=not config.freeze_base_model)
    _set_lora_requires_grad(model, requires_grad=not config.freeze_lora_weights)
