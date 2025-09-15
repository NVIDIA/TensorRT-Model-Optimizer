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

"""Quantization conversion/restore utilities."""

import fnmatch
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch.nn as nn

from modelopt.torch.opt.conversion import ApplyModeError, ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils import get_unwrapped_name

from .config import (
    PEFTConfig,
    _QuantizeExportConfig,
)
from .lora.layer import LoRAModuleRegistry

__all__ = [
    "replace_lora_module",
]


def convert_to_peft_model(model: ModelLikeModule, config: PEFTConfig) -> ConvertReturnType:
    """Convert the model to a quantized one as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # TODO: Replace to LoRA module
    replace_lora_module(model, version=ModeloptStateManager(model).state_version, config=config)
    # set_quantizer_by_cfg(model, config.get("quant_cfg", {}))

    metadata = {}
    # update_quantize_metadata(model, config, metadata)

    return model, metadata

def restore_peft_model(
    model: ModelLikeModule, config: PEFTConfig, metadata: MetadataDict
) -> nn.Module:
    #TODO: implemente the restore logic
    pass



def update_peft_metadata(
    model: nn.Module, config: PEFTConfig, metadata: MetadataDict
) -> None:
    """Update the quantizer state in the metadata dict."""
    pass


def replace_lora_module(model: nn.Module, version=None, config: PEFTConfig = None, registry=LoRAModuleRegistry):
    """Recursively replace the module with quantized module."""
    #TODO: register the extra state for megatron-lm

    if type(model) in registry:
        model = registry.convert(model)
    _replace_lora_module(model, version=version, registry=registry)

def export_peft_model(model: nn.Module, config):
    """Export the quantized model to a quantized model."""
    raise NotImplementedError("Exporting a quantized model is not supported yet.")


def restore_export_peft_model(
    model: nn.Module, config, metadata: MetadataDict
):
    """Restores the quantized model from the given state dict."""
    raise NotImplementedError("Restoring a quantized & exported model is not supported yet.")


def _replace_lora_module(model: nn.Module, version=None,registry=LoRAModuleRegistry):
    for name, child in model.named_children():
        if type(child) in registry:
            lora_module = registry.convert(child)
            setattr(model, name, lora_module)

        _replace_lora_module(getattr(model, name), version=version, registry=registry)


def export_quantized_model(model: nn.Module, config: _QuantizeExportConfig) -> ConvertReturnType:
    """Export the quantized model to a quantized model."""
    raise NotImplementedError("Exporting a quantized model is not supported yet.")


def restore_export_quantized_model(
    model: nn.Module, config: _QuantizeExportConfig, metadata: MetadataDict
) -> nn.Module:
    """Restores the quantized model from the given state dict."""
    raise NotImplementedError("Restoring a quantized & exported model is not supported yet.")
