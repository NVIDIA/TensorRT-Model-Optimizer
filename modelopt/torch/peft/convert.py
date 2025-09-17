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

"""User-facing quantization API."""

import fnmatch
from typing import Any

import torch.nn as nn

# import modelopt.torch.quantization as mtq
from modelopt.torch.opt import apply_mode

# from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.opt.conversion import ModeloptStateManager

# from modelopt.torch.opt.searcher import ForwardLoop
# from modelopt.torch.opt.utils import forward_with_reshard
from modelopt.torch.peft.config import PEFTConfig

from .lora.layer import LoRAModule

# from . import config
# from .algorithms import AutoQuantizeSearcher
# from .config import QuantizeAlgoCfgType
# from .conversion import set_quantizer_attribute
from .mode import PEFTModeRegistry

# from .nn import QuantModule, TensorQuantizer

# __all__ = [
#     "auto_quantize",
#     "calibrate",
#     "disable_quantizer",
#     "enable_quantizer",
#     "fold_weight",
#     "postprocess_amax",
#     "print_quant_summary",
#     "quantize",
# ]


def update_model(
    model: nn.Module,
    config: dict[str, Any | PEFTConfig],
):
    # TODO: deal with extra state, how to save the model
    # TODO: sharded dict
    # TODO: metadate
    # TODO: how to restore the model
    apply_mode(model, mode=[("peft", config)], registry=PEFTModeRegistry)
    return add_adapter(model, config)


def add_adapter(model, config):
    adapter_cfg = config["adapter_cfg"]
    adapter_name = config["adapter_name"]

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
                module.update_layer_lora(adapter_name, adapter_setting["rank"])

    # Update the metadata in ModeloptStateManager after adding adapters
    _update_peft_metadata_in_state(model)
    return model


def _update_peft_metadata_in_state(model: nn.Module) -> None:
    """Update the PEFT metadata in the ModeloptStateManager.

    This function updates the metadata to reflect the current state of LoRA adapters
    after they have been added or modified.
    """
    # Check if model has ModeloptStateManager (has been converted with peft mode)
    if not ModeloptStateManager.is_converted(model):
        return

    # Get the state manager
    manager = ModeloptStateManager(model)

    # Get current PEFT state from all LoRA modules
    current_peft_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            from modelopt.torch.utils import get_unwrapped_name

            unwrapped_name = get_unwrapped_name(name)
            current_peft_state[unwrapped_name] = module.get_peft_state()

    # Update the metadata in the last mode state (which should be 'peft')
    if manager._state and manager._last_metadata is not None:
        manager._last_metadata["peft_state"] = current_peft_state
