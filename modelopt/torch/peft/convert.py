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
import inspect
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn

# import modelopt.torch.quantization as mtq
from modelopt.torch.opt import apply_mode
# from modelopt.torch.opt.searcher import ForwardLoop
# from modelopt.torch.opt.utils import forward_with_reshard
from modelopt.torch.peft.config import PEFTConfig
# from modelopt.torch.quantization.conversion import set_quantizer_by_cfg

# from . import config
# from .algorithms import AutoQuantizeSearcher
# from .config import QuantizeAlgoCfgType
# from .conversion import set_quantizer_attribute
from .mode import PEFTModeRegistry
from .lora.layer import LoRAModule
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
    #TODO: deal with extra state, how to save the model
    #TODO: sharded dict
    #TODO: metadate
    #TODO: how to restore the model
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
    return model