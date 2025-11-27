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

import dataclasses
import functools
import json
import os
import pathlib
import re
import warnings
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)


def block_config_to_str(block_config: BlockConfig | dict[str, Any] | None) -> str | None:
    if block_config is None:
        return None
    rep = ""
    if dataclasses.is_dataclass(block_config):
        block_config = dataclasses.asdict(block_config)
    for subblock_name in ["attention", "ffn"]:
        subblock_config = block_config[subblock_name]
        rep += subblock_config_to_str(subblock_config, subblock_name)
    return rep


def subblock_config_to_str(
    subblock_config: FFNConfig | AttentionConfig | dict[str, Any] | None,
    subblock_name: None | str = None,
) -> str | None:
    if subblock_config is None:
        return None
    subblock_name = (
        "ffn"
        if isinstance(subblock_config, FFNConfig)
        else "mamba"
        if isinstance(subblock_config, AttentionConfig) and subblock_config.is_mamba
        else "attention"
        if isinstance(subblock_config, AttentionConfig)
        else subblock_name
    )
    assert subblock_name is not None, "Must provide subblock_name if subblock_config is a dict."

    if dataclasses.is_dataclass(subblock_config):
        subblock_config = dataclasses.asdict(subblock_config)

    if subblock_name == "attention" and subblock_config.get("mamba") is not None:
        subblock_name = "mamba"

    if subblock_name == "ffn" and subblock_config.get("moe") is not None:
        subblock_name = "moe"

    rep = f"  {subblock_name}"
    if subblock_config.get("no_op"):
        rep += "  no_op".ljust(8)
    elif subblock_config.get("replace_with_linear"):
        rep += "  linear".ljust(8)
    elif subblock_name == "ffn":
        intermediate_size = subblock_config["intermediate_size"]
        rep += f"  intermediate_{intermediate_size}".ljust(8)
    elif subblock_name == "attention":
        n_heads_in_group = subblock_config["n_heads_in_group"]
        rep += f"  gqa_{n_heads_in_group}".ljust(8)
    elif subblock_name == "mamba":
        mamba_num_heads = subblock_config["mamba"]["num_heads"]
        mamba_head_dim = subblock_config["mamba"]["head_dim"]
        rep += f"  num_heads_{mamba_num_heads}  head_dim_{mamba_head_dim}".ljust(8)
    elif subblock_name == "moe":
        moe_num_local_experts = subblock_config["moe"]["num_local_experts"]
        moe_expert_intermediate_dim = subblock_config["moe"]["expert_intermediate_dim"]
        shared_expert_intermediate_dim = subblock_config["moe"]["shared_expert_intermediate_dim"]
        num_experts_per_tok = subblock_config["moe"]["num_experts_per_tok"]
        rep += f"  num_experts_{moe_num_local_experts}  expert_intermediate_dim_{moe_expert_intermediate_dim}  shared_expert_intermediate_dim_{shared_expert_intermediate_dim}  num_experts_per_tok_{num_experts_per_tok}".ljust(
            8
        )
    else:
        raise ValueError(f"subblock_config_to_str: unrecognized subblock_name: {subblock_name}.")

    return rep


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
                model = LLaMA(model_config)
            model.load_state_dict(torch.load("llama-lit/7B/lit-llama.pth"))"""

        self.device = device
        self.dtype = dtype

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)
