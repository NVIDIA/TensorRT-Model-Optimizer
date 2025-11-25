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
"""Export functions for vLLM fakequant."""

import os
from pathlib import Path

import torch
import torch.nn as nn

from modelopt.torch.export.layer_utils import is_quantlinear
from modelopt.torch.export.model_config import QUANTIZATION_NONE
from modelopt.torch.quantization.utils import get_quantizer_state_dict


def export_hf_vllm_fq_checkpoint(
    model: nn.Module,
    export_dir: Path | str,
) -> dict[str, torch.Tensor]:
    """Exports the torch model weights and amax values separately.

    This function:
    1. Extracts amax values for calibration
    2. Deletes all quantizer parameters from state dict to store only weights in original dtype

    Args:
        model: The quantized model to export
        export_dir: Directory to save the amax values

    Returns:
        post_state_dict: Dict containing quantized weights
    """
    amax_dict = {
        name + "._amax": param["_amax"].detach().clone().cpu()
        for name, param in get_quantizer_state_dict(model).items()
        if "_amax" in param
    }

    # remove quantizer from model
    for _, module in model.named_modules():
        if is_quantlinear(module):
            delattr(module, "weight_quantizer")
            delattr(module, "input_quantizer")
            delattr(module, "output_quantizer")
            module.export()
    torch.save(amax_dict, f"{export_dir}/quant_amax.pth")
    return model.state_dict()


def get_mcore_vllm_fq_quantized_state(
    module: torch.nn.Module, name_to_value: dict, dtype: torch.dtype = torch.bfloat16
):
    """Return a state_dict, quantization format, and block_size of the quantized module.

    Args:
        module: The target module to perform real quantization.
        name_to_value: The dictionary to store the quantized state.
        dtype: The default data type.

    Returns:
        Tuple: state dict, quantization format, and block_size of the quantized module.

    """
    qformat: str = QUANTIZATION_NONE
    block_size = 0

    for name, param in get_quantizer_state_dict(module).items():
        if "_amax" in param:
            name_to_value[name + "._amax"] = param["_amax"].to(dtype).cpu()
    return name_to_value, qformat, block_size


def gather_mcore_vllm_fq_quantized_state_dict(
    state_dict: dict[str, torch.Tensor], save_directory: str | os.PathLike
):
    """Gather all quantized state dict from all ranks and save them to a file.

    Args:
        state_dict: The state dictionary of the module.
        save_directory: The directory to save the quantized state dict.

    Returns:
        The state dictionary of the module without quantized state.
    """
    amax_state_dict = {
        k: v.detach().clone().cpu() for k, v in state_dict.items() if k.endswith("_amax")
    }

    # Gather all amax dicts to rank 0
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if rank == 0:
        # Rank 0 will collect all amax values
        all_amax_dicts = [None] * world_size
        torch.distributed.gather_object(amax_state_dict, all_amax_dicts, dst=0)

        # Merge all amax dicts into one
        merged_amax_dict = {}
        for amax_dict in all_amax_dicts:
            if amax_dict is not None:
                merged_amax_dict.update(amax_dict)

        print(f"Total amax entries from all ranks: {len(merged_amax_dict.keys())}")
        torch.save(merged_amax_dict, save_directory + "/quant_amax.pth")
    else:
        # Other ranks just send their amax values
        torch.distributed.gather_object(amax_state_dict, None, dst=0)

    torch.distributed.barrier()

    # remove amax values from state_dict
    return {k: v for k, v in state_dict.items() if not k.endswith("_amax")}
