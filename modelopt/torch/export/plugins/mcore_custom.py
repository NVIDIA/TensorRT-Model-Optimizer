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


"""Custom mapping utility."""

import json
import os
from typing import Any

import torch
from safetensors.torch import save_file

from modelopt.torch.utils import import_plugin

with import_plugin("megatron"):
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
    )

MAX_SAFETENSOR_SIZE = 4000000000
PER_RANK_PARTITIONS = 10

COL_PARALLEL = {"sharding_dim": 0}
ROW_PARALLEL = {"sharding_dim": 1}


class CustomModuleMapping:
    """A custom module mapping from Megatron Core to its HF counter part."""

    def __init__(
        self, func_name: str = "", target_name_or_prefix: str = "", func_kwargs: dict[str, Any] = {}
    ):
        """Create a custom module mapping."""
        self.func_name = func_name
        self.target_name_or_prefix = target_name_or_prefix
        self.func_kwargs = func_kwargs


def save_safetensors(state_dict, save_directory: str | os.PathLike):
    """Save safetensors with pipeline model parallel support."""
    pp_rank = get_pipeline_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_world_size()

    all_safetensors = [{} for i in range(10)]

    local_idx = 0
    local_total_size = 0

    for key, val in state_dict.items():
        tensor_size = val.numel() * val.element_size()
        if local_total_size + tensor_size > MAX_SAFETENSOR_SIZE and local_idx < PER_RANK_PARTITIONS:
            local_idx += 1
            local_total_size = 0
        all_safetensors[local_idx][key] = val
        local_total_size += tensor_size

    for local_idx, tensors in enumerate(all_safetensors):
        global_idx = PER_RANK_PARTITIONS * pp_rank + local_idx
        global_count = PER_RANK_PARTITIONS * pp_size
        filename = f"model-{global_idx + 1:05d}-of-{global_count:05d}"
        meta_filename = filename + ".json"
        ckpt_filename = filename + ".safetensors"

        weight_map = {}
        local_total_size = 0

        for key, val in tensors.items():
            local_total_size += val.numel() * val.element_size()
            weight_map[key] = ckpt_filename

        with open(save_directory + "/" + meta_filename, "w") as f:
            json.dump(
                {"metadata": {"total_size": local_total_size}, "weight_map": weight_map},
                f,
                indent=4,
            )
        save_file(tensors, save_directory + "/" + ckpt_filename, metadata={"format": "pt"})

    # Barrier to ensure all ranks have written the metadata
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        global_count = PER_RANK_PARTITIONS * pp_size
        safetensor_index = {
            "metadata": {"total_size": 0},
            "weight_map": {},
        }
        for global_idx in range(global_count):
            meta_filename = f"model-{global_idx + 1:05d}-of-{global_count:05d}.json"
            with open(save_directory + "/" + meta_filename) as f:
                shard = json.load(f)
            safetensor_index["metadata"]["total_size"] += shard["metadata"]["total_size"]
            safetensor_index["weight_map"].update(shard["weight_map"])

        with open(save_directory + "/model.safetensors.index.json", "w") as f:
            json.dump(safetensor_index, f, indent=4)
