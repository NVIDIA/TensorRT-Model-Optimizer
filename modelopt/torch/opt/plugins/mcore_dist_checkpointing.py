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

"""Megatron core distributed checkpointing plugin for sharded ``modelopt_state``."""

# TODO: Add unit tests for this plugin
import copy
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.network import SUPPORTED_WRAPPERS

SUPPORTED_MODELS = [GPTModel, MambaModel]
SUPPORTED_WRAPPERS[Float16Module] = "module"


def _remap_quantizer_state_with_prefix(
    local_state: dict[str, Any], prefix: str, quantizer_state: dict[str, Any]
) -> None:
    """Remap quantizer_state using the given prefix.

    The prefix is usually the local state_dict prefix. The remapped local_state
    will be stored as a ShardedObject with its global state_dict prefix attached.
    For quantizer_state, we further remove _amax and _pre_quant_scale which are
    stored as a part of the model sharded_state_dict.

    Args:
        local_state: the local quantizer_state to be sharded that all has the same local prefix
        prefix: the quantizer_state prefix to extract
        quantizer_state: all local quantizer_state
    """
    prefix_with_dot = prefix + "."

    for key, val in quantizer_state.items():
        if not key.startswith(prefix_with_dot):
            continue
        new_key = key[len(prefix) :]
        if "_amax" in val:
            val.pop("_amax")
        if "_pre_quant_scale" in val:
            val.pop("_pre_quant_scale")
        if new_key in local_state:
            local_state[new_key]["quantizer_state"] = val
        else:
            local_state[new_key] = {"quantizer_state": val}


def _remap_sparsity_state_with_prefix(
    local_state: dict[str, Any],
    prefix: str,
    subnet_config: dict[str, Any],
) -> None:
    """Remap sparsity subnet_config using the given prefix.

    The prefix is usually the local state_dict prefix. The remapped local_state
    will be stored as a ShardedObject with its global state_dict prefix attached.

    Args:
        local_state: the local subnet_config to be sharded that all has the same local prefix
        prefix: the subnet_config prefix to extract
        subnet_config: all local subnet_config for sparsity
    """
    prefix_with_dot = prefix + "."

    for key, val in subnet_config.items():
        if not key.startswith(prefix_with_dot):
            continue
        new_key = key[len(prefix) :]
        if new_key in local_state:
            local_state[new_key]["subnet_config"] = val
        else:
            local_state[new_key] = {"subnet_config": val}


def remove_modelopt_state_metadata(
    modelopt_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove metadata from the modelopt_state.

    The metadata of the modelopt_state contains keys which may change with different pipeline
    parallelism. As a result, the metadata must be stored as several ShardedObject with
    global and local layer offset mapping.

    Args:
        modelopt_state: the state_dict that contains all algorithms have have been applied
            to the given model.
    """
    quantizer_state = {}
    subnet_config = {}
    # TODO (chenhany): the assumption here is that there is at most one quantize and sparsity mode
    if "modelopt_state_dict" in modelopt_state:
        for mode, config in modelopt_state["modelopt_state_dict"]:
            metadata = config.pop("metadata") if "metadata" in config else {}
            if "quantizer_state" in metadata:
                quantizer_state = metadata.get("quantizer_state")
            if "subnet_config" in metadata:
                subnet_config = metadata.get("subnet_config")

    return quantizer_state, subnet_config


def restore_modelopt_state_metadata(sharded_modelopt_state: dict[str, Any]) -> dict[str, Any]:
    """Restore the per rank modelopt_state after the metadata is loaded.

    Args:
        sharded_modelopt_state: the state_dict that contains all algorithms have have been applied
            to the given model.
    """
    modelopt_state_dict = sharded_modelopt_state.pop("modelopt_state_dict")
    modelopt_version = sharded_modelopt_state.pop("modelopt_version")
    quantizer_state = {}
    subnet_config = {}

    for key, val in sharded_modelopt_state.items():
        for subkey, subval in val.items():
            if "quantizer_state" in subval:
                quantizer_state[key + subkey] = subval["quantizer_state"]
            if "subnet_config" in subval:
                subnet_config[key + subkey] = subval["subnet_config"]

    for mode, config in modelopt_state_dict:
        if mode == "quantize":
            config["metadata"] = {"quantizer_state": quantizer_state}
        elif mode == "sparsegpt":
            config["metadata"] = {"subnet_config": subnet_config}
        else:
            config["metadata"] = {}

    return {"modelopt_state_dict": modelopt_state_dict, "modelopt_version": modelopt_version}


def _get_gpt_sharded_modelopt_state(
    num_layers: int = -1,
    num_medusa_heads: int = 0,
    num_eagle_layers: int = 0,
    model: Optional[torch.nn.Module] = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Return the sharded modelopt_state for a GPTModel.

    If a GPTModel is not provided, then the sharded modelopt_state will still return a
    dictionary of ShardedObject. This is used to load the sharded modelopt_state before
    the initialization of a GPTModel.

    Args:
        num_layers: number of decoder layers in the GPTModel
        num_medusa_heads: number of Medusa heads in the GPTModel
        num_eagle_layers: number of Eagle layers in the GPTModel
        model: optionally provide a GPTModel instance
        prefix: the prefix to add to the modelopt_state keys
    """
    if model is not None:
        num_layers = model.config.num_layers
        if hasattr(model, "medusa_heads") and model.medusa_heads is not None:
            num_medusa_heads = len(model.medusa_heads)
        if hasattr(model, "eagle_module") and model.eagle_module is not None:
            num_eagle_layers = len(model.eagle_module.decoder.layers)
    elif num_layers < 0:
        raise ValueError("Either num_layers or a model instance must be provided!")

    modelopt_state = {} if model is None else copy.deepcopy(mto.modelopt_state(model))
    quantizer_state, subnet_config = remove_modelopt_state_metadata(modelopt_state)

    # The sharded modelopt_state remains the part that is shared across all DP, TP, PP ranks.
    sharded_modelopt_state = modelopt_state
    sharded_offsets = []

    # Compute per pp rank num_layers and global_layer_offset
    local_num_layers = num_layers // get_pipeline_model_parallel_world_size()
    global_layer_offset = local_num_layers * get_pipeline_model_parallel_rank()

    # First pp stage
    if get_pipeline_model_parallel_rank() == 0:
        local_key = "embedding"
        global_key = f"{prefix}embedding"
        local_state = {}
        _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
        _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
        sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
            local_state,
            global_key,
            sharded_offsets,
        )

    # Last pp stage
    if get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1):
        local_key = "output_layer"
        global_key = f"{prefix}output_layer"
        local_state = {}
        _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
        _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
        sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
            local_state,
            global_key,
            sharded_offsets,
        )

        local_key = "medusa_heads"
        global_key = f"{prefix}medusa_heads"
        local_state = {}
        _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
        _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
        if num_medusa_heads > 0:
            sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
                local_state,
                global_key,
                sharded_offsets,
            )

        local_key = "eagle_module"
        global_key = f"{prefix}eagle_module"
        local_state = {}
        _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
        _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
        if num_eagle_layers > 0:
            sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
                local_state,
                global_key,
                sharded_offsets,
            )

    # Each pp rank owns some stages
    for local_layer_id in range(local_num_layers):
        global_layer_id = global_layer_offset + local_layer_id
        local_key = f"decoder.layers.{local_layer_id}"
        global_key = f"{prefix}decoder.layers.{global_layer_id}"
        local_state = {}
        _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
        _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
        sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
            local_state,
            global_key,
            sharded_offsets,
        )

    return sharded_modelopt_state


def get_sharded_modelopt_state(
    num_layers: int,
    model: torch.nn.Module,
    prefix: str = "",
    num_medusa_heads: int = 0,
    num_eagle_layers: int = 0,
) -> dict[str, Any]:
    """Return the sharded modelopt_state.

    Most of the modelopt_state is shared across all DP, TP, PP ranks. The metadata of each
    mode, however, is pipeline-parallelism (PP) dependent. The major reason is because the
    state is using the module name which contains decoder layer id that may change
    when pipeline-parallelism is changing. As a result, we need to store metadata
    with ShardedObject which provides the functionality to map the local state to
    the global state. For example, ``embedding.`` only exists in the first PP stage, and
    the same ``decoder.layers.{id}.`` module name can be found on all PP stages, but they
    are mapped to different global layer id.

    For example, for a quantizer_state metadata, ``{"decoder.layers.3.mlp.linear_fc1.input_quantizer": state}``,
    we need to store it as the following which maps local key ``decoder.layers.3`` to its global key
    ``decoder.layers.11`` because of PP:

    .. code-block:: python

        sharded_modelopt_state["decoder.layers.3"] = ShardedObject(
            {"mlp.linear_fc1.input_quantizer": {"quantizer_state": state}},
            global_key="decoder.layers.11",
        )

    To restore the metadata, we simply revert the process.

    Args:
        num_layers: number of decoder layers in the MCore model
        model: a MCore model instance
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
        num_medusa_heads: number of Medusa heads
    """
    support_list = SUPPORTED_MODELS

    if any([isinstance(model, arch) for arch in support_list]):
        return _get_gpt_sharded_modelopt_state(
            num_layers=num_layers,
            model=model,
            prefix=prefix,
            num_medusa_heads=num_medusa_heads,
        )
    else:
        raise ValueError(
            f"get_sharded_modelopt_state() only supports {SUPPORTED_MODELS}, received {type(model)}"
        )


def save_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Save modelopt_state as a part of the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the modelopt optimized model
        state_dict: the current modelopt optimized model state_dict to store
    """
    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) == 1:
        state_dict["modelopt_state"] = mto.modelopt_state(model[0])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict[f"modelopt_state_{i}"] = mto.modelopt_state(model[i])


def restore_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Restore modelopt_state from the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the model to restore the modelopt optimization
        state_dict: the loaded state_dict to extract
    """
    if (
        len(model) == 1
        and "modelopt_state" in state_dict
        and not mto.ModeloptStateManager.is_converted(model[0])
    ):
        model[0] = mto.restore_from_modelopt_state(model[0], state_dict["modelopt_state"])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            if f"modelopt_state_{i}" in state_dict and not mto.ModeloptStateManager.is_converted(
                model[i]
            ):
                model[i] = mto.restore_from_modelopt_state(
                    model[i], state_dict[f"modelopt_state_{i}"]
                )


def save_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: Union[str, Path],
    sharded_strategy: Optional[tuple[str, int]] = None,
    prefix: str = "",
) -> None:
    """Save modelopt_state in the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        sharded_strategy: configures sharded tensors saving behavior and backend
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
    """
    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")
    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if dist.is_master():
        os.makedirs(modelopt_checkpoint_name, exist_ok=True)

    dist_checkpointing.save(
        get_sharded_modelopt_state(-1, model[0], prefix),
        modelopt_checkpoint_name,
        sharded_strategy,
    )


def restore_sharded_modelopt_state(
    model: list[torch.nn.Module], checkpoint_name: Union[str, Path], prefix: str = ""
) -> None:
    """Restore modelopt_state from the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
    """
    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if os.path.exists(modelopt_checkpoint_name):
        if len(model) > 1:
            raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")
        if not mto.ModeloptStateManager.is_converted(model[0]):
            modelopt_state = restore_modelopt_state_metadata(
                dist_checkpointing.load(
                    get_sharded_modelopt_state(-1, model[0], prefix),
                    modelopt_checkpoint_name,
                )
            )
            model[0] = mto.restore_from_modelopt_state(model[0], modelopt_state)
