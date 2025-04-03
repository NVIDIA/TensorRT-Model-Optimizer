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
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import torch
from megatron.core import dist_checkpointing, mpu, parallel_state
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.utils import make_sharded_object_for_checkpoint
from packaging.version import Version

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


def _restore_legacy_modelopt_state_metadata(
    sharded_modelopt_state: dict[str, Any],
) -> dict[str, Any]:
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
        if "quantizer_state" in val:
            for subkey, subval in val["quantizer_state"].items():
                quantizer_state[key + subkey] = subval
        if "subnet_config" in val:
            for subkey, subval in val["subnet_config"].items():
                subnet_config[key + subkey] = subval

    for mode, config in modelopt_state_dict:
        if mode == "quantize":
            config["metadata"] = {"quantizer_state": quantizer_state}
        elif mode == "sparsegpt":
            config["metadata"] = {"subnet_config": subnet_config}
        else:
            config["metadata"] = {}

    return {"modelopt_state_dict": modelopt_state_dict, "modelopt_version": modelopt_version}


def _get_gpt_mamba_sharded_modelopt_state(
    model: torch.nn.Module,
    prefix: str = "",
    num_medusa_heads: int = 0,
    num_eagle_layers: int = 0,
    num_mtp_module: int = 0,
) -> dict[str, Any]:
    """Return the sharded modelopt_state for a GPTModel or MambaModel.

    Args:
        model: a GPTModel or MambaModel instance
        prefix: the prefix to add to the modelopt_state keys
        num_medusa_heads: number of Medusa heads in the model
        num_eagle_layers: number of Eagle layers in the model
        num_mtp_module: number of MTP in the model
    """
    # Update speculative decoding arguments if the model already had the
    # additional module.
    if hasattr(model, "medusa_heads") and model.medusa_heads is not None:
        num_medusa_heads = len(model.medusa_heads)
    if hasattr(model, "eagle_module") and model.eagle_module is not None:
        num_eagle_layers = len(model.eagle_module.decoder.layers)
    if hasattr(model, "mtp") and model.mtp is not None:
        num_mtp_module = len(model.mtp)

    # Extract the quantizer_state and subnet_config (sparsity). When resuming,
    # the model has no modelopt_state, then those are empty sets.
    modelopt_state = copy.deepcopy(mto.modelopt_state(model))
    quantizer_state, subnet_config = remove_modelopt_state_metadata(modelopt_state)

    # The sharded modelopt_state remains the part that is shared across all DP, TP, PP ranks.
    sharded_modelopt_state = modelopt_state
    sharded_offsets = []

    def _extract_module_name_from_key(key: str):
        """Extra the module name and separate the quantizer to be the new key."""
        if len(key) == 0:
            raise ValueError("_extract_module_name_from_key() got an emtpy key")
        attribute_names = key.split(".")
        if len(attribute_names) > 1:
            module_name = ".".join(attribute_names[:-1]) + "."
        else:
            module_name = ""
        new_key = attribute_names[-1]
        return module_name, new_key

    def _local_to_global_prefix_mapping(state_list):
        """Replacing the local prefix to the global prefix for pipeline parallelism."""

        def _prefix_remap(src, tar, state_list):
            for item in state_list:
                if item["global_key"] is not None:
                    continue
                elif item["local_key"].startswith(src):
                    item["global_key"] = item["local_key"].replace(src, tar)
            return state_list

        if get_pipeline_model_parallel_rank() == 0:
            _prefix_remap("embedding.", f"{prefix}embedding.", state_list)

        for local_layer_id, layer in enumerate(model.decoder.layers):
            global_layer_id = layer.layer_number - 1
            pp_local_key = f"decoder.layers.{local_layer_id}."
            pp_global_key = f"{prefix}decoder.layers.{global_layer_id}."
            _prefix_remap(pp_local_key, pp_global_key, state_list)

        if get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1):
            _prefix_remap("output_layer.", f"{prefix}output_layer.", state_list)
            _prefix_remap("medusa_heads.", f"{prefix}medusa_heads.", state_list)
            _prefix_remap("eagle_module.", f"{prefix}eagle_module.", state_list)
            _prefix_remap("mtp.", f"{prefix}mtp.", state_list)

        return state_list

    def _local_to_global_infix_mapping(state_list):
        """Replacing the local prefix to the global prefix for pipeline parallelism."""

        def _infix_remap(src, tar, state_list):
            for item in state_list:
                if src in item["global_key"]:
                    item["global_key"] = item["global_key"].replace(src, tar)
            return state_list

        ep_rank = parallel_state.get_expert_model_parallel_rank()
        # Each (pp, ep) rank owns some stages
        for local_layer_id, layer in enumerate(model.decoder.layers):
            experts = getattr(layer.mlp, "experts", None)
            if experts is None:
                continue
            num_local_experts = experts.num_local_experts
            for local_expert_id, local_expert in enumerate(experts.local_experts):
                global_expert_id = ep_rank * num_local_experts + local_expert_id
                ep_local_key = f"local_experts.{local_expert_id}."
                ep_global_key = f"local_experts.{global_expert_id}."
                _infix_remap(ep_local_key, ep_global_key, state_list)

        return state_list

    def _get_medusa_consolidate_state(model, num_medusa_heads):
        medusa_consolidate_state = {}
        if num_medusa_heads == 0:
            return medusa_consolidate_state

        for head_id in range(num_medusa_heads):
            medusa_head_prefix = "medusa_heads.{}.".format(head_id)
            medusa_consolidate_state[medusa_head_prefix + "lm_head."] = {}
            medusa_consolidate_state[medusa_head_prefix + "medusa_layers.0.linear."] = {}
        return medusa_consolidate_state

    def _get_eagle_consolidate_state(model, num_eagle_layers):
        eagle_consolidate_state = {}
        if num_eagle_layers == 0:
            return eagle_consolidate_state

        eagle_consolidate_state["eagle_module.fc."] = {}
        eagle_decoder_prefix = "eagle_module.decoder.layers."

        last_layer = model.decoder.layers[-1]
        for layer_id in range(num_eagle_layers):
            for name, module in last_layer.named_modules():
                if "Linear" in str(type(module)):
                    key = "{}{}.{}.".format(eagle_decoder_prefix, layer_id, name)
                    eagle_consolidate_state[key] = {}
        return eagle_consolidate_state

    def _get_mtp_consolidate_state(model, num_mtp_module):
        mtp_consolidate_state = {}
        if num_mtp_module == 0:
            return mtp_consolidate_state

        last_layer = model.decoder.layers[-1]
        for module_id in range(num_mtp_module):
            consolidate_state["mtp.{}.fc.".format(module_id)] = {}
            for name, module in last_layer.named_modules():
                if "Linear" in str(type(module)):
                    key = "mtp.{}.decoder.layers.0.{}.".format(module_id, name)
                    mtp_consolidate_state[key] = {}
        return mtp_consolidate_state

    consolidate_state = {}

    for name, module in model.named_modules():
        if "Linear" in str(type(module)):
            consolidate_state[name + "."] = {}

    if len(modelopt_state["modelopt_state_dict"]) > 0:
        for key, val in quantizer_state.items():
            module_name, new_key = _extract_module_name_from_key(key)
            if module_name not in consolidate_state:
                raise ValueError("consolidate_state has no {}!".format(module_name))
            if "quantizer_state" in consolidate_state[module_name]:
                consolidate_state[module_name]["quantizer_state"].update({new_key: val})
            else:
                consolidate_state[module_name]["quantizer_state"] = {new_key: val}

        for key, val in subnet_config.items():
            module_name, new_key = _extract_module_name_from_key(key)
            if module_name not in consolidate_state:
                raise ValueError("consolidate_state has no {}!".format(module_name))
            if "subnet_config" in consolidate_state[module_name]:
                consolidate_state[module_name]["subnet_config"].update({new_key: val})
            else:
                consolidate_state[module_name]["subnet_config"] = {new_key: val}
    else:
        # Create sharded objects for speculative decoding
        if get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1):
            medusa_consolidate_state = _get_medusa_consolidate_state(model, num_medusa_heads)
            consolidate_state.update(medusa_consolidate_state)
            eagle_consolidate_state = _get_eagle_consolidate_state(model, num_eagle_layers)
            consolidate_state.update(eagle_consolidate_state)
            mtp_consolidate_state = _get_mtp_consolidate_state(model, num_mtp_module)
            consolidate_state.update(mtp_consolidate_state)

    consolidate_state_list = []

    for key, val in consolidate_state.items():
        consolidate_state_list.append({"local_key": key, "global_key": None, "value": val})

    consolidate_state_list = _local_to_global_prefix_mapping(consolidate_state_list)

    for item in consolidate_state_list:
        if item["global_key"] is None:
            raise ValueError("{} does not have prefix mapping".format(item["local_key"]))

    consolidate_state_list = _local_to_global_infix_mapping(consolidate_state_list)

    for item in consolidate_state_list:
        sharded_object = make_sharded_object_for_checkpoint(
            item["value"],
            item["global_key"],
            sharded_offsets,
        )
        if "local_experts" in item["local_key"]:
            replica_id = sharded_object.replica_id
            sharded_object.replica_id = (
                *replica_id[:2],
                parallel_state.get_expert_data_parallel_rank(),
            )
        sharded_modelopt_state[item["local_key"]] = sharded_object

    return sharded_modelopt_state


def _get_legacy_sharded_modelopt_state(
    model: torch.nn.Module,
    prefix: str = "",
    num_medusa_heads: int = 0,
    num_eagle_layers: int = 0,
    num_mtp_module: int = 0,
) -> dict[str, Any]:
    if hasattr(model, "medusa_heads") and model.medusa_heads is not None:
        num_medusa_heads = len(model.medusa_heads)
    if hasattr(model, "eagle_module") and model.eagle_module is not None:
        num_eagle_layers = len(model.eagle_module.decoder.layers)
    if hasattr(model, "mtp") and model.mtp is not None:
        num_mtp_module = len(model.mtp)

    modelopt_state = copy.deepcopy(mto.modelopt_state(model))
    quantizer_state, subnet_config = remove_modelopt_state_metadata(modelopt_state)

    # The sharded modelopt_state remains the part that is shared across all DP, TP, PP ranks.
    sharded_modelopt_state = modelopt_state
    sharded_offsets = []

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

        # Medusa heads
        if num_medusa_heads > 0:
            local_key = "medusa_heads"
            global_key = f"{prefix}medusa_heads"
            local_state = {}
            _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
            _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
            sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
                local_state,
                global_key,
                sharded_offsets,
            )

        # Eagle module
        if num_eagle_layers > 0:
            local_key = "eagle_module"
            global_key = f"{prefix}eagle_module"
            local_state = {}
            _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
            _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
            sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
                local_state,
                global_key,
                sharded_offsets,
            )
        # MTP module
        if num_mtp_module > 0:
            local_key = "mtp"
            global_key = f"{prefix}mtp"
            local_state = {}
            _remap_quantizer_state_with_prefix(local_state, local_key, quantizer_state)
            _remap_sparsity_state_with_prefix(local_state, local_key, subnet_config)
            sharded_modelopt_state[local_key] = make_sharded_object_for_checkpoint(
                local_state,
                global_key,
                sharded_offsets,
            )

    # Each pp rank owns some stages
    for local_layer_id, layer in enumerate(model.decoder.layers):
        global_layer_id = layer.layer_number - 1
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
    model: torch.nn.Module,
    prefix: str = "",
    num_medusa_heads: int = 0,
    num_eagle_layers: int = 0,
    num_mtp_module: int = 0,
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
        model: a MCore model instance
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
        num_medusa_heads: number of Medusa heads
        num_eagle_layers: number of Eagle layers
        num_mtp_module: number of MTP modules
    """
    if any([isinstance(model, arch) for arch in SUPPORTED_MODELS]):
        return _get_gpt_mamba_sharded_modelopt_state(
            model=model,
            prefix=prefix,
            num_medusa_heads=num_medusa_heads,
            num_eagle_layers=num_eagle_layers,
            num_mtp_module=num_mtp_module,
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
        get_sharded_modelopt_state(model[0], prefix), modelopt_checkpoint_name, sharded_strategy
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
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"

    # Early return if the model already has a modelopt_state or the checkpoint does not exist.
    if not os.path.exists(modelopt_checkpoint_name) or mto.ModeloptStateManager.is_converted(
        model[0]
    ):
        return

    # Loading the common modelopt_state (replicated on all ranks)
    common_modelopt_state = torch.load(
        modelopt_checkpoint_name + "/" + COMMON_STATE_FNAME, weights_only=False
    )
    modelopt_load_version = common_modelopt_state["modelopt_version"]

    extra_kwargs = {}

    for mode, mode_cfg in common_modelopt_state["modelopt_state_dict"]:
        if mode == "medusa":
            extra_kwargs.update({"num_medusa_heads": mode_cfg["config"]["medusa_num_heads"]})
        if mode == "eagle":
            extra_kwargs.update({"num_eagle_layers": mode_cfg["config"]["eagle_num_layers"]})
        if mode == "mtp":
            extra_kwargs.update({"num_mtp_module": mode_cfg["config"]["mtp_num_module"]})

    is_dev_load_ver = "dev" in modelopt_load_version
    is_legacy_load_ver = Version(modelopt_load_version) < Version("0.27.0")

    if is_legacy_load_ver and not is_dev_load_ver:
        warnings.warn(
            "nvidia-modelopt>=0.27 updated how model_state is stored NeMo-MCore distributed checkpoint."
            "Newly generated checkpoints can no longer be loaded with older nvidia-modelopt<0.27."
            "Old checkpoints can still be loaded for next 2 releases and should be ported to"
            "the new format by creating a new checkpoint with nvidia-modelopt>=0.27."
        )
        legacy_sharded_modelopt_state = dist_checkpointing.load(
            _get_legacy_sharded_modelopt_state(model=model[0], prefix=prefix, **extra_kwargs),
            modelopt_checkpoint_name,
            get_default_load_sharded_strategy(modelopt_checkpoint_name),
        )
        modelopt_state = _restore_legacy_modelopt_state_metadata(legacy_sharded_modelopt_state)
        model[0] = mto.restore_from_modelopt_state(model[0], modelopt_state)
    else:
        sharded_modelopt_state = dist_checkpointing.load(
            get_sharded_modelopt_state(model=model[0], prefix=prefix, **extra_kwargs),
            modelopt_checkpoint_name,
            get_default_load_sharded_strategy(modelopt_checkpoint_name),
        )
        modelopt_state = restore_modelopt_state_metadata(sharded_modelopt_state)
        model[0] = mto.restore_from_modelopt_state(model[0], modelopt_state)
