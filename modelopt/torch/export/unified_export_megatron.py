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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


"""Code that export quantized Megatron Core models for deployment."""

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open, save_file

from modelopt import __version__
from modelopt.torch.utils import import_plugin

from .mcore_hf_export_map import all_mcore_hf_export_mapping, all_mcore_hf_import_mapping
from .model_config import QUANTIZATION_FP8
from .quant_utils import (
    get_activation_scaling_factor,
    get_kv_cache_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    to_quantized_weight,
)

with import_plugin("transformers", verbose=False):
    import transformers

has_mcore = False
with import_plugin("megatron"):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
    )

    has_mcore = True

__all__ = ["export_mcore_gpt_to_hf", "import_mcore_gpt_from_hf"]


def get_quantized_state(
    module: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """Return a state_dict, quantization format, and block_size of the module.

    Args:
        module: The target module to perform real quantization.
        dtype: The default data type.

    Returns:
        Tuple: state_dict, quantization format, and block_size of the module.
    """
    name_to_value = {}

    weight = module.weight.to(dtype).cpu()
    name_to_value["weight"] = weight

    if hasattr(module, "bias") and module.bias is not None:
        name_to_value["bias"] = module.bias.to(dtype).cpu()

    qformat: str = get_quantization_format(module)
    block_size = get_weight_block_size(module)

    # Getting the weight scales
    weight_scale = get_weight_scaling_factor(module)
    weight_scale_2 = get_weight_scaling_factor_2(module)
    if weight_scale is not None:
        name_to_value["weight_scale"] = weight_scale

    if weight_scale_2 is not None:
        name_to_value["weight_scale_2"] = weight_scale_2

    # Getting the input scale
    input_scale = get_activation_scaling_factor(module)
    if input_scale is not None:
        name_to_value["input_scale"] = input_scale
        # TODO (chenhany): support AWQ with pre_quant_scale
        if hasattr(module.input_quantizer, "_pre_quant_scale"):
            raise ValueError("Detect pre_quant_scale! SmoothQuant/AWQ are not yet supported!")

    if hasattr(module, "output_quantizer"):
        output_scale = get_kv_cache_scaling_factor([module])
        if output_scale is not None:
            name_to_value["output_scale"] = output_scale

    return name_to_value, qformat, block_size


class GPTModelExporter:
    """Megatron Core GPTModel Exporter.

    The Exporter is created by `export_mcore_gpt_to_hf` to host attributes
    and methods that export a quantized Megatron Core GPTModel to the Hugging
    Face unified checkpoint.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to export the unquantized layers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        dtype=torch.bfloat16,
    ):
        """Create a GPTModel exporter instance."""
        if not isinstance(model, GPTModel):
            raise ValueError("Input to GPTModelExport must be a megatron.core.models.GPTModel!")

        if pretrained_model_name_or_path is None:
            self.arch = "GPTModel"
        else:
            self._hf_config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
            # Update hf_config
            self._hf_config.num_hidden_layers = model.config.num_layers
            self._hf_config.hidden_size = model.config.hidden_size
            self._hf_config.head_dim = model.config.kv_channels
            self._hf_config.num_attention_heads = model.config.num_attention_heads
            self._hf_config.num_key_value_heads = model.config.num_query_groups
            self._hf_config.intermediate_size = model.config.ffn_hidden_size
            self.arch = self._hf_config.architectures[0]

        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]
        self.model = model
        self.dtype = dtype
        self._state_dict = OrderedDict()
        self._hf_pretrained_model_name = pretrained_model_name_or_path

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """Save a unified checkpoint which can be deploied by vLLM and TensorRT-LLM.

        Args:
            save_directory: Directory to which to save. Will be created if it doesn't exist.
        """
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()

        meta_filename = "model-{:05d}-of-{:05d}.json".format(pp_rank + 1, pp_size)
        ckpt_filename = "model-{:05d}-of-{:05d}.safetensors".format(
            pp_rank + 1,
            pp_size,
        )

        # Main export process
        state_dict = self.state_dict
        quantization = None
        kv_cache_quantization = None
        total_size = 0
        weight_map = {}

        for key, val in state_dict.items():
            total_size += val.numel() * val.element_size()
            weight_map[key] = ckpt_filename
            if "weight_scale" in key:
                quantization = "FP8"
            if "k_scale" in key or "output_scale" in key:
                kv_cache_quantization = "FP8"

        if torch.distributed.get_rank() == 0:
            if self._hf_pretrained_model_name:
                self._hf_config.save_pretrained(save_directory)
                try:
                    generation_config = transformers.GenerationConfig.from_pretrained(
                        self._hf_pretrained_model_name
                    )
                    generation_config.save_pretrained(save_directory)
                except OSError:
                    pass
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        self._hf_pretrained_model_name
                    )
                    tokenizer.save_pretrained(save_directory)
                except OSError:
                    pass
            else:
                os.makedirs(save_directory, exist_ok=True)

        # Barrier to ensure the export_dir has been created.
        torch.distributed.barrier()

        # Write multi-file metadata
        with open(save_directory + "/" + meta_filename, "w") as f:
            json.dump(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=4
            )

        save_file(self.state_dict, save_directory + "/" + ckpt_filename, metadata={"format": "pt"})

        # Barrier to ensure all ranks have written the metadata
        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            hf_quant_config = {
                "producer": {
                    "name": "modelopt",
                    "version": __version__,
                },
                "quantization": {
                    "quant_algo": quantization,
                    "kv_cache_quant_algo": kv_cache_quantization,
                },
            }
            with open(save_directory + "/hf_quant_config.json", "w") as f:
                json.dump(hf_quant_config, f, indent=4)
            safetensor_index = {
                "metadata": {"total_size": 0},
                "weight_map": {},
            }
            for pp_rank in range(get_pipeline_model_parallel_world_size()):
                meta_filename = "model-{:05d}-of-{:05d}.json".format(
                    pp_rank + 1,
                    get_pipeline_model_parallel_world_size(),
                )
                with open(save_directory + "/" + meta_filename) as f:
                    shard = json.load(f)
                safetensor_index["metadata"]["total_size"] += shard["metadata"]["total_size"]
                safetensor_index["weight_map"].update(shard["weight_map"])

            with open(save_directory + "/model.safetensors.index.json", "w") as f:
                json.dump(safetensor_index, f, indent=4)

    @property
    def state_dict(self):
        """Return the real quantized state_dict."""
        if len(self._state_dict) == 0:
            self._get_state_dict()
        return self._state_dict

    def _populate_rule_book(self):
        all_rules = {}

        def _custom_mapping_to_lambda(mapping):
            method_map = {
                "name_remapping": self._name_remapping,
                "qkv_slicing": self._qkv_slicing,
                "gated_mlp_slicing": self._gated_mlp_slicing,
            }
            func = method_map[mapping.func_name]
            prefix = mapping.target_name_or_prefix
            func_kwargs = mapping.func_kwargs
            return lambda m, *args: func(m, prefix.format(*args), **func_kwargs)

        for arch, mappings in all_mcore_hf_export_mapping.items():
            all_rules[arch] = {k: _custom_mapping_to_lambda(v) for (k, v) in mappings.items()}

        return all_rules

    def _get_weight_scales(self, quantized_state: Dict[str, Any], qformat: str):
        weight_scale = quantized_state.get("weight_scale", None)
        weight_scale_2 = quantized_state.get("weight_scale_2", None)

        if weight_scale is not None:
            weight_scale = weight_scale.clone().detach()
            if qformat == QUANTIZATION_FP8 and weight_scale.numel() == 1:
                weight_scale = weight_scale.squeeze()
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.clone().detach()

        return weight_scale, weight_scale_2

    def _name_remapping(self, module, prefix, skip_output_scale=True):
        name_to_value, qformat, block_size = get_quantized_state(module, self.dtype)

        weight = name_to_value.pop("weight")
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        if weight_scale is None:
            self._state_dict[prefix + "weight"] = weight
        else:
            self._state_dict[prefix + "weight"] = to_quantized_weight(
                weight,
                weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )

        for key, val in name_to_value.items():
            if "output_scale" == key and skip_output_scale:
                continue
            else:
                self._state_dict[prefix + key] = val

    def _gated_mlp_slicing(
        self, module, prefix, gate_proj_name="gate_proj", up_proj_name="up_proj"
    ):
        name_to_value, qformat, block_size = get_quantized_state(module, self.dtype)

        weight = name_to_value.pop("weight")
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        gate_proj_prefix = prefix + gate_proj_name + "."
        up_proj_prefix = prefix + up_proj_name + "."

        ffn_hidden_size = module.config.ffn_hidden_size
        gate_proj_weight = weight[:ffn_hidden_size, :]
        up_proj_weight = weight[ffn_hidden_size:, :]

        if weight_scale is None:
            self._state_dict[gate_proj_prefix + "weight"] = gate_proj_weight
            self._state_dict[up_proj_prefix + "weight"] = up_proj_weight
        else:
            if len(weight_scale.shape) == 0:
                gate_proj_weight_scale = weight_scale.detach().clone()
                up_proj_weight_scale = weight_scale.detach().clone()
            else:
                gate_proj_weight_scale = weight_scale[:ffn_hidden_size]
                up_proj_weight_scale = weight_scale[ffn_hidden_size:]
            self._state_dict[gate_proj_prefix + "weight"] = to_quantized_weight(
                gate_proj_weight,
                gate_proj_weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )
            self._state_dict[up_proj_prefix + "weight"] = to_quantized_weight(
                up_proj_weight,
                up_proj_weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )
            self._state_dict[gate_proj_prefix + "weight_scale"] = gate_proj_weight_scale
            self._state_dict[up_proj_prefix + "weight_scale"] = up_proj_weight_scale

        # weight and weight_scale have been pop out.
        for key, val in name_to_value.items():
            gate_proj_key = gate_proj_prefix + key
            up_proj_key = up_proj_prefix + key
            if "output_scale" == key:
                continue
            else:
                self._state_dict[gate_proj_key] = val.detach().clone()
                self._state_dict[up_proj_key] = val.detach().clone()

    def _qkv_slicing(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
        k_scale_name="k_scale",
        v_scale_name="v_scale",
    ):
        name_to_value, qformat, block_size = get_quantized_state(module, self.dtype)

        q_proj_prefix = prefix + q_proj_name + "."
        k_proj_prefix = prefix + k_proj_name + "."
        v_proj_prefix = prefix + v_proj_name + "."

        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        weight = name_to_value.pop("weight").reshape([qkv_total_dim, head_size, hidden_size])
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
        ## Example of slices
        ## 7b: num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## 70b (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]
        slices = [q_slice, k_slice, v_slice]
        prefixes = [q_proj_prefix, k_proj_prefix, v_proj_prefix]

        proj_weights = [weight[s].reshape(-1, hidden_size) for s in slices]
        proj_keys = [p + "weight" for p in prefixes]

        if weight_scale is None:
            for key, weight in zip(proj_keys, proj_weights):
                self._state_dict[key] = weight
        else:
            if len(weight_scale.shape) > 0:
                # AWQ per-block or per-channel scaling
                proj_weight_scales = [weight_scale[s] for s in slices]
            else:
                # per-tensor scaling
                proj_weight_scales = [weight_scale.detach().clone()] * 3

            for weight, scale, key in zip(proj_weights, proj_weight_scales, proj_keys):
                quantized_weight = to_quantized_weight(
                    weight,
                    scale,
                    qformat,
                    weight_scale_2,
                    block_size,
                )
                self._state_dict[key] = quantized_weight
                self._state_dict[key + "_scale"] = scale

        # weight and weight_scale have been pop out.
        for key, val in name_to_value.items():
            q_proj_key = q_proj_prefix + key
            k_proj_key = k_proj_prefix + key
            v_proj_key = v_proj_prefix + key
            if "output_scale" == key:
                self._state_dict[prefix + k_scale_name] = val.detach().clone()
                self._state_dict[prefix + v_scale_name] = val.detach().clone()
            else:
                self._state_dict[q_proj_key] = val.detach().clone()
                self._state_dict[k_proj_key] = val.detach().clone()
                self._state_dict[v_proj_key] = val.detach().clone()

    def _get_state_dict(self):
        model = self.model

        # Embedding
        if hasattr(model, "embedding"):
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        # Output layer
        if hasattr(model, "output_layer"):
            self.rules["output_layer"](model.output_layer)

        # Decoder layers
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1
            self.rules["input_layernorm"](layer.input_layernorm, layer_id)
            self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
            self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
            self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)
            self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
            self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)

        if hasattr(model, "medusa_heads"):
            for head_id, head in enumerate(model.medusa_heads):
                self.rules["medusa_heads.lm_head"](head.lm_head, head_id)
                for layer_id, layer in enumerate(head.medusa_layers):
                    self.rules["medusa_heads.medusa_layers.linear"](layer.linear, head_id, layer_id)

        if hasattr(model, "eagle_module"):
            self.rules["eagle_module.fc"](model.eagle_module.fc)
            for layer in model.eagle_module.decoder.layers:
                layer_id = layer.layer_number - 1
                self.rules["input_layernorm"](layer.input_layernorm, layer_id)
                self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
                self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
                self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)
                self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)


def export_mcore_gpt_to_hf(
    model: torch.nn.Module,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
    dtype: torch.dtype = torch.float16,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
):
    """Export Megatron Core GPTModel to unified checkpoint and save to export_dir.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to export the unquantized layers.
        export_dir: The target export path.
    """
    exporter = GPTModelExporter(model, pretrained_model_name_or_path, dtype)
    exporter.save_pretrained(export_dir)


class GPTModelImporter:
    """Megatron Core GPTModel HuggingFace Importer.

    The Importer is created by `import_mcore_gpt_from_hf` to host attributes
    and methods that import a Megatron Core GPTModel from a supported Hugging
    Face model.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to export the unquantized layers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str,
        dtype=torch.bfloat16,
    ):
        """Create a GPTModel importer instance."""
        self._hf_config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
        pretrained_model_path = Path(pretrained_model_name_or_path)
        if not pretrained_model_path.is_dir():
            pretrained_model_path = tempfile.gettempdir() + "/" + pretrained_model_name_or_path
            snapshot_download(
                repo_id=pretrained_model_name_or_path,
                local_dir=pretrained_model_path,
            )
        self.arch = self._hf_config.architectures[0]
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.dtype = dtype

    def _populate_rule_book(self):
        """The rule book maps each state_dict key to a Callable."""
        all_rules = {}

        def _custom_mapping_to_lambda(mapping):
            method_map = {
                "name_remapping": self._name_remapping,
                "qkv_merging": self._qkv_merging,
                "gated_mlp_merging": self._gated_mlp_merging,
            }
            func = method_map[mapping.func_name]
            prefix = mapping.target_name_or_prefix
            func_kwargs = mapping.func_kwargs
            return lambda m, *args: func(m, prefix.format(*args), **func_kwargs)

        for arch, mappings in all_mcore_hf_import_mapping.items():
            all_rules[arch] = {k: _custom_mapping_to_lambda(v) for (k, v) in mappings.items()}

        return all_rules

    def _get_safetensor(self, key):
        """Get a safetensor from the sharded checkpoint."""
        safetensors_file = Path(self.pretrained_model_path + "/model.safetensors")
        safetensors_index_file = Path(self.pretrained_model_path + "/model.safetensors.index.json")
        if safetensors_file.is_file():
            pass
        elif safetensors_index_file.is_file():
            with open(safetensors_index_file) as f:
                safetensors_index = json.load(f)
            safetensors_file = Path(
                self.pretrained_model_path + "/" + safetensors_index["weight_map"][key]
            )
        else:
            raise ValueError("Only safetensors (single of multi- files) are supported.")

        with safe_open(safetensors_file, framework="pt") as f:
            tensor = f.get_tensor(key)
        return tensor

    def _name_remapping(self, module, prefix):
        state_dict = {}
        for key, val in module.state_dict().items():
            if "extra_state" in key:
                state_dict[key] = val
            else:
                tensor = self._get_safetensor(prefix + key)
                state_dict[key] = tensor.to(dtype=self.dtype).to(device=val.device)
        module.load_state_dict(state_dict)

    def _gated_mlp_merging(
        self, module, prefix, gate_proj_name="gate_proj", up_proj_name="up_proj"
    ):
        gate_proj = self._get_safetensor(prefix + gate_proj_name + ".weight")
        up_proj = self._get_safetensor(prefix + up_proj_name + ".weight")
        tensor = torch.cat((gate_proj, up_proj), dim=0)
        state_dict = {"weight": tensor.to(dtype=self.dtype).to(device=module.weight.device)}
        module.load_state_dict(state_dict)

    def _qkv_merging(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
    ):
        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups
        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        q_proj = self._get_safetensor(prefix + q_proj_name + ".weight").reshape(
            -1, head_size, hidden_size
        )
        k_proj = self._get_safetensor(prefix + k_proj_name + ".weight").reshape(
            -1, head_size, hidden_size
        )
        v_proj = self._get_safetensor(prefix + v_proj_name + ".weight").reshape(
            -1, head_size, hidden_size
        )
        tensor = module.weight.detach().clone().reshape([qkv_total_dim, head_size, hidden_size])

        tensor[q_slice] = q_proj.to(dtype=tensor.dtype).to(device=tensor.device)
        tensor[k_slice] = k_proj.to(dtype=tensor.dtype).to(device=tensor.device)
        tensor[v_slice] = v_proj.to(dtype=tensor.dtype).to(device=tensor.device)

        module.load_state_dict({"weight": tensor.reshape(-1, hidden_size)})

    def _import_state_dict(self):
        model = self.model

        # Embedding
        if hasattr(model, "embedding"):
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        # Output layer
        if hasattr(model, "output_layer"):
            self.rules["output_layer"](model.output_layer)

        # Decoder layers
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1
            self.rules["input_layernorm"](layer.input_layernorm, layer_id)
            self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
            self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
            self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)
            self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
            self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)


def import_mcore_gpt_from_hf(
    model: torch.nn.Module,
    pretrained_model_path: str,
    dtype: torch.dtype = torch.float16,
):
    """Import GPTModel state_dict from supported HuggingFace pretrained model path.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_path: A path to a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to import.
    """
    importer = GPTModelImporter(model, pretrained_model_path, dtype)
    importer._import_state_dict()
