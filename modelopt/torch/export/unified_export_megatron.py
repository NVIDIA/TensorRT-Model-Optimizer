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
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

from modelopt import __version__
from modelopt.torch.utils import import_plugin

from .model_config import (
    KV_CACHE_FP8,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_NVFP4,
)
from .plugins.mcore_common import all_mcore_hf_export_mapping, all_mcore_hf_import_mapping
from .plugins.mcore_custom import save_safetensors
from .quant_utils import (
    get_activation_scaling_factor,
    get_kv_cache_dtype,
    get_quantization_format,
    get_scaling_factor,
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
    from megatron.core.models.mamba import MambaModel
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_layer import TransformerLayer

    has_mcore = True

__all__ = ["export_mcore_gpt_to_hf", "import_mcore_gpt_from_hf"]


# This path uses output_quantizer for KV cache quantization.
# The function below is the old version of get_kv_cache_scaling_factor which is now refactored to handle bmm_quantizer.
def get_kv_cache_scaling_factor(kv_module: nn.Module) -> torch.Tensor:
    """Returns the kv_cache scaling factor if output quantizer is set. Else returns None by default."""
    scaling_factor = (
        get_scaling_factor(kv_module.output_quantizer)
        if hasattr(kv_module, "output_quantizer")
        else None
    )

    if not scaling_factor:
        return None

    # For FP8, we recommend default kv cache scaling factor to be 1.
    if get_kv_cache_dtype(kv_module) == KV_CACHE_FP8:
        if scaling_factor.item() > 0.5:
            warn(
                f"!!!!Large KV activations detected: {scaling_factor.item()}, "
                "Quantized KV cache may lead to higher accuracy drop.\n!!!!"
            )
        scaling_factor = torch.max(
            scaling_factor,
            torch.tensor([1.0], dtype=torch.float, device=scaling_factor.device),
        )
    return scaling_factor


def get_quantized_state(
    module: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, torch.Tensor], str, int]:
    """Return a state_dict, quantization format, and block_size of the module.

    Args:
        module: The target module to perform real quantization.
        dtype: The default data type.

    Returns:
        Tuple: state_dict, quantization format, and block_size of the module.
    """
    name_to_value = {}
    qformat: str = get_quantization_format(module)
    block_size = get_weight_block_size(module)

    if hasattr(module, "weight") and module.weight is not None:
        weight = module.weight.to(dtype).cpu()
        name_to_value["weight"] = weight
    else:
        return name_to_value, qformat, block_size

    if hasattr(module, "bias") and module.bias is not None:
        name_to_value["bias"] = module.bias.to(dtype).cpu()

    if hasattr(module, "expert_bias") and module.expert_bias is not None:
        name_to_value["expert_bias"] = module.expert_bias.to(dtype).cpu()

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
        output_scale = get_kv_cache_scaling_factor(module)
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
        export_extra_modules: If True, export extra modules like medusa_heads,
            eagle_module, or mtp. Otherwise, only export the base model.
        dtype: The weights data type to export the unquantized layers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        export_extra_modules: bool = False,
        dtype=torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        """Create a GPTModel exporter instance."""
        if not isinstance(model, GPTModel) and not isinstance(model, MambaModel):
            raise ValueError("Input to GPTModelExport must be a megatron.core.models.GPTModel!")

        self._state_dict = OrderedDict()
        self._hf_pretrained_model_name = pretrained_model_name_or_path
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        # If multimodal, extra the text_config
        self._hf_text_config = getattr(self._hf_config, "text_config", self._hf_config)

        # Update hf_config
        self._hf_text_config.num_hidden_layers = model.config.num_layers
        self._hf_text_config.hidden_size = model.config.hidden_size
        self._hf_text_config.head_dim = model.config.kv_channels
        self._hf_text_config.num_attention_heads = model.config.num_attention_heads
        self._hf_text_config.num_key_value_heads = model.config.num_query_groups
        self._hf_text_config.intermediate_size = model.config.ffn_hidden_size
        self._hf_quant_config = None
        self._hf_extra_config = None
        self.export_extra_modules = export_extra_modules
        self.model = model
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.arch = self._hf_config.architectures[0]
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]

        if not hasattr(model, "_modelopt_state"):
            return

        for mode, mode_cfg in model._modelopt_state:
            if mode == "medusa" and export_extra_modules:
                medusa_config = {
                    "num_medusa_heads": mode_cfg["config"]["medusa_num_heads"],
                    "num_medusa_layers": mode_cfg["config"]["medusa_num_layers"],
                }
                self._hf_config.medusa = medusa_config
                self.rules = self.all_rules["MedusaLlamaForCausalLM"]

            if mode == "eagle" and export_extra_modules:
                is_eagle3 = mode_cfg["config"]["use_aux_hidden_state"]

                architectures = "LlamaForCausalLMEagle3" if is_eagle3 else "LlamaForCausalLMEagle"

                self.rules = self.all_rules[architectures]

                # By default, we use Llama-3.1
                self._hf_extra_config = transformers.AutoConfig.from_pretrained(
                    "nvidia/Llama-3.1-8B-Instruct-FP8", trust_remote_code=self.trust_remote_code
                )

                eagle_config = {
                    "use_input_layernorm_in_first_layer": mode_cfg["config"][
                        "use_input_layernorm_in_first_layer"
                    ],
                    "use_last_layernorm": mode_cfg["config"]["use_last_layernorm"],
                    "use_mtp_layernorm": mode_cfg["config"]["use_mtp_layernorm"],
                    "use_aux_hidden_state": mode_cfg["config"]["use_aux_hidden_state"],
                    "eagle_aux_hidden_state_layer_ids": model.eagle_aux_hidden_state_layer_ids,
                }

                eagle_config_update = {
                    "architectures": [architectures],
                    "head_dim": self._hf_text_config.head_dim,
                    "hidden_act": self._hf_text_config.hidden_act,
                    "hidden_size": self._hf_text_config.hidden_size,
                    "intermediate_size": self._hf_text_config.intermediate_size,
                    "max_position_embeddings": self._hf_text_config.max_position_embeddings,
                    "num_attention_heads": self._hf_text_config.num_attention_heads,
                    "num_key_value_heads": self._hf_text_config.num_key_value_heads,
                    "num_hidden_layers": mode_cfg["config"]["eagle_num_layers"],
                    "vocab_size": self._hf_text_config.vocab_size,
                    # Unset any special token ids given that the tokenizer can change here.
                    "bos_token_id": None,
                    "eos_token_id": None,
                    "pad_token_id": None,
                    "sep_token_id": None,
                    # The following attributes are EAGLE specific
                    "eagle_config": eagle_config,
                }

                # [TODO] (yeyu): there is also target_hidden_size
                if mode_cfg["config"]["draft_vocab_size"] > 0:
                    eagle_config_update["draft_vocab_size"] = mode_cfg["config"]["draft_vocab_size"]
                else:
                    eagle_config_update["draft_vocab_size"] = None

                self._hf_extra_config.update(eagle_config_update)

            if mode == "mtp" and export_extra_modules:
                mtp_config = {
                    "hidden_size": self._hf_config.hidden_size,
                    "head_dim": self._hf_config.head_dim,
                    "intermediate_size": self._hf_config.intermediate_size,
                    "max_position_embeddings": self._hf_config.max_position_embeddings,
                    "num_attention_heads": self._hf_config.num_attention_heads,
                    "num_hidden_layers": mode_cfg["config"]["mtp_num_layers"],
                    "num_mtp_module": mode_cfg["config"]["mtp_num_module"],
                    "num_key_value_heads": self._hf_config.num_key_value_heads,
                    "rms_norm_eps": self._hf_config.rms_norm_eps,
                    "rope_theta": self._hf_config.rope_theta,
                    "use_input_layernorm_in_first_layer": True,
                    "use_last_layernorm": False,
                }
                self._hf_config.mtp = mtp_config

    def save_pretrained(self, save_directory: str | os.PathLike):
        """Save a unified checkpoint which can be deploied by vLLM and TensorRT-LLM.

        Args:
            save_directory: Directory to which to save. Will be created if it doesn't exist.
        """
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()

        # Main export process
        state_dict = self.extra_state_dict if self.export_extra_modules else self.state_dict
        quantization_format = get_quantization_format(self.model)
        quantization = None
        kv_cache_quantization = None

        if quantization_format in (
            QUANTIZATION_FP8_PB_REAL,
            QUANTIZATION_FP8_PB_WO,
        ):
            quantization = quantization_format
        elif quantization_format == QUANTIZATION_FP8:
            quantization = "FP8"
        elif quantization_format == QUANTIZATION_NVFP4:
            quantization = "NVFP4"

        # TODO (chenhany): need to handle Medusa and EAGLE meatadata
        if torch.distributed.get_rank() == 0:
            if self.export_extra_modules and self._hf_extra_config is not None:
                # os.makedirs(save_directory, exist_ok=True)
                # with open(save_directory + "/config.json", 'w') as file:
                #    json.dump(self._hf_extra_config, file, indent=4)
                self._hf_extra_config.save_pretrained(save_directory)
            else:
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
                except TypeError:
                    pass

        if torch.distributed.get_rank() == 0:
            hf_quant_config = {
                "producer": {
                    "name": "modelopt",
                    "version": __version__,
                },
                "quantization": {
                    "quant_algo": quantization,
                    "kv_cache_quant_algo": kv_cache_quantization,
                    "exclude_modules": ["lm_head"],
                },
            }
            with open(save_directory + "/hf_quant_config.json", "w") as f:
                json.dump(hf_quant_config, f, indent=4)

        # Barrier to ensure the export_dir has been created.
        torch.distributed.barrier()

        if self.export_extra_modules:
            if pp_rank == pp_size - 1:
                save_file(
                    state_dict, save_directory + "/model.safetensors", metadata={"format": "pt"}
                )
            return

        save_safetensors(state_dict, save_directory)

    @property
    def state_dict(self):
        """Return the real quantized state_dict of the base model."""
        if len(self._state_dict) == 0:
            self._get_state_dict()
        return self._state_dict

    @property
    def extra_state_dict(self):
        if len(self._state_dict) == 0:
            self._get_medusa_heads_state_dict()
            self._get_eagle_module_state_dict()
            self._get_mtp_state_dict()
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

    def _get_weight_scales(self, quantized_state: dict[str, Any], qformat: str):
        weight_scale = quantized_state.get("weight_scale")
        weight_scale_2 = quantized_state.get("weight_scale_2")

        if weight_scale is not None:
            weight_scale = weight_scale.clone().detach()
            if qformat == QUANTIZATION_FP8 and weight_scale.numel() == 1:
                weight_scale = weight_scale.squeeze()
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.clone().detach()

        return weight_scale, weight_scale_2

    def _name_remapping(
        self,
        module: torch.nn.Module | torch.Tensor,
        prefix: str,
        skip_output_scale: bool = True,
        mapping={},
    ):
        if isinstance(module, torch.Tensor):
            self._state_dict[prefix] = module
            return

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
            if key == "output_scale" and skip_output_scale:
                continue
            else:
                source_key = mapping.get(key, key)
                self._state_dict[prefix + source_key] = val

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
            if key == "output_scale":
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
        head_size = config.kv_channels
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        weight = name_to_value.pop("weight")

        if weight.shape[-1] == 2 * hidden_size:
            print(
                "Parameter linear_qkv.weight has 2x the hidden_size."
                "Set hidden_size to 2x the hidden_size. EAGLE3 is the only known"
                "use case which has this behavior."
            )
            hidden_size = 2 * hidden_size

        weight = weight.reshape([qkv_total_dim, head_size, hidden_size])
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
                weight_scale_dtype = weight_scale.dtype
                weight_scale_hidden_size = weight_scale.shape[-1]
                weight_scale = weight_scale.to(dtype=float).reshape(
                    [qkv_total_dim, head_size, weight_scale_hidden_size]
                )
                proj_weight_scales = [
                    weight_scale[s]
                    .reshape(-1, weight_scale_hidden_size)
                    .to(dtype=weight_scale_dtype)
                    for s in slices
                ]
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
            if key == "output_scale":
                self._state_dict[prefix + k_scale_name] = val.detach().clone()
                self._state_dict[prefix + v_scale_name] = val.detach().clone()
            else:
                self._state_dict[q_proj_key] = val.detach().clone()
                self._state_dict[k_proj_key] = val.detach().clone()
                self._state_dict[v_proj_key] = val.detach().clone()

    def _get_medusa_heads_state_dict(self):
        medusa_heads = getattr(self.model, "medusa_heads", None)
        if medusa_heads is None:
            return

        for head_id, head in enumerate(medusa_heads):
            self.rules["medusa_heads.lm_head"](head.lm_head, head_id)
            for layer_id, layer in enumerate(head.medusa_layers):
                self.rules["medusa_heads.medusa_layers.linear"](layer.linear, head_id, layer_id)

    def _get_eagle_module_state_dict(self):
        eagle_module = getattr(self.model, "eagle_module", None)

        if eagle_module is None:
            return

        # if hasattr(self.model, "embedding"):
        #    self.rules["word_embeddings"](self.model.embedding.word_embeddings)

        self.rules["fc"](eagle_module.fc)
        if self.model.use_aux_hidden_state:
            self.rules["enorm"](eagle_module.enorm)
        elif self.model.use_mtp_layernorm:
            self.rules["enorm"](eagle_module.enorm)
            self.rules["hnorm"](eagle_module.hnorm)

        if self.model.use_last_layernorm:
            self.rules["final_layernorm"](eagle_module.decoder.final_layernorm)

        if self.model.draft_vocab_size > 0:
            self.rules["output_layer"](eagle_module.eagle_output_layer)
            self.rules["d2t"](eagle_module.d2t)

        for layer in eagle_module.decoder.layers:
            layer_id = layer.layer_number - 1

            if layer_id > 0 or self.model.use_input_layernorm_in_first_layer:
                self.rules["input_layernorm"](layer.input_layernorm, layer_id)

            if "MLASelfAttention" in str(type(layer.self_attention)):
                if hasattr(layer.self_attention, "linear_q_proj"):
                    self.rules["eagle_module.linear_q_proj"](
                        layer.self_attention.linear_q_proj, layer_id
                    )
                else:
                    self.rules["eagle_module.linear_q_down_proj"](
                        layer.self_attention.linear_q_down_proj, layer_id
                    )
                    self.rules["eagle_module.linear_q_layernorm"](
                        layer.self_attention.q_layernorm, layer_id
                    )
                    self.rules["eagle_module.linear_q_up_proj"](
                        layer.self_attention.linear_q_up_proj, layer_id
                    )

                self.rules["eagle_module.linear_kv_down_proj"](
                    layer.self_attention.linear_kv_down_proj, layer_id
                )
                self.rules["eagle_module.linear_kv_layernorm"](
                    layer.self_attention.kv_layernorm, layer_id
                )
                self.rules["eagle_module.linear_kv_up_proj"](
                    layer.self_attention.linear_kv_up_proj, layer_id
                )
            else:
                self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)

            self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
            self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

            if "MoE" in str(type(layer.mlp)):
                self.rules["eagle_module.router"](layer.mlp.router, layer_id)
                if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
                    self.rules["eagle_module.shared_experts.linear_fc1"](
                        layer.mlp.shared_experts.linear_fc1, layer_id
                    )
                    self.rules["eagle_module.shared_experts.linear_fc2"](
                        layer.mlp.shared_experts.linear_fc2, layer_id
                    )
                for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                    self.rules["eagle_module.local_experts.linear_fc1"](
                        expert.linear_fc1, layer_id, expert_id
                    )
                    self.rules["eagle_module.local_experts.linear_fc2"](
                        expert.linear_fc2, layer_id, expert_id
                    )
            else:
                self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)

    def _get_mtp_state_dict(self):
        mtp = getattr(self.model, "mtp", None)
        if mtp is None:
            return

        for module_id, module in enumerate(mtp):
            self.rules["mtp.fc"](module.fc, module_id)
            self.rules["mtp.enorm"](module.enorm, module_id)
            self.rules["mtp.hnorm"](module.hnorm, module_id)
            for layer in module.decoder.layers:
                layer_id = layer.layer_number - 1
                self.rules["mtp.input_layernorm"](layer.input_layernorm, module_id, layer_id)

                if "MLASelfAttention" in str(type(layer.self_attention)):
                    if hasattr(layer.self_attention, "linear_q_proj"):
                        self.rules["mtp.linear_q_proj"](
                            layer.self_attention.linear_q_proj, module_id, layer_id
                        )
                    else:
                        self.rules["mtp.linear_q_down_proj"](
                            layer.self_attention.linear_q_down_proj, module_id, layer_id
                        )
                        self.rules["mtp.linear_q_layernorm"](
                            layer.self_attention.q_layernorm, module_id, layer_id
                        )
                        self.rules["mtp.linear_q_up_proj"](
                            layer.self_attention.linear_q_up_proj, module_id, layer_id
                        )

                    self.rules["mtp.linear_kv_down_proj"](
                        layer.self_attention.linear_kv_down_proj, module_id, layer_id
                    )
                    self.rules["mtp.linear_kv_layernorm"](
                        layer.self_attention.kv_layernorm, module_id, layer_id
                    )
                    self.rules["mtp.linear_kv_up_proj"](
                        layer.self_attention.linear_kv_up_proj, module_id, layer_id
                    )
                else:
                    self.rules["mtp.linear_qkv"](
                        layer.self_attention.linear_qkv, module_id, layer_id
                    )

                self.rules["mtp.linear_proj"](layer.self_attention.linear_proj, module_id, layer_id)
                self.rules["mtp.pre_mlp_layernorm"](layer.pre_mlp_layernorm, module_id, layer_id)

                if "MoE" in str(type(layer.mlp)):
                    self.rules["mtp.router"](layer.mlp.router, module_id, layer_id)
                    if hasattr(layer.mlp, "shared_experts"):
                        self.rules["mtp.shared_experts.linear_fc1"](
                            layer.mlp.shared_experts.linear_fc1, module_id, layer_id
                        )
                        self.rules["mtp.shared_experts.linear_fc2"](
                            layer.mlp.shared_experts.linear_fc2, module_id, layer_id
                        )
                    for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                        self.rules["mtp.local_experts.linear_fc1"](
                            expert.linear_fc1, module_id, layer_id, expert_id
                        )
                        self.rules["mtp.local_experts.linear_fc2"](
                            expert.linear_fc2, module_id, layer_id, expert_id
                        )
                else:
                    self.rules["mtp.linear_fc1"](layer.mlp.linear_fc1, module_id, layer_id)
                    self.rules["mtp.linear_fc2"](layer.mlp.linear_fc2, module_id, layer_id)

    def _get_state_dict(self):
        model = self.model

        # Embedding
        if hasattr(model, "embedding"):
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        if hasattr(model.decoder, "final_norm") and model.decoder.final_norm:
            self.rules["final_norm"](model.decoder.final_norm)

        # Output layer
        if hasattr(model, "output_layer") and not model.share_embeddings_and_output_weights:
            self.rules["output_layer"](model.output_layer)

        # Decoder layers
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1

            if isinstance(layer, MambaLayer):
                if not isinstance(layer.norm, IdentityOp):
                    self.rules["norm"](layer.norm, layer_id)

                self.rules["mixer_norm"](layer.mixer.norm, layer_id)
                self.rules["A_log"](layer.mixer.A_log, layer_id)
                self.rules["D"](layer.mixer.D, layer_id)
                self.rules["dt_bias"](layer.mixer.dt_bias, layer_id)

                self.rules["conv1d"](layer.mixer.conv1d, layer_id)
                self.rules["in_proj"](layer.mixer.in_proj, layer_id)
                self.rules["out_proj"](layer.mixer.out_proj, layer_id)

            elif isinstance(layer, TransformerLayer):
                if not isinstance(layer.input_layernorm, IdentityOp):
                    self.rules["input_layernorm"](layer.input_layernorm, layer_id)

                if not isinstance(layer.self_attention, IdentityOp):
                    if "MLASelfAttention" in str(type(layer.self_attention)):
                        if hasattr(layer.self_attention, "linear_q_proj"):
                            self.rules["linear_q_proj"](
                                layer.self_attention.linear_q_proj, layer_id
                            )
                        else:
                            self.rules["linear_q_down_proj"](
                                layer.self_attention.linear_q_down_proj, layer_id
                            )
                            self.rules["linear_q_layernorm"](
                                layer.self_attention.q_layernorm, layer_id
                            )
                            self.rules["linear_q_up_proj"](
                                layer.self_attention.linear_q_up_proj, layer_id
                            )

                        self.rules["linear_kv_down_proj"](
                            layer.self_attention.linear_kv_down_proj, layer_id
                        )
                        self.rules["linear_kv_layernorm"](
                            layer.self_attention.kv_layernorm, layer_id
                        )
                        self.rules["linear_kv_up_proj"](
                            layer.self_attention.linear_kv_up_proj, layer_id
                        )
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
                    else:
                        if layer.self_attention.q_layernorm is not None and not isinstance(
                            layer.self_attention.q_layernorm, IdentityOp
                        ):
                            self.rules["q_layernorm"](layer.self_attention.q_layernorm, layer_id)
                            self.rules["k_layernorm"](layer.self_attention.k_layernorm, layer_id)
                        self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)

                if not isinstance(layer.pre_mlp_layernorm, IdentityOp):
                    self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

                if not isinstance(layer.mlp, IdentityOp):
                    if "MoE" in str(type(layer.mlp)):
                        self.rules["router"](layer.mlp.router, layer_id)
                        if (
                            hasattr(layer.mlp, "shared_experts")
                            and layer.mlp.shared_experts is not None
                        ):
                            self.rules["shared_experts.linear_fc1"](
                                layer.mlp.shared_experts.linear_fc1, layer_id
                            )
                            self.rules["shared_experts.linear_fc2"](
                                layer.mlp.shared_experts.linear_fc2, layer_id
                            )
                        for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                            self.rules["local_experts.linear_fc1"](
                                expert.linear_fc1, layer_id, expert_id
                            )
                            self.rules["local_experts.linear_fc2"](
                                expert.linear_fc2, layer_id, expert_id
                            )
                    else:
                        self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                        self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)
            else:
                raise ValueError("Only TransformerLayer or MambaLayer are supported.")


def export_mcore_gpt_to_hf(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike | None = None,
    export_extra_modules: bool = False,
    dtype: torch.dtype = torch.float16,
    export_dir: Path | str = tempfile.gettempdir(),
):
    """Export Megatron Core GPTModel to unified checkpoint and save to export_dir.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        export_extra_modules: If True, export extra modules like medusa_heads,
            eagle_module, or mtp. Otherwise, only export the base model.
        dtype: The weights data type to export the unquantized layers.
        export_dir: The target export path.
    """
    exporter = GPTModelExporter(
        model, pretrained_model_name_or_path, export_extra_modules=export_extra_modules, dtype=dtype
    )
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

    weight_scale_name: str = "weight_scale_inv"

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str,
        workspace_dir: str | None = None,
        dtype=torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        """Create a GPTModel importer instance."""
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        pretrained_model_path = Path(pretrained_model_name_or_path)
        if not pretrained_model_path.is_dir():
            if workspace_dir is None:
                workspace_dir = tempfile.gettempdir()
            pretrained_model_path = workspace_dir + "/" + pretrained_model_name_or_path
            if torch.distributed.get_rank() == 0:
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    local_dir=pretrained_model_path,
                )
            torch.distributed.barrier()
        self.arch = self._hf_config.architectures[0]
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.dtype = dtype
        self.disable_tqdm = torch.distributed.get_rank() > 0

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

    def _get_safetensor(self, key, sharding_dim: int | None = None):
        """Get a safetensor from the sharded checkpoint."""
        safetensors_file = Path(self.pretrained_model_path) / "model.safetensors"
        safetensors_index_file = Path(self.pretrained_model_path) / "model.safetensors.index.json"
        if safetensors_file.is_file():
            pass
        elif safetensors_index_file.is_file():
            with open(safetensors_index_file) as f:
                safetensors_index = json.load(f)
            safetensors_file = (
                Path(self.pretrained_model_path) / safetensors_index["weight_map"][key]
            )
        else:
            raise ValueError("Only safetensors (single of multi- files) are supported.")

        with safe_open(safetensors_file, framework="pt") as f:
            if sharding_dim is None:
                tensor = f.get_tensor(key)
            else:
                tensor_slice = f.get_slice(key)
                assert tensor_slice is not None
                shape = tensor_slice.get_shape()
                # MCore tensor parallel model sharding
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                per_rank_size = shape[sharding_dim] // tp_size
                rank_offset = tp_rank * per_rank_size
                assert len(shape) == 2
                assert shape[sharding_dim] % tp_size == 0
                if sharding_dim in (1, -1):
                    tensor = tensor_slice[:, rank_offset : rank_offset + per_rank_size]
                else:
                    tensor = tensor_slice[rank_offset : rank_offset + per_rank_size, :]
        return tensor

    def _get_tensor_parallel_shard(self, tensor: torch.Tensor, dim: int):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return tensor
        return torch.chunk(tensor, tp_size, dim=dim)[tp_rank]

    def _name_remapping(
        self,
        module,
        prefix,
        mapping={},
        sharding_dim: int | None = None,
    ):
        if isinstance(module, torch.Tensor):
            module.data.copy_(self._get_safetensor(prefix))
            return

        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        state_dict = {}

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")
        else:
            tensor = self._get_safetensor(prefix + "weight", sharding_dim=sharding_dim)

        if weight_scale is not None:
            scale_name = prefix + self.weight_scale_name
            if weight_scale.ndim > 0:
                scale = self._get_safetensor(scale_name, sharding_dim=sharding_dim)
            else:
                scale = self._get_safetensor(scale_name)
            scale = scale.to(weight_scale.dtype).to(device=weight_scale.device)
            state_dict["weight_quantizer._scale"] = scale

            if tensor.shape != weight.shape:
                expanded_tensor = torch.zeros(weight.shape, dtype=tensor.dtype)
                expanded_tensor[: tensor.shape[0], : tensor.shape[1]] = tensor
                tensor = expanded_tensor
            state_dict["weight"] = tensor.view(dtype=weight.dtype).to(device=weight.device)
        else:
            state_dict["weight"] = tensor.to(dtype=self.dtype).to(device=weight.device)

        # Handle the rest of the state_dict.
        for key, val in module.state_dict().items():
            if key in {"weight", "weight_quantizer._scale"}:
                continue
            elif "extra_state" in key:
                state_dict[key] = val
            else:
                source_key = mapping.get(key, key)
                tensor = self._get_safetensor(prefix + source_key, sharding_dim=sharding_dim)
                state_dict[key] = tensor.to(dtype=self.dtype).to(device=val.device)

        module.load_state_dict(state_dict)

    def _gated_mlp_merging(
        self,
        module,
        prefix,
        gate_proj_name="gate_proj",
        up_proj_name="up_proj",
        sharding_dim: int | None = None,
    ):
        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        state_dict = {}

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")
        else:
            gate_proj = self._get_safetensor(
                prefix + gate_proj_name + ".weight", sharding_dim=sharding_dim
            )
            up_proj = self._get_safetensor(
                prefix + up_proj_name + ".weight", sharding_dim=sharding_dim
            )
            tensor = torch.cat((gate_proj, up_proj), dim=0)

        if weight_scale is not None:
            gate_scale_name = prefix + gate_proj_name + "." + self.weight_scale_name
            up_scale_name = prefix + up_proj_name + "." + self.weight_scale_name
            if weight_scale.ndim > 0:
                gate_scale = self._get_safetensor(gate_scale_name, sharding_dim=sharding_dim)
                up_scale = self._get_safetensor(up_scale_name, sharding_dim=sharding_dim)
                scale = torch.cat((gate_scale, up_scale), dim=0)
            else:
                scale = self._get_safetensor(gate_scale_name)
                # If source model is per tensor, compute a per tensor scale with max.
                if scale.ndim > 0:
                    scale = scale.max(dim=0).max(dim=0)
            state_dict["weight_quantizer._scale"] = scale.to(weight_scale.dtype).to(
                device=weight_scale.device
            )
            state_dict["weight"] = tensor.view(weight.dtype).to(device=weight.device)
        else:
            state_dict["weight"] = tensor.to(self.dtype).to(device=weight.device)

        module.load_state_dict(state_dict)

    def _qkv_merging(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
        sharding_dim: int | None = None,
    ):
        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = config.kv_channels

        if sharding_dim is not None:
            tp_size = get_tensor_model_parallel_world_size()
            assert head_num % tp_size == 0
            assert num_query_groups % tp_size == 0
            head_num = head_num // tp_size
            num_query_groups = num_query_groups // tp_size

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

        state_dict = {}

        weight = module.state_dict().get("weight", None)
        weight_scale = module.state_dict().get("weight_quantizer._scale", None)

        if weight is None:
            raise ValueError(f"{module!s} does not contain weight!")

        if weight_scale is not None:
            q_scale_name = prefix + q_proj_name + "." + self.weight_scale_name
            k_scale_name = prefix + k_proj_name + "." + self.weight_scale_name
            v_scale_name = prefix + v_proj_name + "." + self.weight_scale_name

            if weight_scale.ndim > 0:
                q_scale = self._get_safetensor(q_scale_name, sharding_dim=sharding_dim)
                k_scale = self._get_safetensor(k_scale_name, sharding_dim=sharding_dim)
                v_scale = self._get_safetensor(v_scale_name, sharding_dim=sharding_dim)
                weight_scale[q_slice] = q_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
                weight_scale[k_slice] = k_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
                weight_scale[v_slice] = v_scale.to(weight_scale.dtype).to(
                    device=weight_scale.device
                )
            else:
                q_scale = self._get_safetensor(q_scale_name)
                weight_scale = q_scale.to(weight_scale.dtype).to(device=weight_scale.device)
            state_dict["weight_quantizer._scale"] = weight_scale

        q_proj = self._get_safetensor(prefix + q_proj_name + ".weight", sharding_dim=sharding_dim)
        k_proj = self._get_safetensor(prefix + k_proj_name + ".weight", sharding_dim=sharding_dim)
        v_proj = self._get_safetensor(prefix + v_proj_name + ".weight", sharding_dim=sharding_dim)
        q_proj = q_proj.reshape(-1, head_size, hidden_size)
        k_proj = k_proj.reshape(-1, head_size, hidden_size)
        v_proj = v_proj.reshape(-1, head_size, hidden_size)
        tensor = weight.detach().clone().reshape([qkv_total_dim, head_size, hidden_size])

        if weight_scale is not None:
            tensor[q_slice] = q_proj.view(dtype=tensor.dtype).to(device=tensor.device)
            tensor[k_slice] = k_proj.view(dtype=tensor.dtype).to(device=tensor.device)
            tensor[v_slice] = v_proj.view(dtype=tensor.dtype).to(device=tensor.device)
        else:
            tensor[q_slice] = q_proj.to(dtype=tensor.dtype).to(device=tensor.device)
            tensor[k_slice] = k_proj.to(dtype=tensor.dtype).to(device=tensor.device)
            tensor[v_slice] = v_proj.to(dtype=tensor.dtype).to(device=tensor.device)

        state_dict["weight"] = tensor.reshape(-1, hidden_size)

        module.load_state_dict(state_dict)

    def _import_state_dict(self):
        model = self.model

        layer_pbar = tqdm(model.decoder.layers, disable=self.disable_tqdm)

        # Embedding
        if hasattr(model, "embedding"):
            layer_pbar.set_description("Importing word embedding")
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Decoder layers
        for layer in layer_pbar:
            layer_id = layer.layer_number - 1

            if isinstance(layer, MambaLayer):
                if not isinstance(layer.norm, IdentityOp):
                    self.rules["norm"](layer.norm, layer_id)

                self.rules["mixer_norm"](layer.mixer.norm, layer_id)
                self.rules["A_log"](layer.mixer.A_log, layer_id)
                self.rules["D"](layer.mixer.D, layer_id)
                self.rules["dt_bias"](layer.mixer.dt_bias, layer_id)

                self.rules["conv1d"](layer.mixer.conv1d, layer_id)
                self.rules["in_proj"](layer.mixer.in_proj, layer_id)
                self.rules["out_proj"](layer.mixer.out_proj, layer_id)

            elif isinstance(layer, TransformerLayer):
                if not isinstance(layer.input_layernorm, IdentityOp):
                    self.rules["input_layernorm"](layer.input_layernorm, layer_id)

                if not isinstance(layer.self_attention, IdentityOp):
                    if "MLASelfAttention" in str(type(layer.self_attention)):
                        if hasattr(layer.self_attention, "linear_q_proj"):
                            layer_pbar.set_description("Importing MLA (without q LoRA)")
                            self.rules["linear_q_proj"](
                                layer.self_attention.linear_q_proj, layer_id
                            )
                        else:
                            layer_pbar.set_description("Importing MLA (with q LoRA)")
                            self.rules["linear_q_down_proj"](
                                layer.self_attention.linear_q_down_proj, layer_id
                            )
                            self.rules["linear_q_layernorm"](
                                layer.self_attention.q_layernorm, layer_id
                            )
                            self.rules["linear_q_up_proj"](
                                layer.self_attention.linear_q_up_proj, layer_id
                            )
                        self.rules["linear_kv_down_proj"](
                            layer.self_attention.linear_kv_down_proj, layer_id
                        )
                        self.rules["linear_kv_layernorm"](
                            layer.self_attention.kv_layernorm, layer_id
                        )
                        self.rules["linear_kv_up_proj"](
                            layer.self_attention.linear_kv_up_proj, layer_id
                        )
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
                    else:
                        layer_pbar.set_description("Importing GQA/MHA")
                        if layer.self_attention.q_layernorm is not None and not isinstance(
                            layer.self_attention.q_layernorm, IdentityOp
                        ):
                            self.rules["q_layernorm"](layer.self_attention.q_layernorm, layer_id)
                            self.rules["k_layernorm"](layer.self_attention.k_layernorm, layer_id)
                        self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)

                if not isinstance(layer.pre_mlp_layernorm, IdentityOp):
                    self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

                if not isinstance(layer.mlp, IdentityOp):
                    if "MoE" in str(type(layer.mlp)):
                        layer_pbar.set_description("Importing MoE")
                        self.rules["router"](layer.mlp.router, layer_id)
                        if (
                            hasattr(layer.mlp, "shared_experts")
                            and layer.mlp.shared_experts is not None
                        ):
                            layer_pbar.set_description("Importing MoE shared experts")
                            self.rules["shared_experts.linear_fc1"](
                                layer.mlp.shared_experts.linear_fc1, layer_id
                            )
                            self.rules["shared_experts.linear_fc2"](
                                layer.mlp.shared_experts.linear_fc2, layer_id
                            )
                        for expert_id, expert in tqdm(
                            enumerate(layer.mlp.experts.local_experts),
                            desc="Importing MoE local experts",
                            leave=False,
                            disable=self.disable_tqdm,
                        ):
                            self.rules["local_experts.linear_fc1"](
                                expert.linear_fc1, layer_id, expert_id
                            )
                            self.rules["local_experts.linear_fc2"](
                                expert.linear_fc2, layer_id, expert_id
                            )
                    else:
                        layer_pbar.set_description("Importing MLP")
                        self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                        self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        if hasattr(model.decoder, "final_norm") and model.decoder.final_norm:
            self.rules["final_norm"](model.decoder.final_norm)

        # Output layer
        if hasattr(model, "output_layer") and not model.share_embeddings_and_output_weights:
            self.rules["output_layer"](model.output_layer)

        # MTP
        if hasattr(model, "mtp"):
            # MTP is the last layer in DeepSeek V3/R1
            layer_id += 1
            for mtp in model.mtp:
                self.rules["mtp.fc"](mtp.fc, layer_id)
                self.rules["mtp.enorm"](mtp.enorm, layer_id)
                self.rules["mtp.hnorm"](mtp.hnorm, layer_id)
                self.rules["mtp.input_layernorm"](mtp.decoder.layers[0].input_layernorm, layer_id)
                if hasattr(mtp.decoder.layers[0].self_attention, "linear_q_proj"):
                    self.rules["mtp.linear_q_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_proj, layer_id
                    )
                else:
                    self.rules["mtp.linear_q_down_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_down_proj, layer_id
                    )
                    self.rules["mtp.linear_q_layernorm"](
                        mtp.decoder.layers[0].self_attention.q_layernorm, layer_id
                    )
                    self.rules["mtp.linear_q_up_proj"](
                        mtp.decoder.layers[0].self_attention.linear_q_up_proj, layer_id
                    )
                self.rules["mtp.linear_kv_down_proj"](
                    mtp.decoder.layers[0].self_attention.linear_kv_down_proj, layer_id
                )
                self.rules["mtp.linear_kv_layernorm"](
                    mtp.decoder.layers[0].self_attention.kv_layernorm, layer_id
                )
                self.rules["mtp.linear_kv_up_proj"](
                    mtp.decoder.layers[0].self_attention.linear_kv_up_proj, layer_id
                )
                self.rules["mtp.linear_proj"](
                    mtp.decoder.layers[0].self_attention.linear_proj, layer_id
                )
                self.rules["mtp.pre_mlp_layernorm"](
                    mtp.decoder.layers[0].pre_mlp_layernorm, layer_id
                )
                self.rules["mtp.router"](mtp.decoder.layers[0].mlp.router, layer_id)
                self.rules["mtp.shared_experts.linear_fc1"](
                    mtp.decoder.layers[0].mlp.shared_experts.linear_fc1, layer_id
                )
                self.rules["mtp.shared_experts.linear_fc2"](
                    mtp.decoder.layers[0].mlp.shared_experts.linear_fc2, layer_id
                )
                for expert_id, expert in tqdm(
                    enumerate(mtp.decoder.layers[0].mlp.experts.local_experts),
                    desc="Importing MoE local experts",
                    leave=False,
                    disable=self.disable_tqdm,
                ):
                    self.rules["mtp.local_experts.linear_fc1"](
                        expert.linear_fc1, layer_id, expert_id
                    )
                    self.rules["mtp.local_experts.linear_fc2"](
                        expert.linear_fc2, layer_id, expert_id
                    )


def import_mcore_gpt_from_hf(
    model: torch.nn.Module,
    pretrained_model_path: str,
    workspace_dir: str | None = None,
    dtype: torch.dtype = torch.float16,
):
    """Import GPTModel state_dict from supported HuggingFace pretrained model path.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_path: A path to a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to import.
    """
    importer = GPTModelImporter(
        model, pretrained_model_path, workspace_dir=workspace_dir, dtype=dtype
    )
    importer._import_state_dict()
