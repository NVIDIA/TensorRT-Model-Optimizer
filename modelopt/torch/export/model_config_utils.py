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

"""Common utils for the ModelConfig."""

import dataclasses
import math
from typing import Optional, Union, get_args, get_origin

import numpy as np
import torch

from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    DecoderLayerConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
    ModelConfig,
    MOEConfig,
    QKVConfig,
)
from .quant_utils import to_quantized_weight

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype("V2", metadata={"dtype": "bfloat16"})


def _numpy_to_torch(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x

    if x.dtype != np_bfloat16:
        return torch.tensor(x)
    return torch.tensor(x.view(np.int16)).view(torch.bfloat16)


def model_config_to_dict(model_config: ModelConfig) -> dict:
    """Converts the instance to a python dict."""
    assert model_config is not None, "model_config is None"

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        return obj

    return _to_dict(model_config)


def split_config_and_weights(
    config,
    weights: dict[str, torch.tensor],
    prefix: str = "transformer",
    layer_config_dict: dict = {},
):
    """Util function to split the weights or any torch.Tensor in nested config to weights.

    A weight id starts with transformers or lm_head will also be generated to link the original key to the weights dict.
    The weights in the weights dict are contiguous.

    layer_config_dict: A dictionary containing layerwise quantization format information and awq_block_size information
    when relevant. It is used to export quantization.json for auto_quant checkpoint.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            if k == "lm_head" and "medusa_heads" not in prefix:
                # lm_head is not part of the transformer.
                array_key = k
            elif k == "experts":
                # Omit the 'experts' key that is not in the model name
                array_key = prefix
            elif k == "medusa_heads":
                # medusa_heads is not part of the transformer
                array_key = k
            elif "rel_attn_table" in prefix:
                # rel_attn_table is treated as a weight for quantize loop whereas in TRTLLM it is a Tensor
                array_key = prefix
            else:
                array_key = f"{prefix}.{k}"

            # Construct per_layer quantization dictionary, with block size information
            if array_key != "transformer.quantization" and (
                "quantization" in array_key or "awq_block_size" in array_key
            ):
                layer_config_dict[array_key] = v

            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[k] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key, layer_config_dict)
    elif isinstance(config, list):
        for i, v in enumerate(config):
            array_key = f"{prefix}.{i}"
            if isinstance(v, torch.Tensor):
                weights[array_key] = v
                config[i] = f"{array_key}"
            else:
                split_config_and_weights(v, weights, array_key, layer_config_dict)


def _unified_weights_key(k: str) -> str:
    """Try to unify the weights dict key between old npz and the new safetensors format."""
    prefixes = ["transformer.", "_np:"]
    for prefix in prefixes:
        if k.startswith(prefix):
            k = k[len(prefix) :]

    k = k.replace("final_layernorm", "ln_f")

    return k.replace(":", ".")


def _restore_model_config(model_config, weights: dict[str, Union[np.ndarray, torch.Tensor]]):
    def _is_tensor_key(k):
        return isinstance(k, str) and _unified_weights_key(k) in weights

    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if _is_tensor_key(v):
                model_config[k] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)
    if isinstance(model_config, list):
        for i, v in enumerate(model_config):
            if _is_tensor_key(v):
                model_config[i] = _numpy_to_torch(weights[_unified_weights_key(v)])
            else:
                _restore_model_config(v, weights)


def restore_model_config(model_config, weights: dict[str, Union[np.ndarray, torch.Tensor]]):
    """Recursively restores the model_config from json and loads np.ndarray or torch.Tensor weights from weights."""
    unified_key_weights = {}
    for k, v in weights.items():
        unified_key_weights[_unified_weights_key(k)] = v

    _restore_model_config(model_config, unified_key_weights)


def _from_dict(class_type, data):
    """Helper function to load the data as a class_type. class_type must be a dataclass."""
    if data is None:
        return None

    # class_type of quantization is Optional[str], which is catergrorized as Union
    if class_type != Optional[str] and get_origin(class_type) == Union:
        # Handle QKV
        if all([key in data for key in ["q", "k", "v"]]):
            # splitted qkv case
            class_type = QKVConfig
        elif all([key in data for key in ["router", "experts"]]):
            # moe
            class_type = MOEConfig
        elif all([key in data for key in ["fc", "gate", "proj"]]):
            # mlp
            class_type = MLPConfig
        else:
            # merged qkv case
            assert "linear_type" in data, f"{data} is not a valid LinearConfig"
            class_type = LinearConfig

    if dataclasses.is_dataclass(class_type):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(class_type)}
        fields_map = {}
        for k, v in data.items():
            if k in fieldtypes:
                # We only handle keys available in the fields.
                # Deprecated fields in the checkpoint will be ignored.
                fields_map[k] = _from_dict(fieldtypes[k], v)
        return class_type(**fields_map)
    elif get_origin(class_type) is list and dataclasses.is_dataclass(get_args(class_type)[0]):
        list_value = []
        for child in data:
            child_class_type = get_args(class_type)[0]
            list_value.append(_from_dict(child_class_type, child))
        return list_value
    else:
        return data


def model_config_from_dict(d: dict) -> ModelConfig:
    """Load a dict to a `ModelConfig` instance."""
    config_type = ModelConfig

    config_type_map = {}
    for t in [ModelConfig, DecoderLayerConfig, LayernormConfig, LinearConfig]:
        config_type_map[t.__name__] = t

    if "__name__" in d:
        config_name = d.pop("__name__")
        try:
            config_type = config_type_map[config_name]
        except Exception as e:
            raise NotImplementedError(f"{config_name} not supported") from e

    return _from_dict(config_type, d)


def pad_weights(weights, tp_size):
    """Returns the padded weights to tp_size."""
    assert len(weights.shape) > 1

    def _pad_size(original_size, tp_size):
        return int(math.ceil(original_size / tp_size) * tp_size)

    original_size = weights.shape[0]
    padded_size = _pad_size(original_size, tp_size)

    if original_size != padded_size:
        pad_width = padded_size - original_size
        return torch.nn.functional.pad(weights, (0, 0, 0, pad_width), "constant", value=0)
    return weights


def merge_qkv(model_config):
    """Merges the qkv fields in model_config from QKVConfig to a single LinearConfig."""
    for decoder_config in model_config.layers:
        for attention_key in ["attention", "self_attention", "cross_attention"]:
            attention = getattr(decoder_config, attention_key, None)
            if attention and isinstance(attention.qkv, QKVConfig):
                splitted_qkv = attention.qkv
                attention.qkv = LinearConfig()
                attention.qkv.weight = splitted_qkv.weight
                attention.qkv.bias = splitted_qkv.bias
                attention.qkv.activation_scaling_factor = splitted_qkv.activation_scaling_factor
                attention.qkv.weights_scaling_factor = splitted_qkv.weights_scaling_factor
                attention.qkv.weights_scaling_factor_2 = splitted_qkv.weights_scaling_factor_2
                attention.qkv.prequant_scaling_factor = splitted_qkv.prequant_scaling_factor
                attention.qkv.awq_block_size = splitted_qkv.awq_block_size
                # Assert q,k,v have same quantization formats before merging
                assert (
                    splitted_qkv.q.quantization
                    == splitted_qkv.k.quantization
                    == splitted_qkv.v.quantization
                ), "Quantization formats of q,k,v must be the same."
                attention.qkv.quantization = splitted_qkv.q.quantization

                # Collect GPU memory from the deleted tensors
                del splitted_qkv


def pack_linear_weights(model_config: ModelConfig):
    """Packs the quantized linear weights in the model_config to the quantized format."""

    def _linear_layer_to_quantized_weight(linear_layers):
        for linear_layer in linear_layers:
            if isinstance(linear_layer, LinearConfig):
                # Check if quantization of the layer is None to support auto_quant
                if (
                    linear_layer.weights_scaling_factor is not None
                    and linear_layer.quantization is not None
                ):
                    # Quantize on CPU if we are short of GPU memory.
                    # Using 2x of the tensor size as a threshold.
                    if linear_layer.weight.is_cuda:
                        free_mem, _ = torch.cuda.mem_get_info(linear_layer.weight.device)
                        if (
                            free_mem
                            < 2
                            * linear_layer.weight.element_size()
                            * linear_layer.weight.nelement()
                        ):
                            linear_layer.weight = linear_layer.weight.cpu()

                    # Save the quantize layer weights to cpu and save gpu memory.
                    linear_layer.weight = to_quantized_weight(
                        linear_layer.weight,
                        linear_layer.weights_scaling_factor,
                        linear_layer.quantization,
                        linear_layer.weights_scaling_factor_2,
                        linear_layer.awq_block_size,
                    ).cpu()

                    linear_layer.weights_scaling_factor = linear_layer.weights_scaling_factor.cpu()

    if not model_config.quantization:
        return

    attention_key_list = ["attention", "self_attention", "cross_attention"]
    for decoder_config in model_config.layers:
        linear_layers = []
        if any([hasattr(decoder_config, attention_key) for attention_key in attention_key_list]):
            for attention_key in attention_key_list:
                attention = getattr(decoder_config, attention_key, None)
                if attention:
                    linear_layers += [
                        attention.qkv,
                        attention.dense,
                    ]
        if decoder_config.recurrent:
            linear_layers = [
                decoder_config.recurrent.linear_y,
                decoder_config.recurrent.linear_x,
                decoder_config.recurrent.linear_out,
            ]

        if isinstance(decoder_config.mlp, MOEConfig):
            if model_config.quantization not in [
                QUANTIZATION_FP8,
                QUANTIZATION_INT4_AWQ,
                QUANTIZATION_NVFP4,
                QUANTIZATION_NVFP4_AWQ,
            ]:
                raise NotImplementedError(
                    f"MOE quantization for {model_config.quantization} is not supported yet."
                )
            else:
                linear_layers.append(decoder_config.mlp.experts.fc)
                linear_layers.append(decoder_config.mlp.experts.proj)
        elif decoder_config.mlp is not None:
            linear_layers.append(decoder_config.mlp.fc)
            linear_layers.append(decoder_config.mlp.proj)
            linear_layers.append(decoder_config.mlp.gate)

        _linear_layer_to_quantized_weight(linear_layers)

    if model_config.medusa_heads is not None:
        linear_layers = []

        for head in model_config.medusa_heads:
            linear_layers.append(head.lm_head)
            for layer in head.medusa_layers:
                linear_layers.append(layer.linear)

        _linear_layer_to_quantized_weight(linear_layers)

    # lm_head can be quantized by AutoQuant
    if model_config.lm_head is not None:
        _linear_layer_to_quantized_weight([model_config.lm_head])
