# Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/fp8_cast_bf16.py
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from modelopt import __version__
from modelopt.torch.export.quant_utils import process_layer_quant_config
from modelopt.torch.export.tensorrt_llm_utils import _prefix_wildcard_summarize_exclude_modules
from modelopt.torch.quantization.qtensor import NVFP4QTensor

sys.path.append(str(Path(__file__).resolve().parent / "DeepSeek-V3/inference"))
from kernel import weight_dequant


def load_and_preprocess_state_dict(modelopt_state_root, world_size=8):
    state_dict_list = []
    # load amax from nvfp4 state dict
    for rank in range(world_size):
        state_dict_list.append(
            torch.load(f"{modelopt_state_root}/amax_dict_rank{rank}-mp{world_size}.pt")
        )

    # calculate the max acroos all TP
    merged_state_dict = state_dict_list[0]
    for rank in range(world_size):
        for key, amax in state_dict_list[rank].items():
            if key in merged_state_dict.items():
                amax = torch.max(amax, merged_state_dict[key].to(amax.device))
            merged_state_dict[key] = amax

    # renaming the module to match HF modeling
    mappig = {
        "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
        "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
        "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
        "ffn.w1": "mlp.gate_proj",
        "ffn.w2": "mlp.down_proj",
        "ffn.w3": "mlp.up_proj",
    }
    renamed_state_dict = {}
    for ori_key, amax in merged_state_dict.items():
        item = merged_state_dict[ori_key]
        key = ori_key.replace("layers", "model.layers")
        for original_pattern, replace_pattern in mappig.items():
            key = key.replace(original_pattern, replace_pattern)

        # ffn.experts.xx.w1/w2/w3- > mlp.experts.xx.gate_proj/down_proj/up_proj
        key = re.sub(r"ffn\.experts\.(\d+)\.w1", r"mlp.experts.\1.gate_proj", key)
        key = re.sub(r"ffn\.experts\.(\d+)\.w2", r"mlp.experts.\1.down_proj", key)
        key = re.sub(r"ffn\.experts\.(\d+)\.w3", r"mlp.experts.\1.up_proj", key)

        renamed_state_dict[key] = item

    # set amax for modules to be fused and make sure they share the same input
    for key, amax in renamed_state_dict.items():
        if "up_proj" in key:
            gate_proj_key = key.replace("up_proj", "gate_proj")
            if "weight_quantizer" in key:
                fused_amax = torch.max(amax, renamed_state_dict[gate_proj_key])
                renamed_state_dict[key] = fused_amax
                renamed_state_dict[gate_proj_key] = fused_amax
            elif "input_quantizer" in key:
                assert amax == renamed_state_dict[gate_proj_key]
            else:
                raise NotImplementedError

    return renamed_state_dict


def find_safetensors_files(directory: str):
    """Recursively finds all `.safetensors` files in the specified directory."""
    safetensors_files = glob.glob(os.path.join(directory, "**", "*.safetensors"), recursive=True)

    return safetensors_files


# Adopted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/fp8_cast_bf16.py
def convert_fp8_ckpt_to_nvfp4(renamed_state_dict, fp8_root, save_root):
    def amax_to_weights_scaling_factor_2(amax):
        return amax.float() / 6.0 / 448.0

    torch.set_default_dtype(torch.bfloat16)
    model_index_file = os.path.join(fp8_root, "model.safetensors.index.json")
    os.makedirs(save_root, exist_ok=True)
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_root, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    all_safetensor_files = find_safetensors_files(fp8_root)
    for safetensor_file in tqdm(all_safetensor_files, desc="Quantizing"):
        # convert FP8 safetensor to BF16
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        bf16_state_dict = {}
        for key, item in current_state_dict.items():
            # insert inv
            if key.endswith("_scale_inv"):
                continue
            elif item.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{key}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(key)
                    bf16_state_dict[key] = weight_dequant(item, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {key}, skipping conversion")
                    bf16_state_dict[key] = item
            else:
                bf16_state_dict[key] = item

        new_dict_nvfp4 = {}
        for key, item in bf16_state_dict.items():
            if "weight" in key and any(sub in key for sub in ["up_proj", "gate_proj", "down_proj"]):
                amax_key = key + "_quantizer._amax"
                layer_name = key.replace(".weight", "")
                input_scale_key = layer_name + ".input_quantizer._amax"
                if amax_key in renamed_state_dict:
                    weight = item
                    weights_scaling_factor_2 = amax_to_weights_scaling_factor_2(
                        renamed_state_dict[amax_key]
                    )
                    input_scaling_factor = amax_to_weights_scaling_factor_2(
                        renamed_state_dict[input_scale_key]
                    )
                    quantized_weight, scaling_factor, scaling_factor_2 = NVFP4QTensor.quantize(
                        weight.to(weights_scaling_factor_2.device),
                        16,
                        None,
                        weights_scaling_factor_2,
                    )

                    # adding input_scale, weight_scaling_factor,
                    new_dict_nvfp4[key] = quantized_weight._quantized_data
                    new_dict_nvfp4[layer_name + ".input_scale"] = input_scaling_factor
                    new_dict_nvfp4[key + "_scale"] = scaling_factor
                    new_dict_nvfp4[key + "_scale_2"] = scaling_factor_2
                    continue
            new_dict_nvfp4[key] = item

        save_file(new_dict_nvfp4, os.path.join(save_root, file_name))

        # Memory management: keep only the 2 most recently used files
        while len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(save_root, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def construct_quant_config(save_root):
    all_keys = set()

    all_safetensor_files = find_safetensors_files(save_root)
    for safetensor_file in tqdm(all_safetensor_files, desc="Gathering all keys"):
        state_dict = load_file(safetensor_file)  # Load tensors
        all_keys.update(state_dict.keys())  # Add keys to the set

    # construct quantization layer dict
    layer_config_dict = {}
    for k in all_keys:
        if k.endswith(".weight"):
            name = k.rsplit(".weight", 1)[0]
            # Note: assume nvfp4 if scales exists
            quantization = None
            block_size = None
            if k + "_scale" in all_keys:
                quantization = "nvfp4"
                block_size = 16
            layer_config_dict.update({name + ".quantization": quantization})
            layer_config_dict.update({name + ".awq_block_size": block_size})

    per_layer_quantization = process_layer_quant_config(layer_config_dict)

    def get_exclude_modules(layer_config_dict):
        """Returns the list of modules to exclude from quantization."""
        quantized_layers = set()
        unquantized_layers = set()
        for k, v in layer_config_dict.items():
            if "awq_block_size" in k:
                continue

            prefix = ".".join(k.rsplit(".", 1)[:-1])
            if v is not None:
                quantized_layers.add(prefix)
            else:
                unquantized_layers.add(prefix)

        unquantized_layers = unquantized_layers - quantized_layers

        res_with_wildcards = _prefix_wildcard_summarize_exclude_modules(
            unquantized_layers, quantized_layers
        )
        return list(res_with_wildcards)

    exclude_modules = []
    if not per_layer_quantization:
        exclude_modules = get_exclude_modules(layer_config_dict)

    # construct hf_quant_config
    quant_config: dict[str, Any] = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
    }
    quant_config["quantization"].update(
        {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "exclude_modules": exclude_modules,
        }
    )
    # dump the config
    with open(os.path.join(save_root, "hf_quant_config.json"), "w") as json_file:
        json.dump(quant_config, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--amax_path", type=str, required=True, help="path to the expoted amax.")
    parser.add_argument(
        "--fp4_path", type=str, required=True, help="path to save the fp4 checkpoint."
    )
    parser.add_argument("--fp8_hf_path", type=str, required=True, help="fp8 hf ckpt.")
    parser.add_argument("--world_size", type=int, required=True, help="world size used by ptq.")
    args = parser.parse_args()

    renamed_state_dict = load_and_preprocess_state_dict(
        modelopt_state_root=args.amax_path, world_size=args.world_size
    )
    convert_fp8_ckpt_to_nvfp4(
        renamed_state_dict, fp8_root=args.fp8_hf_path, save_root=args.fp4_path
    )
    construct_quant_config(save_root=args.fp4_path)
