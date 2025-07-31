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

"""Convert modelopt quantization export config to align with llm-compressor config format."""

from typing import Any


def convert_hf_quant_config_format(input_config: dict[str, Any]) -> dict[str, Any]:
    """Converts modelopt quantization config dictionary to align with llm-compressor config format.

    Args:
        input_config: The original quantization config dictionary.

    Note:
        The "targets" field specifies which PyTorch module types to quantize. Compressed-tensors
        works with any PyTorch module type and uses dynamic matching against module.__class__.__name__.
        Typically this includes "Linear" modules, but can also include "Embedding" and other types.

        See: https://github.com/neuralmagic/compressed-tensors/blob/fa6a48f1da6b47106912bcd25eba7171ba7cfec7/src/sparsetensors/quantization/quant_scheme.py#L29
        Example usage: https://github.com/neuralmagic/compressed-tensors/blob/9938a6ec6e10498d39a3071dfd1c40e3939ee80b/tests/test_quantization/lifecycle/test_apply.py#L118

    Example:

        .. code-block:: python

            {
                "producer": {"name": "modelopt", "version": "0.19.0"},
                "quantization": {
                    "quant_algo": "FP8",
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": ["lm_head"],
                },
            }

    Returns:
        A new dictionary in the target format.

        Example (for FP8 input):

        .. code-block:: python

            {
                "config_groups": {
                    "group_0": {
                        "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                        "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    }
                },
                "ignore": ["lm_head"],
                "quant_algo": "FP8",
                "kv_cache_scheme": "FP8",
                "producer": {"name": "modelopt", "version": "0.29.0"},
            }
    """
    new_config: dict[str, Any] = {}

    original_quantization_details = input_config.get("quantization", {})
    quant_algo_value = original_quantization_details.get("quant_algo")

    # This structure is derived based on the example for "FP8" and "NVFP4"
    # TODO: Handle other quantization algorithms
    if quant_algo_value == "FP8":
        config_group_details = {
            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
            "targets": ["Linear"],
        }
        new_config["config_groups"] = {"group_0": config_group_details}
    elif quant_algo_value == "NVFP4":
        group_size = original_quantization_details.get("group_size", 16)
        config_group_details = {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": group_size,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": group_size},
            "targets": ["Linear"],
        }
        new_config["config_groups"] = {"group_0": config_group_details}

    exclude_modules = original_quantization_details.get("exclude_modules")

    new_config["ignore"] = exclude_modules if exclude_modules is not None else []

    if quant_algo_value:
        new_config["quant_algo"] = quant_algo_value

    kv_cache_quant_algo = original_quantization_details.get("kv_cache_quant_algo")
    if kv_cache_quant_algo:
        if kv_cache_quant_algo == "FP8":
            new_config["kv_cache_scheme"] = {"dynamic": False, "num_bits": 8, "type": "float"}
        else:
            # TODO: Handle other kv cache quantization algorithms
            new_config["kv_cache_scheme"] = kv_cache_quant_algo

    producer_info = input_config.get("producer")
    if producer_info:
        new_config["producer"] = producer_info

    new_config["quant_method"] = "modelopt"

    return new_config
