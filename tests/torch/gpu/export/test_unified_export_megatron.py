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

import json
import tempfile
from functools import partial

import pytest
import torch
import transformers
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import (
    get_mcore_gpt_model,
    initialize_for_megatron,
)

import modelopt.torch.speculative as mtsp
from modelopt.torch.export import unified_export_megatron
from modelopt.torch.speculative.plugins.megatron import (
    _DynamicEagleGPTModel,
    _DynamicMedusaGPTModel,
)


def _test_unified_export_megatron(model_type, arch, algo, rank, size):
    num_layers = 2
    hidden_size = 256
    num_attention_heads = 8
    num_query_groups = size
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    arch = "NemotronForCausalLM" if model_type == "nemotron" else "LlamaForCausalLM"
    activation_func = "squared_relu" if model_type == "nemotron" else "swiglu"
    normalization = "LayerNorm" if model_type == "nemotron" else "RMSNorm"

    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=size)

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    )

    if algo == "medusa":
        config = {
            "medusa_num_heads": 1,
            "medusa_num_layers": 1,
        }
        model = mtsp.convert(model, [("medusa", config)])
        assert isinstance(model, _DynamicMedusaGPTModel)
    elif algo == "eagle":
        config = {"eagle_num_layers": 1}
        model = mtsp.convert(model, [("eagle", config)])
        assert isinstance(model, _DynamicEagleGPTModel)

    with tempfile.TemporaryDirectory() as pretrained_config_dir:
        pretrained_config = {
            "architectures": [arch],
            "attention_bias": False,
            "hidden_size": hidden_size,
            "intermediate_size": ffn_hidden_size,
            "max_position_embeddings": max_sequence_length,
            "model_type": "llama",
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_layers,
            "num_key_value_heads": num_query_groups,
            "torch_dtype": "bfloat16",
        }

        with open(pretrained_config_dir + "/config.json", "w") as f:
            json.dump(pretrained_config, f)

        unified_export_megatron.export_mcore_gpt_to_hf(
            model,
            pretrained_config_dir if arch is not None else None,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    "model_type,arch,algo",
    [
        ("nemotron", None, None),
        ("nemotron", "NemotronForCausalLM", None),
        ("nemotron", "NemotronForCausalLM", "eagle"),
        ("nemotron", "NemotronForCausalLM", "medusa"),
        ("llama", None, None),
        ("llama", "LlamaForCausalLM", None),
        ("llama", "LlamaForCausalLM", "eagle"),
        ("llama", "LlamaForCausalLM", "medusa"),
    ],
)
def test_unified_export_megatron(model_type, arch, algo):
    spawn_multiprocess_job(
        size=1,
        job=partial(
            _test_unified_export_megatron,
            model_type,
            arch,
            algo,
        ),
        backend="nccl",
    )


def _test_unified_import_megatron(pretrained_model, rank, size):
    config = transformers.AutoConfig.from_pretrained(pretrained_model)

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads
    ffn_hidden_size = config.intermediate_size
    max_sequence_length = config.max_position_embeddings
    vocab_size = config.vocab_size
    activation_func = "swiglu"
    normalization = "RMSNorm"

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    )

    unified_export_megatron.import_mcore_gpt_from_hf(
        model,
        pretrained_model,
    )


@pytest.mark.parametrize(
    "pretrained_model",
    [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
)
def test_unified_import_megatron(pretrained_model):
    spawn_multiprocess_job(
        size=1,
        job=partial(
            _test_unified_import_megatron,
            pretrained_model,
        ),
        backend="nccl",
    )
