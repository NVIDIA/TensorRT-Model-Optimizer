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

from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import (
    get_mcore_gpt_model,
    initialize_for_megatron,
    run_mcore_gpt_inference_with_dummy_input,
)

import modelopt.torch.prune as mtp


def _test_mcore_gpt_width_pruning(
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    pruned_ffn_div,
    pruned_num_attention_heads_div,
    pruned_num_query_groups_div,
    pruned_hidden_size_div,
    pruned_num_layers_div,
    rank,
    size,
):
    num_layers = min(size * 2, 8)
    hidden_size = 256
    ffn_hidden_size = 256
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=size)

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
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

    def forward_loop(m):
        for _ in range(5):
            run_mcore_gpt_inference_with_dummy_input(m, batch_size, hidden_size)

    pruned_ffn = ffn_hidden_size // pruned_ffn_div
    pruned_num_attention_heads = num_attention_heads // pruned_num_attention_heads_div
    pruned_num_query_groups = num_query_groups // pruned_num_query_groups_div
    pruned_num_heads_per_group = pruned_num_attention_heads // pruned_num_query_groups
    pruned_hidden_size = hidden_size // pruned_hidden_size_div
    pruned_num_layers = num_layers // pruned_num_layers_div

    export_config = {}
    if pruned_ffn_div != 1:
        export_config["ffn_hidden_size"] = pruned_ffn
    if pruned_num_attention_heads_div != 1 or pruned_num_query_groups_div != 1:
        export_config["num_attention_heads"] = pruned_num_attention_heads
        export_config["num_query_groups"] = pruned_num_query_groups
    if pruned_hidden_size_div != 1:
        export_config["hidden_size"] = pruned_hidden_size
    if pruned_num_layers_div != 1:
        export_config["num_layers"] = pruned_num_layers

    model, _ = mtp.prune(
        model,
        mode="mcore_gpt_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"forward_loop": forward_loop},
    )

    # Assert weights are pruned correctly
    for layer in model.decoder.layers:
        assert layer.mlp.linear_fc1.weight.shape == (
            pruned_ffn * (2 if activation_func == "swiglu" else 1),
            pruned_hidden_size,
        )
        assert layer.mlp.linear_fc2.weight.shape == (pruned_hidden_size, pruned_ffn)
        assert layer.self_attention.linear_qkv.weight.shape == (
            (pruned_num_heads_per_group + 2) * pruned_num_query_groups * model.config.kv_channels,
            pruned_hidden_size,
        )
        assert layer.self_attention.linear_proj.weight.shape == (
            pruned_hidden_size,
            pruned_num_heads_per_group * pruned_num_query_groups * model.config.kv_channels,
        )

    # Assert forward pass works on the pruned model
    run_mcore_gpt_inference_with_dummy_input(model, batch_size, pruned_hidden_size)

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.num_query_groups == pruned_num_query_groups
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.num_layers == pruned_num_layers


@pytest.mark.parametrize(
    "num_attention_heads, num_query_groups, activation_func, normalization, ffn_div, num_attention_heads_div, num_query_groups_div, hidden_size_div, num_layers_div",  # noqa: E501
    [
        (8, 8, "squared_relu", "LayerNorm", 4, 1, 1, 1, 1),  # MHA - pruned ffn/4
        (8, 4, "squared_relu", "RMSNorm", 1, 2, 2, 1, 1),  # GQA - pruned attention/2
        (8, 4, "swiglu", "RMSNorm", 1, 1, 1, 4, 1),  # GQA - pruned hidden_size/4
        (8, 8, "swiglu", "LayerNorm", 1, 1, 1, 1, 2),  # MHA - pruned num_layers/2
        (8, 4, "swiglu", "RMSNorm", 2, 2, 2, 2, 2),  # GQA - pruned all/2
    ],
)
def test_mcore_gpt_width_pruning(
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    ffn_div,
    num_attention_heads_div,
    num_query_groups_div,
    hidden_size_div,
    num_layers_div,
):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_mcore_gpt_width_pruning,
            num_attention_heads,
            num_query_groups,
            activation_func,
            normalization,
            ffn_div,
            num_attention_heads_div,
            num_query_groups_div,
            hidden_size_div,
            num_layers_div,
        ),
        backend="nccl",
    )
