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
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch_misc import set_seed
from megatron.core.parallel_state import destroy_model_parallel
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

import modelopt.torch.nas as mtn
from modelopt.torch.nas.plugins.megatron import (
    _DynamicColumnParallelLinear,
    _DynamicMCoreLanguageModel,
    _DynamicMLP,
    _DynamicProjRowParallelLinear,
    _DynamicQKVColumnParallelLinear,
    _DynamicRowParallelLinear,
    _DynamicSelfAttention,
    _DynamicTransformerLayer,
    _DynamicVocabParallelEmbedding,
    expand_head_indices,
)
from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.opt.utils import named_dynamic_modules, search_space_size
from modelopt.torch.prune.plugins.mcore_minitron import _convert_model_to_dynamic_space
from modelopt.torch.utils import flatten_tree
from modelopt.torch.utils.random import centroid

SEED = 1234


def _test_gpt_search_space(
    num_attention_heads, num_query_groups, activation_func, normalization, rank, size
):
    channel_divisor = 64

    num_layers = min(size * 2, 8)
    hidden_size = 256
    ffn_hidden_size = 128
    max_sequence_length = 16
    vocab_size = 64
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
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

    model = mtn.convert(model, "mcore_minitron")

    assert isinstance(model, _DynamicMCoreLanguageModel)
    for m in model.modules():
        if isinstance(m, VocabParallelEmbedding):
            assert isinstance(m, _DynamicVocabParallelEmbedding)
        elif isinstance(m, TransformerLayer):
            assert isinstance(m, _DynamicTransformerLayer)
        elif isinstance(m, MLP):
            assert isinstance(m, _DynamicMLP)
            assert isinstance(m.linear_fc1, _DynamicColumnParallelLinear)
            assert isinstance(m.linear_fc2, _DynamicRowParallelLinear)
        elif isinstance(m, SelfAttention):
            assert isinstance(m, _DynamicSelfAttention)
            assert isinstance(m.linear_qkv, _DynamicQKVColumnParallelLinear)
            assert isinstance(m.linear_proj, _DynamicProjRowParallelLinear)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    ffn_hidden_size_choices = ffn_hidden_size // channel_divisor
    hidden_size_choices = hidden_size // channel_divisor
    num_layers_per_pp = num_layers // size
    assert (
        ss_size_per_pp
        == (num_attention_heads * ffn_hidden_size_choices) ** num_layers_per_pp
        * num_layers
        * hidden_size_choices
    )

    # Make sure forward pass works on min and centroid subnets
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    for sample_func in [min, max, centroid]:
        mtn.sample(model, sample_func)
        output = run_mcore_inference(model, prompt_tokens)
        assert output.shape == (batch_size, max_sequence_length, vocab_size)

    # Make sure export and forward pass works on centroid model
    mtn.export(model)
    _ = run_mcore_inference(model, prompt_tokens, model.hidden_size)
    assert not any(named_dynamic_modules(model))


@pytest.mark.parametrize(
    ("num_attention_heads", "num_query_groups", "activation_func", "normalization"),
    [
        (8, 8, "squared_relu", "LayerNorm"),  # MHA
        (8, 4, "swiglu", "RMSNorm"),  # GQA
        # (8, 1, "swiglu", "RMSNorm"),  # MQA
    ],
)
def test_gpt_search_space(num_attention_heads, num_query_groups, activation_func, normalization):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_gpt_search_space,
            num_attention_heads,
            num_query_groups,
            activation_func,
            normalization,
        ),
        backend="nccl",
    )


def _test_gpt_parameter_sorting(activation_func, rank, size):
    num_layers = size
    hidden_size = 128
    num_attention_heads = 8
    num_query_groups = 4
    ffn_hidden_size = 64
    max_sequence_length = 32
    vocab_size = 128
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        bf16=False,
    )

    # Randomize layernorm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "layernorm" in n and not isinstance(m, IdentityOp):
            m.weight.data = torch.randn_like(m.weight)

    model.eval()
    dynamic_space = _convert_model_to_dynamic_space(model)

    # Compute activations for sorting
    for _ in range(5):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    mtn.utils.sort_parameters(model)

    # check if all ffn_hidden_size, num_heads_per_group, num_query_groups, hidden_size have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # 3 hps per layer + 1 for hidden_size (num_layers is not sorted!)
    assert len(sortable_per_pp) == 3 * num_layers // size + 1

    # Export since sorting force reassigns SelfAttention weights which we dont want to re-sort!
    # TODO: ideally we shouldn't need this
    dynamic_space.export(DMRegistry)

    # sanity check if the model functionality is preserved after sorting
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    assert all(
        torch.allclose(t1, t2, rtol=1e-5, atol=1e-3)
        for t1, t2 in zip(flatten_tree(y1)[0], flatten_tree(y2)[0])
    )


@pytest.mark.parametrize("activation_func", ["swiglu"])
def test_gpt_parameter_sorting(activation_func, need_2_gpus):
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_gpt_parameter_sorting, activation_func),
        backend="nccl",
    )


def test_expand_head_indices():
    heads = torch.LongTensor([1, 3, 2, 0])
    hidden_size_per_head = 2
    assert expand_head_indices(heads, hidden_size_per_head).tolist() == [2, 3, 6, 7, 4, 5, 0, 1]


def test_megatron_self_attention_head_sorting(distributed_setup_size_1):
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=1,
        hidden_size=16,
        num_attention_heads=8,
        num_query_groups=2,
        ffn_hidden_size=16,
        activation_func="squared_relu",
    )

    model = mtn.convert(model, "mcore_minitron")

    self_attn = model.decoder.layers[0].self_attention
    assert isinstance(self_attn, _DynamicSelfAttention)
    assert isinstance(self_attn.linear_qkv, _DynamicQKVColumnParallelLinear)
    assert isinstance(self_attn.linear_proj, _DynamicProjRowParallelLinear)

    hp_num_heads_per_group = self_attn.get_hparam("num_heads_per_group")
    hp_num_query_groups = self_attn.get_hparam("num_query_groups")

    assert hp_num_heads_per_group.choices == [1, 2, 3, 4]
    assert hp_num_query_groups.choices == [1, 2]

    # Set importance and slice order
    hp_num_heads_per_group._get_importance = lambda: torch.tensor(
        [2.2, 0.1, 1.1, 2.1, 3.0, 2.0, 0.0, 1.0]
    )
    hp_num_query_groups._get_importance = lambda: torch.tensor([0.0, 3.0])
    hp_num_heads_per_group.enforce_order(
        torch.argsort(hp_num_heads_per_group.importance, descending=True)
    )
    hp_num_query_groups.enforce_order(
        torch.argsort(hp_num_query_groups.importance, descending=True)
    )
    assert hp_num_heads_per_group._slice_order.tolist() == [4, 0, 3, 5, 2, 7, 1, 6]
    assert hp_num_query_groups._slice_order.tolist() == [1, 0]

    # check if we get correct selection of sorted + pruned heads after setting active values
    hp_num_heads_per_group.active = 2  # top 2 query and their k,v heads per group (based on imp)
    hp_num_query_groups.active = 2  # top 2 query groups

    expected_q_heads = [4, 5, 0, 3]
    expected_qkv_heads = [6, 7, 10, 11, 0, 3, 4, 5]  # 4 heads / group -> 6 qkv heads / group
    assert (
        self_attn.linear_qkv._get_output_size_indices().tolist()
        == expand_head_indices(
            torch.LongTensor(expected_qkv_heads), model.config.kv_channels
        ).tolist()
    )
    assert (
        self_attn.linear_proj._get_input_size_indices().tolist()
        == expand_head_indices(
            torch.LongTensor(expected_q_heads), model.config.kv_channels
        ).tolist()
    )

    # Clean up since this is not a spawned process
    destroy_model_parallel()
