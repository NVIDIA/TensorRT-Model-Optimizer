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


import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True, mamba_required=True)

from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import (
    get_mcore_mamba_model,
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)
from _test_utils.torch_misc import set_seed
from megatron.core.parallel_state import is_pipeline_first_stage, is_pipeline_last_stage
from megatron.core.transformer.identity_op import IdentityOp

import modelopt.torch.nas as mtn
from modelopt.torch.nas.modules.conv import _DynamicConvNd
from modelopt.torch.nas.plugins.megatron import (
    MambaDInnerHp,
    MambaNumHeadsHp,
    _DynamicColumnParallelLinear,
    _DynamicExtendedRMSNorm,
    _DynamicLayerNorm,
    _DynamicMambaLayer,
    _DynamicMambaMixer,
    _DynamicMCoreLanguageModel,
    _DynamicRowParallelLinear,
    _DynamicVocabParallelEmbedding,
)
from modelopt.torch.nas.traced_hp import TracedHp
from modelopt.torch.opt.utils import named_dynamic_modules, search_space_size
from modelopt.torch.prune.plugins.mcore_minitron import _convert_model_to_dynamic_space
from modelopt.torch.utils import flatten_tree
from modelopt.torch.utils.random import centroid

SEED = 1234


def _test_mamba_search_space(rank, size):
    channel_divisor = 64
    mamba_num_heads_divisor = 4
    mamba_head_dim_divisor = 4

    num_layers = size
    hybrid_override_pattern = "M" * size
    hidden_size = 256
    mamba_state_dim = 64
    mamba_head_dim = 16
    mamba_num_groups = 2
    max_sequence_length = 16
    vocab_size = 32
    batch_size = 2

    model = get_mcore_mamba_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
    )
    mamba_num_heads = model.decoder.layers[0].mixer.nheads

    model = mtn.convert(model, "mcore_minitron")

    assert isinstance(model, _DynamicMCoreLanguageModel)
    if is_pipeline_first_stage():
        assert isinstance(model.embedding.word_embeddings, _DynamicVocabParallelEmbedding)
    for layer in model.decoder.layers:
        assert isinstance(layer, _DynamicMambaLayer)
        assert isinstance(layer.mixer, _DynamicMambaMixer)
        assert isinstance(layer.mixer.in_proj, _DynamicColumnParallelLinear)
        assert isinstance(layer.mixer.out_proj, _DynamicRowParallelLinear)
        assert isinstance(layer.mixer.conv1d, _DynamicConvNd)
        if layer.mixer.rmsnorm:
            assert isinstance(layer.mixer.norm, _DynamicExtendedRMSNorm)
    if is_pipeline_last_stage():
        assert isinstance(model.decoder.final_norm, _DynamicLayerNorm)
        assert isinstance(model.output_layer, _DynamicColumnParallelLinear)

    # NOTE: `search_space_size` does not reduce across TP/PP groups
    ss_size_per_pp = search_space_size(model)
    num_heads_choices = mamba_num_heads // mamba_num_heads_divisor
    head_dim_choices = mamba_head_dim // mamba_head_dim_divisor
    hidden_size_choices = hidden_size // channel_divisor
    num_layers_per_pp = num_layers // size
    assert (
        ss_size_per_pp
        == (num_heads_choices * head_dim_choices) ** num_layers_per_pp
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


def test_mamba_search_space():
    spawn_multiprocess_job(
        size=torch.cuda.device_count(), job=_test_mamba_search_space, backend="nccl"
    )


def _test_mamba_parameter_sorting(rank, size):
    num_layers = size
    hybrid_override_pattern = "M" * size
    hidden_size = 256
    mamba_state_dim = 64
    mamba_head_dim = 16
    mamba_num_groups = 2
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    model = get_mcore_mamba_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hybrid_override_pattern=hybrid_override_pattern,
        hidden_size=hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        bf16=False,
    )

    # Randomize norm weights instead of all zeros or ones
    for n, m in model.named_modules():
        if "norm" in n and not isinstance(m, IdentityOp):
            m.weight.data = torch.randn_like(m.weight)

    model.eval()
    dynamic_space = _convert_model_to_dynamic_space(model)

    # Compute activations for sorting
    for _ in range(5):
        run_mcore_inference_with_dummy_input(model, batch_size)

    # Get the output of the original model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    y1 = run_mcore_inference(model, prompt_tokens)

    dynamic_space.sort_parameters()

    # check if all mamba_num_heads, mamba_head_dim, hidden_size have been sorted
    sortable_per_pp = [
        n for n, hp in dynamic_space.named_hparams(configurable=True) if hp.importance is not None
    ]
    # 2 mamba hps per layer + 1 for hidden_size (num_layers is not sorted!)
    assert len(sortable_per_pp) == 2 * num_layers // size + 1

    # sanity check if the model functionality is preserved after sorting
    y2 = run_mcore_inference(model, prompt_tokens)

    # check if the inference results after sorting is the same
    assert all(
        torch.allclose(t1, t2, rtol=1e-5, atol=1e-3)
        for t1, t2 in zip(flatten_tree(y1)[0], flatten_tree(y2)[0])
    )


def test_mamba_parameter_sorting(need_2_gpus):
    set_seed(SEED)
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=_test_mamba_parameter_sorting,
        backend="nccl",
    )


def test_mamba_num_heads_hp():
    num_heads = MambaNumHeadsHp([2, 4, 6, 8], ngroups=2)  # 4 heads per group
    assert num_heads.choices == [2, 4, 6, 8]
    assert num_heads.active_slice == slice(8)

    num_heads.active = 4  # 2 heads per group
    assert num_heads.active_slice.tolist() == [0, 1, 4, 5]

    num_heads_ranking = torch.tensor([1, 0, 3, 2, 4, 7, 6, 5])
    num_heads_ranking.argsort = lambda *args, **kwargs: num_heads_ranking
    num_heads._get_importance = lambda: num_heads_ranking
    num_heads.enforce_order(num_heads.importance.argsort(descending=True))
    assert num_heads.active_slice.tolist() == [1, 0, 4, 7]


def test_mamba_d_inner_hp():
    num_heads = TracedHp([2, 4, 6, 8])
    head_dim = TracedHp([1, 2, 3])
    d_inner = MambaDInnerHp(num_heads, head_dim)

    assert d_inner.choices == [2, 4, 6, 8, 12, 16, 18, 24]
    assert d_inner.active_slice == slice(24)

    # Set importance and slice order
    num_heads._get_importance = lambda: torch.tensor([2.2, 0.1, 1.1, 2.1, 3.0, 2.0, 0.0, 1.0])
    head_dim._get_importance = lambda: torch.tensor([2.0, 3.0, 1.0])
    num_heads.enforce_order(torch.argsort(num_heads.importance, descending=True))
    head_dim.enforce_order(torch.argsort(head_dim.importance, descending=True))
    assert num_heads.active_slice.tolist() == [4, 0, 3, 5, 2, 7, 1, 6]
    assert head_dim.active_slice.tolist() == [1, 0, 2]

    # check if we get correct selection of sorted + pruned heads after setting active values
    num_heads.active = 6  # top 6 heads
    head_dim.active = 2  # top 2 dims per head
    assert d_inner.active == 12  # (6 * 2)
    assert d_inner.active_slice.tolist() == [13, 12, 1, 0, 10, 9, 16, 15, 7, 6, 22, 21]
