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

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)

import modelopt.torch.prune as mtp


def _test_mcore_gpt_pruning(
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    pruned_ffn_div,
    pruned_num_attention_heads_div,
    pruned_num_query_groups_div,
    pruned_hidden_size_div,
    pruned_num_layers_div,
    uneven_pp,
    position_embedding_type,
    skip_sorting,
    ckpt_path,
    rank,
    size,
):
    hidden_size = 256
    ffn_hidden_size = 256
    max_sequence_length = 16
    vocab_size = 64
    batch_size = 2

    num_layers = min(size * 2, 8)
    num_layers_in_first_pipeline_stage = None
    num_layers_in_last_pipeline_stage = None
    if uneven_pp and size > 1:
        num_layers = size * 2
        if size == 2:  # [1, 3]
            num_layers_in_first_pipeline_stage = 1
        elif size == 4:  # [3, 2, 2, 1]
            num_layers_in_first_pipeline_stage = 3
            num_layers_in_last_pipeline_stage = 1
        elif size == 8:  # [4, 1, 1, 1, 1, 1, 1, 6]
            num_layers_in_first_pipeline_stage = 4
            num_layers_in_last_pipeline_stage = 6
        else:
            raise ValueError(f"Unsupported size {size}")

    def _get_model(initialize_megatron=True):
        model = get_mcore_gpt_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            ffn_hidden_size=ffn_hidden_size,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            position_embedding_type=position_embedding_type,
            activation_func=activation_func,
            normalization=normalization,
            num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
            use_cpu_initialization=True,  # Ensure deterministic weight init across CUDA versions
        ).cuda()
        return model

    model = _get_model()

    sd = model.state_dict()

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

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

    config = {
        "scores_path": ckpt_path,
        "skip_sorting": skip_sorting,
    }
    if skip_sorting:
        assert ckpt_path is None
    else:
        config["forward_loop"] = forward_loop
    model, pruning_scores = mtp.prune(
        model,
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config=config,
    )
    if not skip_sorting:
        assert pruning_scores["layer_scores"]
        assert pruning_scores["activations_per_rank"]

        # TODO: Simplify it: this unit test is too long,
        # hard to read (the same set of assertions across different test cases with if-else).

        assert len(pruning_scores["activations_per_rank"]) == 1
        rank_0_activations = pruning_scores["activations_per_rank"][0]

        # Test case 1: MHA - pruned ffn/4 (num_attention_heads=8, num_query_groups=8, ffn_div=4)
        if pruned_ffn_div == 4:
            # Layer scores
            if rank == 0:
                print("\n=== TEST CASE 1 ===")
                print(f"layer_scores[1] = {pruning_scores['layer_scores'][1]:.16f}")
                print(f"layer_scores[2] = {pruning_scores['layer_scores'][2]:.16f}")
            assert pruning_scores["layer_scores"][1] == pytest.approx(2.0868452191352844, abs=1e-3)
            assert pruning_scores["layer_scores"][2] == pytest.approx(1.7638601660728455, abs=1e-3)

            # Validate decoder.layers.0.mlp activations
            mlp_0_acts = rank_0_activations["decoder.layers.0.mlp"]
            if rank == 0:
                print(f"mlp_0_acts.min() = {mlp_0_acts.min().item():.16f}")
                print(f"mlp_0_acts.max() = {mlp_0_acts.max().item():.16f}")
                print(f"mlp_0_acts.mean() = {mlp_0_acts.mean().item():.16f}")
            assert mlp_0_acts.min().item() == pytest.approx(0.0015609927941114, abs=1e-3)
            assert mlp_0_acts.max().item() == pytest.approx(0.3844809532165527, abs=1e-3)
            assert mlp_0_acts.mean().item() == pytest.approx(0.0629318505525589, abs=1e-3)

            # Validate decoder.layers.1.mlp activations
            mlp_1_acts = rank_0_activations["decoder.layers.1.mlp"]
            if rank == 0:
                print(f"mlp_1_acts.min() = {mlp_1_acts.min().item():.16f}")
                print(f"mlp_1_acts.max() = {mlp_1_acts.max().item():.16f}")
                print(f"mlp_1_acts.mean() = {mlp_1_acts.mean().item():.16f}")
                print("=" * 50 + "\n")
            assert mlp_1_acts.min().item() == pytest.approx(0.0001484956446802, abs=1e-3)
            assert mlp_1_acts.max().item() == pytest.approx(0.7835369110107422, abs=1e-3)
            assert mlp_1_acts.mean().item() == pytest.approx(0.0926810950040817, abs=1e-3)

        # Test case 2: GQA - pruned attention/2 (num_attention_heads=8, num_query_groups=4, attention_div=2)
        elif pruned_num_attention_heads_div == 2 and pruned_ffn_div == 1:
            # Layer scores
            assert pruning_scores["layer_scores"][1] == pytest.approx(2.1415508985519409, abs=1e-3)
            assert pruning_scores["layer_scores"][2] == pytest.approx(1.7198008894920349, abs=1e-3)

            # Validate decoder.layers.0.self_attention activations
            assert "decoder.layers.0.self_attention" in rank_0_activations
            attn_0_acts = rank_0_activations["decoder.layers.0.self_attention"]
            assert attn_0_acts.shape == torch.Size([256])
            assert attn_0_acts.min().item() == pytest.approx(0.0409194342792034, abs=1e-3)
            assert attn_0_acts.max().item() == pytest.approx(0.5261313319206238, abs=1e-3)
            assert attn_0_acts.mean().item() == pytest.approx(0.1613342612981796, abs=1e-3)

            # Validate decoder.layers.1.self_attention activations
            assert "decoder.layers.1.self_attention" in rank_0_activations
            attn_1_acts = rank_0_activations["decoder.layers.1.self_attention"]
            assert attn_1_acts.shape == torch.Size([256])
            assert attn_1_acts.min().item() == pytest.approx(0.1189328655600548, abs=1e-3)
            assert attn_1_acts.max().item() == pytest.approx(1.3832759857177734, abs=1e-3)
            assert attn_1_acts.mean().item() == pytest.approx(0.4782669544219971, abs=1e-3)

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

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.num_query_groups == pruned_num_query_groups
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.num_layers == pruned_num_layers

    # Assert forward pass works on the pruned model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    output = run_mcore_inference(model, prompt_tokens, pruned_hidden_size)

    # Assert re-pruning from scores_path works without running the forward loop again
    if ckpt_path:
        model_rerun = _get_model(initialize_megatron=False)
        model_rerun.load_state_dict(sd)
        mtp.prune(
            model_rerun,
            mode="mcore_minitron",
            constraints={"export_config": export_config},
            dummy_input=None,  # Not used
            config={"scores_path": ckpt_path},
        )

        output_rerun = run_mcore_inference(model_rerun, prompt_tokens, pruned_hidden_size)
        assert torch.allclose(output, output_rerun, atol=1e-5)


@pytest.mark.parametrize(
    (
        "num_attention_heads",
        "num_query_groups",
        "activation_func",
        "normalization",
        "ffn_div",
        "num_attention_heads_div",
        "num_query_groups_div",
        "hidden_size_div",
        "num_layers_div",
        "uneven_pp",
        "position_embedding_type",
        "skip_sorting",
        "test_ckpt",
    ),
    [
        # MHA - pruned ffn/4
        (8, 8, "squared_relu", "LayerNorm", 4, 1, 1, 1, 1, False, "rope", False, False),
        # GQA - pruned attention/2
        (8, 4, "squared_relu", "RMSNorm", 1, 2, 2, 1, 1, False, "rope", False, False),
        # GQA - pruned hidden_size/4
        (8, 4, "swiglu", "RMSNorm", 1, 1, 1, 4, 1, False, "rope", True, False),
        # MHA - pruned num_layers/2
        (8, 8, "swiglu", "LayerNorm", 1, 1, 1, 1, 2, False, "rope", False, False),
        # GQA - pruned all/2, uneven pp
        (8, 4, "swiglu", "RMSNorm", 2, 2, 2, 2, 2, True, "yarn", False, True),
    ],
)
def test_mcore_gpt_pruning(
    tmp_path,
    num_attention_heads,
    num_query_groups,
    activation_func,
    normalization,
    ffn_div,
    num_attention_heads_div,
    num_query_groups_div,
    hidden_size_div,
    num_layers_div,
    uneven_pp,
    position_embedding_type,
    skip_sorting,
    test_ckpt,
):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_mcore_gpt_pruning,
            num_attention_heads,
            num_query_groups,
            activation_func,
            normalization,
            ffn_div,
            num_attention_heads_div,
            num_query_groups_div,
            hidden_size_div,
            num_layers_div,
            uneven_pp,
            position_embedding_type,
            skip_sorting,
            tmp_path / "minitron_scores.pth" if test_ckpt else None,
        ),
        backend="nccl",
    )


def _test_mcore_gpt_pruning_moe(ckpt_path, rank, size):
    num_layers = size
    hidden_size = 128
    moe_ffn_hidden_size = 128
    num_moe_experts = 4
    moe_shared_expert_intermediate_size = 256
    max_sequence_length = 16
    vocab_size = 64
    batch_size = 2

    def _get_model(initialize_megatron=True):
        model = get_mcore_gpt_model(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=size,
            initialize_megatron=initialize_megatron,
            num_layers=num_layers,
            hidden_size=hidden_size,
            max_sequence_length=max_sequence_length,
            vocab_size=vocab_size,
            activation_func="squared_relu",
            num_moe_experts=num_moe_experts,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        ).cuda()
        return model

    model = _get_model()
    sd = model.state_dict()

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    pruned_hidden_size = hidden_size // 2
    pruned_moe_ffn = moe_ffn_hidden_size // 2
    pruned_moe_shared_ffn = moe_shared_expert_intermediate_size // 2
    pruned_num_moe_experts = num_moe_experts // 2

    export_config = {
        "hidden_size": pruned_hidden_size,
        "moe_ffn_hidden_size": pruned_moe_ffn,
        "moe_shared_expert_intermediate_size": pruned_moe_shared_ffn,
        "num_moe_experts": pruned_num_moe_experts,
    }

    mtp.prune(
        model,
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"scores_path": ckpt_path, "forward_loop": forward_loop},
    )

    # Assert weights are pruned correctly
    for layer in model.decoder.layers:
        moe = layer.mlp
        assert moe.router.num_experts == pruned_num_moe_experts
        assert moe.router.expert_bias.shape == (pruned_num_moe_experts,)
        assert moe.router.weight.shape == (pruned_num_moe_experts, pruned_hidden_size)
        assert moe.experts.num_local_experts == pruned_num_moe_experts
        assert len(moe.experts.local_experts) == pruned_num_moe_experts
        for expert in moe.experts.local_experts:
            assert expert.linear_fc1.weight.shape == (pruned_moe_ffn, pruned_hidden_size)
            assert expert.linear_fc2.weight.shape == (pruned_hidden_size, pruned_moe_ffn)
        assert moe.shared_experts.linear_fc1.weight.shape == (
            pruned_moe_shared_ffn,
            pruned_hidden_size,
        )
        assert moe.shared_experts.linear_fc2.weight.shape == (
            pruned_hidden_size,
            pruned_moe_shared_ffn,
        )

    # Assert model.config is updated for correct save/restoring
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.moe_ffn_hidden_size == pruned_moe_ffn
    assert model.config.num_moe_experts == pruned_num_moe_experts
    assert model.config.moe_shared_expert_intermediate_size == pruned_moe_shared_ffn

    # Assert forward pass works on the pruned model
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    output = run_mcore_inference(model, prompt_tokens, pruned_hidden_size)

    # Assert re-pruning from scores_path works without running the forward loop again
    model_rerun = _get_model(initialize_megatron=False)
    model_rerun.load_state_dict(sd)
    mtp.prune(
        model_rerun,
        mode="mcore_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"scores_path": ckpt_path},
    )

    output_rerun = run_mcore_inference(model_rerun, prompt_tokens, pruned_hidden_size)
    assert torch.allclose(output, output_rerun, atol=1e-5)


def test_mcore_gpt_pruning_moe(tmp_path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_mcore_gpt_pruning_moe, tmp_path / "minitron_scores.pth"),
        backend="nccl",
    )
