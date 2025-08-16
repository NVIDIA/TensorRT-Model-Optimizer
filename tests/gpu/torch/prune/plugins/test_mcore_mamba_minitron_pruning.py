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
    run_mcore_inference_with_dummy_input,
)
from megatron.core.ssm.mamba_layer import MambaLayer

import modelopt.torch.prune as mtp


def _test_mcore_mamba_pruning(rank, size):
    num_layers = min(size * 2, 8)
    hidden_size = 256
    ffn_hidden_size = 128
    num_attention_heads = 8
    num_query_groups = 4
    mamba_state_dim = 64
    mamba_head_dim = 16
    mamba_num_groups = 2
    batch_size = 2

    model = get_mcore_mamba_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=size,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
    )

    mamba_num_heads = torch.tensor(0, device=torch.cuda.current_device())
    if rank == 0:
        assert isinstance(model.decoder.layers[0], MambaLayer)
        mamba_num_heads += model.decoder.layers[0].mixer.nheads
    torch.distributed.broadcast(mamba_num_heads, 0, async_op=True)
    mamba_num_heads = mamba_num_heads.item()
    assert mamba_num_heads > 0, "No MambaLayer found in the model rank 0!"

    def forward_loop(m):
        for _ in range(5):
            run_mcore_inference_with_dummy_input(m, batch_size, hidden_size)

    # Traditional GPT pruning parameters
    pruned_ffn_hidden_size = ffn_hidden_size // 2
    pruned_num_attention_heads = num_attention_heads // 2
    pruned_num_query_groups = num_query_groups // 2
    pruned_hidden_size = hidden_size // 2
    pruned_num_layers = num_layers // 2

    # Mamba-specific pruning parameters
    # pruned_mamba_num_heads = mamba_num_heads // 2
    # pruned_mamba_head_dim = mamba_head_dim // 2

    # Base export config with GPT/Attention parameters
    # TODO: enable mamba head pruning after debugging
    export_config = {
        "ffn_hidden_size": pruned_ffn_hidden_size,
        "num_attention_heads": pruned_num_attention_heads,
        "num_query_groups": pruned_num_query_groups,
        "hidden_size": pruned_hidden_size,
        # "mamba_num_heads": pruned_mamba_num_heads,
        # "mamba_head_dim": pruned_mamba_head_dim,
        "num_layers": pruned_num_layers,
    }
    model, _ = mtp.prune(
        model,
        mode="mcore_gpt_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,  # Not used
        config={"forward_loop": forward_loop},
    )

    # Assert forward pass works on the pruned model
    run_mcore_inference_with_dummy_input(model, batch_size, pruned_hidden_size)

    # Assert model.config is updated for correct save/restoring
    assert model.config.ffn_hidden_size == pruned_ffn_hidden_size
    assert model.config.num_attention_heads == pruned_num_attention_heads
    assert model.config.num_query_groups == pruned_num_query_groups
    assert model.config.hidden_size == pruned_hidden_size
    assert model.config.num_layers == pruned_num_layers
    # assert model.config.mamba_num_heads == pruned_mamba_num_heads
    # assert model.config.mamba_head_dim == pruned_mamba_head_dim


def test_mcore_mamba_pruning():
    spawn_multiprocess_job(
        size=torch.cuda.device_count(), job=_test_mcore_mamba_pruning, backend="nccl"
    )
