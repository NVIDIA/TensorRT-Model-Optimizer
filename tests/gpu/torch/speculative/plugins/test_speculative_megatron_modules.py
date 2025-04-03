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
)

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.megatron import (
    _DynamicEagleGPTModel,
    _DynamicMedusaGPTModel,
)


def _test_speculative_gpt_model(
    algo, num_medusa_heads_or_eagle_layers, activation_func, normalization, rank, size
):
    num_attention_heads = 8
    num_query_groups = size
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    ).cuda()

    if algo == "medusa":
        config = {
            "medusa_num_heads": num_medusa_heads_or_eagle_layers,
            "medusa_num_layers": 1,
        }

        model = mtsp.convert(model, [("medusa", config)])

        # Type checking
        assert isinstance(model, _DynamicMedusaGPTModel)
    elif algo == "eagle":
        config = {"eagle_num_layers": 1}

        model = mtsp.convert(model, [("eagle", config)])

        # Type checking
        assert isinstance(model, _DynamicEagleGPTModel)
    else:
        raise ValueError("Only algo={eagle, medusa} are supported!")

    # Prepare inputs for forward.
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    attention_mask = torch.tril(torch.ones((1, 1, max_sequence_length, max_sequence_length))).cuda()
    position_ids = torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(0).cuda()
    attention_mask = attention_mask < 0.5

    # When no labels provided, model.forward should return logits[b, s, vocab / tp]
    logits = model(prompt_tokens, position_ids, attention_mask, labels=None)
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == max_sequence_length
    assert logits.shape[2] == vocab_size / size

    if algo == "medusa":
        # When label provided, model.forward should return
        # medusa_loss[b, s * (num_medusa_heads + 1), b]
        labels = torch.randint(
            0,
            vocab_size,
            (batch_size, max_sequence_length),
        ).cuda()
        medusa_loss = model(prompt_tokens, position_ids, attention_mask, labels=labels)

        assert medusa_loss.shape[0] == batch_size
        assert medusa_loss.shape[1] == max_sequence_length
    elif algo == "eagle":
        labels = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
        eagle_loss = model(prompt_tokens, position_ids, attention_mask, labels=labels)

        print(eagle_loss.shape)
        assert eagle_loss.shape[0] == batch_size
        assert eagle_loss.shape[1] == max_sequence_length


@pytest.mark.parametrize(
    "algo,num_medusa_heads_or_eagle_layers,activation_func,normalization",
    [
        ("eagle", 1, "squared_relu", "LayerNorm"),  # MHA
        ("eagle", 2, "swiglu", "RMSNorm"),  # GQA
        ("medusa", 1, "squared_relu", "LayerNorm"),  # MHA
        ("medusa", 2, "swiglu", "RMSNorm"),  # GQA
    ],
)
def test_speculative_gpt_model(
    algo, num_medusa_heads_or_eagle_layers, activation_func, normalization
):
    if algo == "eagle":
        try:
            import megatron.core.post_training  # noqa: F401
        except ImportError:
            pytest.skip("megatron.core.post_training not found")

    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_speculative_gpt_model,
            algo,
            num_medusa_heads_or_eagle_layers,
            activation_func,
            normalization,
        ),
        backend="nccl",
    )
