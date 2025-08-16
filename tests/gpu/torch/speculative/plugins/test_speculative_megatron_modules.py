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
from collections import deque
from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron(apex_or_te_required=True)

from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import get_mcore_gpt_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel, right_padding
from modelopt.torch.speculative.plugins.megatron_medusa import _DynamicMedusaGPTModel
from modelopt.torch.speculative.utils import Tree, get_default_attention_mask_and_position_ids


def _test_speculative_gpt_model(
    algo, num_medusa_heads_or_eagle_layers, activation_func, normalization, rank, size
):
    num_attention_heads = 8
    num_query_groups = size
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 2

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
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

    # Bfloat16
    model = model.to(torch.bfloat16)

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

        assert eagle_loss.shape[0] == batch_size
        assert eagle_loss.shape[1] == max_sequence_length


@pytest.mark.parametrize(
    ("algo", "num_medusa_heads_or_eagle_layers", "activation_func", "normalization"),
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


def generate_next_tokens(model, eagle_ids, hidden_states, topk=1):
    padded_eagle_ids, seq_len, padded_hidden_states = right_padding(eagle_ids, hidden_states)
    eagle_attention_mask, eagle_position_ids = get_default_attention_mask_and_position_ids(
        padded_eagle_ids
    )

    eagle_inputs = {}
    eagle_inputs["input_ids"] = padded_eagle_ids
    eagle_inputs["embedding"] = model.embedding(
        input_ids=padded_eagle_ids,
        position_ids=eagle_position_ids,
    )
    eagle_inputs["hidden_states"] = padded_hidden_states
    eagle_inputs["attention_mask"] = eagle_attention_mask

    eagle_inputs["rotary_pos_emb"] = None

    _, eagle_logits, eagle_next_hidden_states_input = model._eagle_forward(eagle_inputs, None)

    eagle_logits = eagle_logits[seq_len - 1 : seq_len, :, :]
    eagle_next_hidden_states_input = eagle_next_hidden_states_input[seq_len - 1 : seq_len, :, :]

    draft_token = (
        gather_from_tensor_model_parallel_region(eagle_logits).topk(topk, dim=-1)[1].transpose(0, 1)
    )
    return draft_token, eagle_next_hidden_states_input


def _test_tree_decode(tree_paths, greedy_steps, rank, size):
    activation_func = "squared_relu"
    normalization = "RMSNorm"

    num_attention_heads = 8
    num_query_groups = size
    max_sequence_length = 32
    vocab_size = 64
    batch_size = 1

    config = {"eagle_num_layers": 1}

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func=activation_func,
        normalization=normalization,
    ).cuda()

    model = mtsp.convert(model, [("eagle", config)])

    # Bfloat16
    model = model.to(torch.bfloat16)

    # Prepare inputs for forward.
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, max_sequence_length)).cuda()
    attention_mask = torch.tril(torch.ones((1, 1, max_sequence_length, max_sequence_length))).cuda()
    position_ids = torch.arange(max_sequence_length, dtype=torch.long).unsqueeze(0).cuda()
    attention_mask = attention_mask < 0.5

    model.eval()
    tree = Tree(tree_paths)

    input_id, draft_tokens, pred_tokens = model.tree_decode(prompt_tokens, tree=tree)

    # check for empty tree paths
    if not tree_paths:
        assert draft_tokens is None, "draft_tokens should be None for empty tree paths"
        return

    # check when tree decode is same as greedy decode
    if greedy_steps:
        spec_input_id, spec_draft_tokens = model.pseudo_speculative_generate(
            prompt_tokens, steps=greedy_steps
        )
        assert (pred_tokens == spec_draft_tokens[0]).all(), (
            f"pred_tokens should be equal to spec_draft_tokens, {pred_tokens} != {spec_draft_tokens[0]}"
        )
        assert input_id == spec_input_id[0], (
            f"spec_input_id should be equal to input_id, {input_id} != {spec_input_id[0]}"
        )
        return

    orig_hidden_states, _ = model._base_model_forward(
        prompt_tokens,
        position_ids,
        attention_mask,
    )

    # Get Eagle-specific input hidden states
    eagle_hidden_states = model._get_eagle_input_hidden_states(orig_hidden_states)
    # Extract tokens for Eagle processing (excluding first token)
    eagle_tokens = prompt_tokens[:, 1:]

    # Initialize lists to store draft tokens and hidden states
    draft_tokens_list = [input_id]
    eagle_hidden_states_list = [eagle_hidden_states]
    # Track indices for token and hidden state mapping
    index_list = [[[0, 0]]]

    # Initialize queue for breadth-first tree traversal
    queue = deque([(draft_tokens, 0)])

    # Process tree nodes in breadth-first order
    while queue:
        tree_token, index = queue.popleft()
        if not tree_token.children:
            continue
        # Collect tokens and hidden states for current node
        tokens = []
        hidden_states = []
        for token_idx, state_idx in index_list[index]:
            tokens.append(draft_tokens_list[token_idx])
            hidden_states.append(eagle_hidden_states_list[state_idx])
        # Concatenate tokens and hidden states for processing
        tokens = torch.cat([eagle_tokens, torch.cat(tokens, dim=-1)], dim=-1)
        hidden_states = torch.cat(hidden_states, dim=0)
        # Generate next token and get updated hidden states
        draft_token, eagle_next_hidden_states_input = generate_next_tokens(
            model, tokens, hidden_states, topk=len(tree_token.children)
        )
        # Verify generated tokens match expected tree structure
        for child_idx, tree_node in enumerate(tree_token.children.values()):
            assert tree_node.value[0] == draft_token[0, 0, child_idx], (
                f"token mismatch at {tree_node.value[0]} != {draft_token[0, 0, child_idx]}"
            )
        # Update tracking variables
        cur_len = len(draft_tokens_list)
        eagle_hidden_states_list.append(eagle_next_hidden_states_input)
        # Process children and add them to the queue
        for child_idx, child_tree_token in enumerate(tree_token.children.values()):
            queue.append([child_tree_token, len(index_list)])
            draft_tokens_list.append(draft_token[:, :, child_idx])
            index_list.append(
                index_list[index][:] + [[cur_len + child_idx, len(eagle_hidden_states_list) - 1]]
            )


@pytest.mark.parametrize(
    ("greedy_steps", "tree_paths"),
    [
        (None, []),
        (3, [[0], [0, 0], [0, 0, 0]]),
        (
            None,
            [
                [0],
                [1],
                [0, 0],
                [0, 1],
                [1, 1],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0],
                [0, 0, 1, 0],
            ],
        ),
    ],
)
def test_tree_decode_model(greedy_steps, tree_paths):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_tree_decode, tree_paths, greedy_steps),
        backend="nccl",
    )
