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

"""Utils for speculative decoding."""

import copy
from collections import Counter

import torch
import torch.distributed
from torch import nn

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def calibrate_frequent_vocab(tokenizer, text, target_vocab_size, output_file=None):
    """Given a calibration text, find the most common vocabs and return the mapping."""
    conversations = tokenizer.apply_chat_template(text)
    counter = Counter(conversations)
    vocab = counter.most_common(target_vocab_size)
    mapping = torch.zeros(target_vocab_size, dtype=torch.int64)
    for i in range(target_vocab_size):
        idx = vocab[i][0]
        mapping[i] = idx - i
    if output_file is not None:
        torch.save(mapping, output_file)
    return mapping


def get_default_attention_mask_and_position_ids(input_ids: torch.Tensor):
    """Compute default attention_mask ans position_ids given input_ids."""
    seq_len = input_ids.shape[-1]

    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    attention_mask = (
        torch.triu(
            torch.ones(
                (input_ids.shape[0], seq_len, seq_len),
                device=input_ids.device,
            ),
            diagonal=1,
        )
        .bool()
        .view(input_ids.shape[0], 1, seq_len, seq_len)
    )

    return attention_mask, position_ids


def tree_decode(draft_logits: list[torch.Tensor], tree: list[list[int]]):
    """Decode tokens using the tree.

    Args:
        draft_logits: a list of logits. Each logit represent a future position.
        tree: a tree for decoding. Each sublist is a branch from root where the number
        represents the topk index.
    """
    draft_tokens = []
    for seq in tree:
        tokens = []
        for i, index in enumerate(seq):
            token = draft_logits[i][:, -1].topk(index + 1, dim=-1).indices[:, -1:]
            tokens.append(token)
        draft_tokens.append(torch.cat(tokens, dim=-1))
    return draft_tokens


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """

    def __init__(self, hidden_size: int, bias: bool = True):
        """Init function of ResBlock.

        Args:
            hidden_size: The size of the hidden layers in the block.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock.

        Args:
            x: Input tensor.

        Returns:
            Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class AcceptanceRateValidation:
    """Base acceptance rate (AR) validation class.

    This class is used to validate the AR within ModelOpt.
    self.validate is the main function to validate the AR given a prompt or input_ids.
    Note: currently it only supports TP.
    """

    def __init__(self, model, tokenizer):
        """Init function to take in the model and tokenizer."""
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # Make sure the model is in eval mode
        self.model.eval()

    def tokenize(self, prompt):
        """Apply chat template to the prompt and get input_ids."""
        conversation = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self.tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        ).to(
            torch.cuda.current_device(),
        )
        input_ids = output.input_ids
        return input_ids

    def get_ground_truth(self, input_ids: torch.Tensor, osl: int):
        """This function returns ground truth token ids from the base model.

        This function will be implemented in the plugins.

        Args:
            input_ids: the token ids of the input
            osl: output sequence length
        """

    def check_draft(self, ground_truth, input_ids, draft_tokens, tree=None):
        """This function checks if the draft tokens should be accepted (same as ground truth).

        If tree is None, it is eager mode.
        """
        if draft_tokens is None:
            return input_ids

        if tree is None:
            # eager mode
            for i in range(draft_tokens.shape[-1]):
                input_id = draft_tokens[:, i : i + 1]
                if ground_truth[:, input_ids.shape[1] : input_ids.shape[1] + 1] == input_id:
                    input_ids = torch.cat((input_ids, input_id), dim=-1)
                    if input_ids.shape[1] == ground_truth.shape[1]:
                        break
                else:
                    break
        else:
            # tree decoding
            pass

        return input_ids

    def check_data_consistancy_across_ranks(self, data, group=None):
        """This function checks the data consistancy across all ranks in the group.

        Use rank 0 data as the golden set to broadcast to all ranks.
        Each rank will then compare to this data and through error if different.
        """
        golden_set = copy.deepcopy(data)
        torch.distributed.broadcast(data, src=0, group=group)
        if not torch.equal(data, golden_set):
            raise ValueError(
                "Data diverges across ranks. For Megatron, 'moe-token-dispatcher-type'"
                "should set to 'alltoall'."
            )

    def validate(
        self,
        osl,
        prompt=None,
        input_ids=None,
        ground_truth=None,
        tree=None,
        steps=1,
    ):
        """This function validate the AR of the model given the input sequence."""
        if input_ids is None:
            input_ids = self.tokenize(prompt)

        isl = input_ids.shape[1]

        if ground_truth is None:
            ground_truth = self.get_ground_truth(input_ids, osl)
        self.check_data_consistancy_across_ranks(ground_truth)

        cnt = 0
        draft_tokens = None
        while input_ids.shape[1] < ground_truth.shape[1]:
            cnt += 1
            input_ids = self.check_draft(ground_truth, input_ids, draft_tokens, tree)
            if input_ids.shape[1] == ground_truth.shape[1]:
                break
            input_id, draft_tokens = self.model.pseudo_speculative_generate(input_ids, steps=steps)
            self.check_data_consistancy_across_ranks(input_id)
            self.check_data_consistancy_across_ranks(draft_tokens)
            input_ids = torch.cat((input_ids, input_id), dim=-1)

        ar = (ground_truth.shape[1] - isl) / cnt

        return ground_truth, ar
