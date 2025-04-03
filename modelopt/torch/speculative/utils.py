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

import torch
from torch import nn

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def get_default_attention_mask_and_position_ids(input_ids: torch.Tensor):
    """Compute default attention_mask ans position_ids given input_ids."""
    seq_len = input_ids.shape[-1]

    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    attention_mask = torch.tril(
        torch.ones(
            (input_ids.shape[0], seq_len, seq_len),
            device=input_ids.device,
        ),
    ).view(input_ids.shape[0], 1, seq_len, seq_len)
    attention_mask = attention_mask < 0.5

    return attention_mask, position_ids


def right_padding(input_ids: torch.Tensor, tp: int, hidden_states: torch.Tensor = None):
    """Pad zeros to the right so that the padded_input_ids is a multiple of tp."""
    seq_len = input_ids.shape[-1]
    right_padding_len = 0 if seq_len % tp == 0 else (tp - seq_len % tp)

    if right_padding_len > 0:
        right_token_pad = torch.zeros(
            (input_ids.shape[0], right_padding_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        padded_input_ids = torch.cat((input_ids, right_token_pad), dim=-1)
        if hidden_states is not None:
            padding_zeros = torch.zeros(
                (right_padding_len, hidden_states.shape[1], hidden_states.shape[2]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            padded_hidden_states = torch.cat((hidden_states, padding_zeros), dim=0)
    else:
        padded_input_ids = input_ids
        padded_hidden_states = hidden_states

    if hidden_states is not None:
        return padded_input_ids, seq_len, padded_hidden_states
    else:
        return padded_input_ids, seq_len


def tree_decode(draft_logits, tree):
    """Decode tokens using the tree.

    Args:
        draft_logits (List[torch.Tensor]): a list of logits. Each logit represent a future position.
        tree (List[List[int]]): a tree for decoding. Each sublist is a branch from root where the number
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

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, bias=True):
        """Init function of ResBlock.

        Args:
        hidden_size (int): The size of the hidden layers in the block.
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class AcceptanceRateValidation:
    """Base acceptance rate (AR) validation class.

    This class is used to validate the AR within ModelOpt.
    self.validate is the main function to validate the AR given a prompt or input_ids.
    Note: currently it only supports TP.
    """

    def __init__(self, model, tokenizer, tp):
        """Init function to take in the model and tokenizer."""
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
        self.model = model
        self.tokenizer = tokenizer
        self.end_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        self.tp = tp

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

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth token ids from the base model.

        This function will be implemented in the plugins.

        Args:
        input_ids (torch.Tensor): the token ids of the input
        attention_mask (torch.Tensor): attention mask of the input
        osl (int): output sequence length
        """
        pass

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

        cnt = 0
        draft_tokens = None
        while input_ids.shape[1] < ground_truth.shape[1]:
            cnt += 1
            input_ids = self.check_draft(ground_truth, input_ids, draft_tokens, tree)
            if input_ids.shape[1] == ground_truth.shape[1]:
                break
            input_id, draft_tokens = self.model.pseudo_speculative_generate(
                input_ids, tp=self.tp, steps=steps
            )
            input_ids = torch.cat((input_ids, input_id), dim=-1)

        ar = (ground_truth.shape[1] - isl) / cnt

        return ground_truth, ar
