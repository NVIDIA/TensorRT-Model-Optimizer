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
# mypy: ignore-errors

from abc import ABCMeta, abstractmethod
from typing import List, Callable, Literal, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This class is deprecated. Deci models are now hf models.")


class DummyBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This class is deprecated. Deci models are now hf models.")


RoPECache = torch.Tensor


def normalized_mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
    epsilon: float = 1e-6,
) -> Tensor:
    loss = F.mse_loss(input, target, reduction=reduction) / F.mse_loss(
        target, torch.zeros_like(target) + epsilon, reduction=reduction
    )
    return loss


def cosine_embedding_loss_batched(input: Tensor, target: Tensor) -> Tensor:
    # inputs are of shape (B,T,H)
    batch_size = input.size(0)
    input = input.view(batch_size, -1)
    target = target.view(batch_size, -1)
    target_tensor = input.new(input.size(0)).fill_(1)
    loss = F.cosine_embedding_loss(
        input1=input, input2=target, target=target_tensor, reduction="none"
    )
    return loss


def cross_entropy_probs_batched(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    return F.cross_entropy(
        logits_input.transpose(1, 2), logits_target.softmax(-1).transpose(1, 2), reduction="none"
    ).mean(-1)


def kl_div_logits_batched(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    return (
        F.kl_div(
            logits_input.log_softmax(-1),
            logits_target.log_softmax(-1),
            reduction="none",
            log_target=True,
        )
        .sum(-1)
        .mean(-1)
    )


def kl_div_single_sample(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    return F.kl_div(
        logits_input.log_softmax(-1),
        logits_target.log_softmax(-1),
        reduction="batchmean",
        log_target=True,
    )


def kl_div_logits_batched_mem_efficient(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    batch_size = logits_input.shape[0]
    kl_div_per_sample = [
        kl_div_single_sample(logits_input[i], logits_target[i]) for i in range(batch_size)
    ]
    return torch.stack(kl_div_per_sample)


kl_div = kl_div_logits_batched_mem_efficient


def mse_loss(
    x_input: torch.Tensor,
    x_target: torch.Tensor,
) -> Tensor:
    return torch.stack(
        [F.mse_loss(x_input[i_sample], x_target[i_sample]) for i_sample in range(x_input.shape[0])]
    )


def reverse_kl_div(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    return kl_div_logits_batched_mem_efficient(logits_target, logits_input)


def tv_dist(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    """
    Total Variation Distance: L1-loss between probabilities.
    vocab dimension is summed, sequence dimension is averaged.
    """
    batch_size, seq_len, vocab_size = logits_input.shape
    tv_dist_per_sample = [
        F.l1_loss(logits_input[i].softmax(-1), logits_target[i].softmax(-1), reduction="sum")
        / seq_len
        for i in range(batch_size)
    ]
    return torch.stack(tv_dist_per_sample)


def js_div(logits_input: Tensor, logits_target: Tensor) -> Tensor:
    """
    Jensen-Shannon Divergence for a single sample.
    logits: [tokens, vocab]
    target_probs: [tokens, vocab]
    """
    batch_size = logits_input.shape[0]
    _js_div = []
    for i in range(batch_size):
        input_probs = logits_input[i].softmax(-1)
        target_probs = logits_target[i].softmax(-1)
        mixture_probs = (input_probs + target_probs) * 0.5
        mixture_logprobs = mixture_probs.log().clip(min=-20)
        pred_kl_div = kl_div_single_sample(mixture_logprobs, input_probs)
        target_kl_div = kl_div_single_sample(mixture_logprobs, target_probs)
        js_div_i = 0.5 * (pred_kl_div + target_kl_div)
        _js_div.append(js_div_i)
    return torch.stack(_js_div)


LOGITS_LOSS_NAME_TO_FUNC = {name: func for name, func in globals().items() if callable(func)}


class KDLossWeigher(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
        lm_loss: Tensor,
        kd_block_loss: Tensor,
        kd_logits_loss: Tensor,
    ) -> Tensor:
        raise NotImplementedError()


class StaticKDLossWeigher(KDLossWeigher):
    def __init__(
        self,
        lm_weight: float,
        kd_block_weight: float,
        kd_logits_weight: float,
    ):
        self.lm_weight = lm_weight
        self.kd_block_weight = kd_block_weight
        self.kd_logits_weight = kd_logits_weight

    def __call__(
        self,
        lm_loss: Tensor,
        kd_block_loss: Tensor,
        kd_logits_loss: Tensor,
    ) -> Tuple:
        lm_loss = self.lm_weight * lm_loss
        kd_block_loss = self.kd_block_weight * kd_block_loss
        kd_logits_loss = self.kd_logits_weight * kd_logits_loss

        loss = lm_loss + kd_block_loss + kd_logits_loss
        return loss, lm_loss, kd_block_loss, kd_logits_loss


class KDModel(nn.Module):
    def __init__(
        self,
        student_model,
        teacher_model,
        block_loss_func: Callable,
        logits_loss_func: Callable,
        kd_loss_weigher: StaticKDLossWeigher,
        teacher_requires_rope: bool = False,
    ):
        super().__init__()
        assert not student_model.abs_positional
        student_uses_rope = student_model.config.position_embedding_type in ["rope", "rope_llama4"]
        teacher_uses_rope = teacher_model.config.position_embedding_type in ["rope", "rope_llama4"]
        assert (student_uses_rope and teacher_uses_rope) or (
            not student_uses_rope and not teacher_uses_rope
        ), "We do not support mixed rope usage"
        self.use_rope = student_uses_rope

        self.logits_loss_func = logits_loss_func
        self.block_loss_func = block_loss_func
        # teacher_model.eval()
        # teacher_model.requires_grad_(False)
        self.student_wte, self.teacher_wte = (
            student_model.transformer.wte,
            teacher_model.transformer.wte,
        )
        self.blocks = nn.ModuleList()
        student_blocks = student_model.transformer.h
        teacher_blocks = teacher_model.transformer.h
        for i in range(max(len(teacher_blocks), len(student_blocks))):
            student_block = student_blocks[i] if i < len(student_blocks) else DummyBlock()
            teacher_block = teacher_blocks[i] if i < len(teacher_blocks) else DummyBlock()
            combined_block = KDBlock(
                student_block, teacher_block, teacher_requires_rope=teacher_requires_rope
            )
            self.blocks.append(combined_block)
        self.student_ln_f, self.teacher_ln_f = (
            student_model.transformer.ln_f,
            teacher_model.transformer.ln_f,
        )
        self.student_lm_head, self.teacher_lm_head = student_model.lm_head, teacher_model.lm_head
        self.block_size = student_model.block_size

        for teacher_module in (self.teacher_wte, self.teacher_ln_f, self.teacher_lm_head):
            teacher_module.eval()
            teacher_module.requires_grad_(False)
        self.kd_loss_weigher = kd_loss_weigher

    def forward(
        self,
        idx: Tensor,
        rope_cache: Optional[RoPECache] = None,
        max_seq_length: Optional[int] = None,
        varlen: bool = False,
        concat_token_id: Optional[int] = None,
        input_pos: Optional[int] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        is_decode: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T = idx.size()

        block_size = self.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if varlen:
            assert B == 1, "Varlen can be used only with batch_size==1"
            cu_seqlens = self.prepare_cu_seqlens(idx[0], block_size, concat_token_id)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            assert max_seqlen <= max_seq_length, (
                f"Cannot forward sequence of length {max_seqlen}, max seq length is only {max_seq_length}"
            )
            assert max_seqlen <= block_size, (
                f"Cannot forward sequence of length {max_seqlen}, block size is only {block_size}"
            )
        else:
            cu_seqlens = None
            max_seqlen = None
            assert T <= max_seq_length, (
                f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
            )
            assert T <= block_size, (
                f"Cannot forward sequence of length {T}, block size is only {block_size}"
            )
        assert max_seq_length <= block_size, (
            f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        )

        # forward the model itself
        rope = None
        if self.use_rope:
            if rope_cache is None:
                if self.rope_cache is None:
                    self.rope_cache = self.build_rope_cache(
                        dtype=torch.float32, device=torch.device("cpu")
                    )
                if idx.device != torch.device("meta"):
                    self.rope_cache = self.rope_cache.to(idx.device)
                rope_cache = self.rope_cache
            if input_pos is None:
                rope = rope_cache[:T]
            else:
                rope = rope_cache[input_pos - T : input_pos]

        x_student, x_teacher = self.student_wte(idx), self.teacher_wte(idx)

        kd_block_loss = torch.zeros(B, device=idx.device, dtype=torch.float32)
        for i_block, block in enumerate(self.blocks):
            x_student, x_teacher = block(x_student, x_teacher, rope)
            if self.kd_loss_weigher.kd_block_weight > 0:
                curr_block_loss = self.block_loss_func(x_student, x_teacher)
                kd_block_loss = kd_block_loss + curr_block_loss.float() / len(self.blocks)

        x_student, x_teacher = self.student_ln_f(x_student), self.teacher_ln_f(x_teacher)
        logits_student, logits_teacher = (
            self.student_lm_head(x_student),
            self.teacher_lm_head(x_teacher),
        )  # (b, t, vocab_size)
        if self.kd_loss_weigher.kd_logits_weight > 0:
            kd_logits_loss = self.logits_loss_func(logits_student, logits_teacher)
        else:
            kd_logits_loss = torch.zeros(B, device=idx.device, dtype=torch.float32)

        return logits_student, logits_teacher, kd_block_loss, kd_logits_loss

    def train(self, mode: bool = True):
        self.student_wte.train(mode)
        self.student_ln_f.train(mode)
        self.student_lm_head.train(mode)
        for block in self.blocks:
            block.student_block.train(mode)
        return self


class KDBlock(nn.Module):
    def __init__(self, student_block, teacher_block, teacher_requires_rope):
        super().__init__()
        self.student_block = student_block
        self.teacher_block = teacher_block
        self.teacher_requires_rope = teacher_requires_rope

        self.teacher_block.eval()
        self.teacher_block.requires_grad_(False)

    def forward(
        self, x_student: Tensor, x_teacher: Tensor, rope: Optional[RoPECache]
    ) -> tuple[Tensor, Tensor]:
        x_student = self.forward_block(x_student, self.student_block, rope=rope)
        x_teacher = self.forward_block(
            x_teacher, self.teacher_block, rope=rope if self.teacher_requires_rope else None
        )

        return x_student, x_teacher

    def forward_block(self, x: Tensor, block: Block, rope: Optional[RoPECache] = None) -> Tensor:
        x = block(x=x, rope=rope, input_pos=None, kv_cache=None, is_decode=False)
        return x
