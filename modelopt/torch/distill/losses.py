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

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Different types of distillation losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss

__all__ = ["LogitsDistillationLoss", "MGDLoss"]


class LogitsDistillationLoss(Loss):
    """KL-Divergence loss on output logits.

    This function implements the distillation loss found in the paper: https://arxiv.org/abs/1503.02531.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        """Constructor.

        Args:
            temperature: A value used to soften the logits_t and logits_s before computing loss on them.
            reduction: How to reduce the final pointwise loss before returning. Pass ``"none"`` to
                use your own reduction function afterwards, i.e. with loss masks.
        """
        super().__init__()
        self._temperature = temperature
        self._reduction = reduction

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        """Compute KD loss on student and teacher logits.

        Args:
            logits_s: Student's logits, treated as prediction.
            logits_t: Teacher's logits, treated as label.

        .. note::

            Assumes class logits dimension is last.
        """
        soft_log_probs = F.log_softmax(logits_s / self._temperature, dim=-1)
        soft_targets = F.softmax(logits_t / self._temperature, dim=-1)

        soft_log_probs = soft_log_probs.view(-1, soft_log_probs.size(-1))
        soft_targets = soft_targets.view(-1, soft_targets.size(-1))

        kd_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction=self._reduction)

        # Since the magnitudes of the gradients produced by the soft logits scale as 1/(T^2),
        # multiplying them by T^2 ensures that the relative contributions of the logits
        # remain roughly unchanged while experimenting with meta-parameters.
        kd_loss *= self._temperature**2

        return kd_loss


class MGDLoss(Loss):
    """PyTorch version of Masked Generative Distillation.

    This function implements the distillation loss found in the paper: https://arxiv.org/abs/2205.01529.
    """

    def __init__(
        self,
        num_student_channels: int,
        num_teacher_channels: int,
        alpha_mgd: float = 1.0,
        lambda_mgd: float = 0.65,
    ):
        """Constructor.

        Args:
            num_student_channels: Number of channels in the student's feature map.
            num_teacher_channels: Number of channels in the teacher's feature map.
            alpha_mgd: Scalar final loss is multiplied by. Defaults to 1.0.
            lambda_mgd: Masked ratio. Defaults to 0.65.
        """
        super().__init__()
        self._alpha_mgd = alpha_mgd
        self._lambda_mgd = lambda_mgd

        if num_student_channels != num_teacher_channels:
            self.align = nn.Conv2d(
                num_student_channels,
                num_teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.align = nn.Identity()

        self.generation = nn.Sequential(
            nn.Conv2d(
                num_teacher_channels,
                num_teacher_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_teacher_channels,
                num_teacher_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _loss_fn(self, out_s: torch.Tensor, out_t: torch.Tensor):
        n, _, h, w = out_t.shape

        mat = torch.rand((n, 1, h, w), device=out_s.device)
        mat = torch.where(mat > 1 - self._lambda_mgd, 0, 1)

        masked_feats = torch.mul(out_s, mat)
        new_feats = self.generation(masked_feats)

        kd_loss = F.mse_loss(new_feats, out_t)

        return kd_loss

    def forward(self, out_s: torch.Tensor, out_t: torch.Tensor):
        """Forward function.

        Args:
            out_s: Student's feature map (shape BxCxHxW).
            out_t: Teacher's feature map (shape BxCxHxW).
        """
        assert out_s.shape[-2:] == out_t.shape[-2:]

        out_s = self.align(out_s)
        loss = self._loss_fn(out_s, out_t) * self._alpha_mgd

        return loss
