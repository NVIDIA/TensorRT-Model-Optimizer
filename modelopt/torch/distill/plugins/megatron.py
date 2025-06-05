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

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Distillation loss function(s)."""

import logging
import types
from abc import ABCMeta
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer import MegatronModule, TransformerConfig
from torch import Tensor
from torch.nn.modules.loss import _Loss

import modelopt.torch.distill as mtd

logger = logging.getLogger(__name__)


def load_distillation_config(
    config_path: str | None, student_cfg: TransformerConfig, teacher_cfg: TransformerConfig
) -> dict[str, Any]:
    """Read the distillation yaml config file specified by ``args.export_kd_cfg``.

    Args:
        config_path: Path to user-defined distillation settings yaml file.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if not config_path:
        logger.warning("Distillation config not provided. Using default.")
        cfg = {
            "logit_layers": ["output_layer", "output_layer"],
            "intermediate_layer_pairs": [],
            "skip_lm_loss": True,
            "kd_loss_scale": 1.0,
        }
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    intermediate_pairs: list[str] = cfg["intermediate_layer_pairs"]
    logit_pair: list[str] = cfg["logit_layers"]
    skip_lm_loss: bool = cfg["skip_lm_loss"]
    loss_scale: float = cfg["kd_loss_scale"]

    criterion = {tuple(logit_pair): LogitsKLLoss(student_cfg, teacher_cfg)}
    for layer_names in intermediate_pairs:
        if torch.distributed.get_rank() == 0:
            print(
                "Distillation: Adding intermediate loss between"
                f" `{layer_names[0]}` of student (hidden size {student_cfg.hidden_size}) and"
                f" `{layer_names[1]}` of teacher (hidden size {teacher_cfg.hidden_size})."
            )
        criterion[tuple(layer_names)] = HiddenStateCosineLoss(student_cfg, teacher_cfg)

    loss_balancer = LogitsAndIntermediatesLossBalancer(
        kd_loss_scale=loss_scale, skip_original_loss=skip_lm_loss
    )

    cfg["criterion"] = criterion
    cfg["loss_balancer"] = loss_balancer

    return cfg


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(
        self,
        student_config: TransformerConfig,
        teacher_config: TransformerConfig,
        projection_layer: bool = False,
    ):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
            projection_layer: If True, create a linear layer to project student tensor to teacher's hidden dim.
        """
        super().__init__()
        self._config = student_config
        self._tensor_parallel = self._config.tensor_model_parallel_size > 1
        self._sequence_parallel = self._config.sequence_parallel

        if projection_layer:
            self._projection = ProjectionLayer(student_config, teacher_config)
        else:
            self._projection = None

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        if self._projection is not None:
            predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor, tp_reduce: bool = False) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce)


class MSELoss(BaseLoss):
    """Calculates Mean Squared Error loss between two tensors without reducing the sequence dim."""

    def __init__(self, student_config: TransformerConfig, teacher_config: TransformerConfig):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(student_config, teacher_config)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            MSE loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # TP irrelevant since MSE loss gradients are per-input element.
        loss = F.mse_loss(predictions, targets, reduction="none")
        loss = loss.sum(dim=-1)

        return self.post_forward(loss)


class HiddenStateCosineLoss(BaseLoss):
    """Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, so extra restrictions are in place.
    """

    def __init__(self, student_config: TransformerConfig, teacher_config: TransformerConfig):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(student_config, teacher_config, projection_layer=True)

        if self._tensor_parallel and not self._sequence_parallel:
            logger.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement or use `--sequence_parallel` if tensor parallel is enabled."
            )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            Cosine loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.cosine_embedding_loss(
            predictions.view(-1, predictions.size(-1)),
            targets.view(-1, targets.size(-1)),
            targets.new_ones(1),
            reduction="none",
        )
        loss = loss.view(*predictions.shape[:2])

        if self._sequence_parallel:
            # Can efficiently gather size [s, b] tensor now for loss-masking purposes.
            # TODO(aanoosheh) Reconsider for memory savings by splitting loss mask instead.
            loss = gather_from_sequence_parallel_region(loss)

        return self.post_forward(loss)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self,
        student_config: TransformerConfig,
        teacher_config: TransformerConfig,
        temperature: float = 1.0,
        reverse: bool = False,
    ):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(student_config, teacher_config)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # Division by temp should happen prior to finding max for both student and teacher.
        # Currently we don't use temperature in any of ours runs (temp=1.0)
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Compute local softmax, and the reweight to compute global softmax.
        if self._tensor_parallel:
            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use `gather_from_tensor_model_parallel_region` here since it discards
            # gradients from other ranks - we need to all_reduce the gradients as well.
            denom_teacher = all_reduce_autograd(
                denom_teacher, group=get_tensor_model_parallel_group()
            )

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(
                denom_student, group=get_tensor_model_parallel_group()
            )

            slen, bsz, sharded_vocab_size = output_student.shape
            student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )
            teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )

            if self._reverse:
                loss = torch.sum(
                    F.kl_div(teacher_log_prob, student_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )

        elif self._reverse:
            loss = torch.sum(
                F.kl_div(
                    F.log_softmax(output_teacher, dim=-1),
                    F.softmax(output_student, dim=-1),
                    reduction="none",
                ),
                dim=-1,
            )
        else:
            loss = torch.sum(
                F.kl_div(
                    F.log_softmax(output_student, dim=-1),
                    F.softmax(output_teacher, dim=-1),
                    reduction="none",
                ),
                dim=-1,
            )

        return self.post_forward(loss, tp_reduce=True)


########################################################


class LogitsAndIntermediatesLossBalancer(mtd.DistillationLossBalancer):
    """LossBalancer implementation for Logit and Intermediate losses.

    Dynamically weighs distillation and original losses to balance during training.
    """

    def __init__(self, kd_loss_scale: float = 1.0, skip_original_loss: bool = False):
        """Constructor.

        Args:
            kd_loss_scale: Multiply distillation losses by this before weighing.
                (Not used when `skip_original_loss` is True.)
            skip_original_loss: Used to signal whether the original loss should be used, regardless
                of whether it was passed into ``mtd.DistillationModel.compute_kd_loss()`` or not.
        """
        super().__init__()
        self._kd_loss_scale = kd_loss_scale
        self._skip_original_loss = skip_original_loss

    def forward(self, loss_dict: dict[str, Tensor]) -> Tensor:
        """Forward function.

        Args:
            loss_dict: All individual scalar losses, passed in during ``mtd.DistillationModel.compute_kd_loss()``

        Returns:
            Aggregate total scalar loss.
        """
        original_loss = loss_dict.pop(mtd.loss_balancers.STUDENT_LOSS_KEY)
        for _key, _loss in loss_dict.items():
            if _key.startswith(LogitsKLLoss.__name__):
                logits_loss = _loss  # should only be one
        intermediate_loss = sum(loss_dict.values())

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.item() / intermediate_loss.item()
            intermediate_loss *= dynamic_scale
            kd_loss_scale = self._kd_loss_scale / 2.0
        else:
            kd_loss_scale = self._kd_loss_scale

        if self._skip_original_loss:
            kd_loss = logits_loss + intermediate_loss
            total_loss = kd_loss
        else:
            kd_loss = (logits_loss + intermediate_loss) * kd_loss_scale
            dynamic_scale = original_loss.item() / kd_loss.item()
            total_loss = original_loss + kd_loss * dynamic_scale

        return total_loss


########################################################


class ProjectionLayer(MegatronModule):
    """Module to project student layer activations to teacher's size."""

    def __init__(
        self,
        student_config: TransformerConfig,
        teacher_config: TransformerConfig,
    ):
        """Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(config=student_config)
        if student_config.hidden_size == teacher_config.hidden_size:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            self.apply(self._init_weights)
            setattr(self._fit.weight, "sequence_parallel", self.config.sequence_parallel)
            setattr(self._fit.bias, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, student_tensor: Tensor):
        """Forward function.

        Args:
            student_tensor: Tensor to be fit to teacher size.
        """
        return self._fit(student_tensor)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(
    tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD
):
    """AllReduce with autograd."""
    return _AllReduce.apply(op, group, tensor)


########################################################


def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: dict[str, Any]):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = types.MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if self.training:
            return torch.zeros_like(labels)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = types.MethodType(_compute_language_model_loss, model)
