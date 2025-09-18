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
import re
from abc import ABCMeta
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.transformer import MegatronModule, TransformerLayer
from megatron.core.utils import get_model_config
from torch import Tensor
from torch.nn.modules.loss import _Loss

import modelopt.torch.distill as mtd
from modelopt.torch.distill.config import Criterion

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.transformer import TransformerConfig


logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Knowledge-Distillation config.

    Args:
        intermediate_layer_pairs: List of tuples of intermediate layer names.
        logit_layers: Tuple of logit layer names.
        skip_lm_loss: Whether to skip computing the standard language model loss (default: ``True``).
        kd_loss_scale: Relative scaling factor for the distillation loss if ``skip_lm_loss`` is ``False``.
        logit_kl_temperature: Temperature for the logit KL-divergence loss.
    """

    intermediate_layer_pairs: list[tuple[str, str]] = field(default_factory=list)
    logit_layers: tuple[str, str] = ("output_layer", "output_layer")
    skip_lm_loss: bool = True
    kd_loss_scale: float = 1.0
    logit_kl_temperature: float = 1.0
    criterion: Criterion | None = None
    loss_balancer: mtd.DistillationLossBalancer | None = None

    def __post_init__(self):
        assert len(self.logit_layers) == 2, f"{self.logit_layers=}"
        assert all(len(pair) == 2 for pair in self.intermediate_layer_pairs), (
            f"{self.intermediate_layer_pairs=}"
        )
        assert self.kd_loss_scale > 0, f"{self.kd_loss_scale=}"
        assert self.logit_kl_temperature > 0, f"{self.logit_kl_temperature=}"


def load_distillation_config(
    config_path: str | None, student_cfg: "TransformerConfig", teacher_cfg: "TransformerConfig"
) -> DistillationConfig:
    """Read the distillation yaml config file specified by ``args.export_kd_cfg``.

    Args:
        config_path: Path to user-defined distillation settings yaml file.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg = DistillationConfig(**cfg)
    else:
        logger.warning("Distillation config not provided. Using default.")
        cfg = DistillationConfig()

    criterion = {}
    if student_cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage():
        criterion[tuple(cfg.logit_layers)] = LogitsKLLoss(
            student_cfg, temperature=cfg.logit_kl_temperature
        )
        # NOTE: Projection layer shared among intermediate layer pairs.
        projection_layer = ProjectionLayer(student_cfg, teacher_cfg)

        for student_layer, teacher_layer in cfg.intermediate_layer_pairs:
            if parallel_state.get_tensor_and_context_parallel_rank() == 0:
                logger.info(
                    "Distillation: Adding intermediate loss between"
                    f" `{student_layer}` of student (hidden size {student_cfg.hidden_size}) and"
                    f" `{teacher_layer}` of teacher (hidden size {teacher_cfg.hidden_size})."
                )
            student_layer = _adjust_layer_index_for_pp(student_layer, student_cfg)
            teacher_layer = _adjust_layer_index_for_pp(teacher_layer, teacher_cfg)
            criterion[(student_layer, teacher_layer)] = HiddenStateCosineLoss(
                student_cfg, projection_layer=projection_layer
            )

    loss_balancer = LogitsAndIntermediatesLossBalancer(
        kd_loss_scale=cfg.kd_loss_scale, skip_original_loss=cfg.skip_lm_loss
    )

    cfg.criterion = criterion
    cfg.loss_balancer = loss_balancer

    return cfg


def _adjust_layer_index_for_pp(submodule_name, model_cfg):
    """Adjust any sequence-based layer indices found in a submodule name for Pipeline Parallelism."""
    match = re.search(r"(?<=\.)\d+(?=\.)", submodule_name)
    if not match:
        return submodule_name

    offset = TransformerLayer._get_layer_offset(model_cfg)
    new_layer_idx = int(match.group(0)) - offset
    if new_layer_idx < 0:
        raise ValueError(f"Layer {submodule_name} does not fall on final PP rank.")

    new_submodule_name = submodule_name.replace(match.group(0), str(new_layer_idx))
    if parallel_state.get_tensor_and_context_parallel_rank() == 0:
        logger.info(
            f'Distillation: Renamed layer "{submodule_name}" on final PP rank to "{new_submodule_name}"'
        )
    return new_submodule_name


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(
        self, model_config: "TransformerConfig", projection_layer: nn.Module | None = None
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__()
        self._config = model_config
        self._projection = projection_layer

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        if self._projection is not None:
            predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(
        self, loss: Tensor, tp_reduce: bool = False, is_sequence_parallel: bool = False
    ) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce, is_sequence_parallel)


class MSELoss(BaseLoss):
    """Calculates MSE loss between two tensors without reducing the sequence dim."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            MSE loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.mse_loss(predictions, targets, reduction="none")
        loss = loss.sum(dim=-1)

        return self.post_forward(loss)


class HiddenStateCosineLoss(BaseLoss):
    """Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, with full hidden dimension size.
    We recommend only applying this loss to LayerNorm outputs, which have full hidden dim even when TP is used.
    """

    def __init__(
        self, model_config: "TransformerConfig", projection_layer: nn.Module | None = None
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__(model_config, projection_layer=projection_layer)

        if self._config.tensor_model_parallel_size > 1:
            logger.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement. We recommend only applying this loss to LayerNorm outputs, "
                "which have full hidden dim even when TP is used."
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

        # NOTE: Tensor sequence length is still split among TP ranks.
        return self.post_forward(loss, is_sequence_parallel=self._config.sequence_parallel)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self, model_config: "TransformerConfig", temperature: float = 1.0, reverse: bool = False
    ):
        """Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(model_config)
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
        if self._config.tensor_model_parallel_size > 1:
            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use standard reduction function here since the computation
            # that follows it isn't identical across TP ranks.
            denom_teacher = all_reduce_autograd(
                denom_teacher, group=parallel_state.get_tensor_model_parallel_group()
            )

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(
                denom_student, group=parallel_state.get_tensor_model_parallel_group()
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
        for _key in loss_dict:
            if _key.startswith(LogitsKLLoss.__name__):
                logits_key = _key  # should only be one
        logits_loss = loss_dict.pop(logits_key)
        intermediate_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.item() / intermediate_loss.item()
            intermediate_loss_scaled = intermediate_loss * dynamic_scale
        else:
            intermediate_loss = logits_loss.new_tensor(intermediate_loss)
            intermediate_loss_scaled = intermediate_loss

        if self._skip_original_loss:
            total_loss = logits_loss + intermediate_loss_scaled
        else:
            kd_loss = logits_loss + intermediate_loss_scaled
            kd_loss *= original_loss.item() / kd_loss.item()
            total_loss = original_loss + kd_loss * self._kd_loss_scale

        out_dict = {
            "kd_loss": total_loss,
            "logits_loss": logits_loss,
            "intermediate_loss": intermediate_loss,
        }
        return out_dict


class ProjectionLayer(MegatronModule):
    """Module to project student layer activations to teacher's size."""

    def __init__(self, student_config: "TransformerConfig", teacher_config: "TransformerConfig"):
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
            # Attribute below needed to reduce gradients during backward properly.
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
        if isinstance(module, nn.Linear):
            self.config.init_method(module.weight.data)
            if module.bias is not None:
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
    """Custom all-reduce function.

    Needed instead of other all-reduce functions available when the computation following
    the all-reduce call differs per rank. In KL loss, this corresponds to the different numerators.
    """
    return _AllReduce.apply(op, group, tensor)


########################################################


def adjust_distillation_model_for_mcore(
    model: mtd.DistillationModel, distill_cfg: DistillationConfig
):
    """Extra modifications to ``mtd.DistillationModel`` required for Megatron-Core."""

    # Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return type(self).sharded_state_dict(self, *args, **kwargs)

    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_student_lm_loss(self, labels, logits) -> Tensor:
        if distill_cfg.skip_lm_loss and self.training:
            return torch.zeros_like(labels, dtype=logits.dtype)
        return type(self).compute_language_model_loss(self, labels, logits)

    model.compute_language_model_loss = MethodType(_compute_student_lm_loss, model)

    # Skip `lm_loss` always for teacher.
    def _compute_teacher_lm_loss(self, labels, logits) -> Tensor:
        return torch.zeros_like(labels, dtype=logits.dtype)

    model.teacher_model.compute_language_model_loss = MethodType(
        _compute_teacher_lm_loss, model.teacher_model
    )

    # HACK: Pipeline-parallel Distillation requires splitting input tensor into student and teacher parts.
    def _set_student_input_tensor_shape(self, shapes: list[tuple[int]]):
        self._tensor_split_idx = shapes[0][-1]

    def _set_input_tensor(self, input_tensors: list[Tensor]):
        teacher_inputs = [
            t[..., self._tensor_split_idx :] if t is not None else t for t in input_tensors
        ]
        student_inputs = [
            t[..., : self._tensor_split_idx] if t is not None else t for t in input_tensors
        ]
        type(self).set_input_tensor(self.teacher_model, teacher_inputs)
        type(self).set_input_tensor(self, student_inputs)

    model.set_student_input_tensor_shape = MethodType(_set_student_input_tensor_shape, model)
    model.set_input_tensor = MethodType(_set_input_tensor, model)

    # HACK: Concatenate output tensors when PP>1 so they can be passed between ranks.
    def _forward(self, *args, **kwargs):
        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(*args, **kwargs)
        with self.only_student_forward():
            student_output = type(self).forward(self, *args, **kwargs)

        if not parallel_state.is_pipeline_last_stage():
            return torch.cat([student_output, teacher_output], dim=-1)
        else:
            return student_output

    model.forward = MethodType(_forward, model)


def get_tensor_shapes_adjust_fn_for_distillation(
    model: torch.nn.Module | list[torch.nn.Module], **kwargs
) -> Callable | None:
    """Return the function to adjust tensor shapes for Distillation in Megatron-Core's forward pass.

    Currently only used during non-interleaved pipelining for Distillation.
    Concatenates sizes of student and teacher output tensors for inter-process communication.
    """
    if (
        parallel_state.get_pipeline_model_parallel_world_size() == 1
        or parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        return None
    # Unwrap
    if isinstance(model, list):
        model = model[0]
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, mtd.DistillationModel):
        return None

    def adjust_tensor_shapes(
        recv_tensor_shapes: list[tuple[int, ...]], send_tensor_shapes: list[tuple[int, ...]]
    ):
        teacher_config = get_model_config(model.teacher_model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        teacher_recv_tensor_shapes = get_tensor_shapes(
            config=teacher_config, tp_group=tp_group, cp_group=cp_group, **kwargs
        )
        teacher_send_tensor_shapes = get_tensor_shapes(
            config=teacher_config, tp_group=tp_group, cp_group=cp_group, **kwargs
        )
        model.set_student_input_tensor_shape(recv_tensor_shapes)

        for i, shape in enumerate(recv_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_recv_tensor_shapes[0][-1]  # type: ignore[index]
            recv_tensor_shapes[i] = tuple(shape)
        for i, shape in enumerate(send_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_send_tensor_shapes[0][-1]  # type: ignore[index]
            send_tensor_shapes[i] = tuple(shape)

        return recv_tensor_shapes, send_tensor_shapes

    return adjust_tensor_shapes
