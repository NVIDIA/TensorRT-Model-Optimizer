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

"""ModelOpt plugin for transformers Trainer."""

import os
from contextlib import contextmanager, suppress

import torch
from transformers import Trainer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill import KDLossConfig
from modelopt.torch.distill.mode import _convert_for_kd
from modelopt.torch.distill.plugins.huggingface import KDTrainer
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils import print_rank_0


class EvalOnlyError(Exception):
    """Exception to raise when evaluation is only needed."""


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    metrics = {"perplexity": float(torch.exp(torch.tensor(metrics["eval_loss"]))), **metrics}
    return metrics


def save_modelopt_state_with_weights(model, modelopt_state_path, save_weights=False):
    """Save the modelopt weights for fsdp2 models."""
    modelopt_state = mto.modelopt_state(model)
    modelopt_state["modelopt_state_dict"] = [
        state
        for state in modelopt_state["modelopt_state_dict"]
        if "kd_loss" not in state and "export_student" not in state
    ]
    if save_weights:
        state_dict = model.state_dict()
        modelopt_weights = {}
        for k, v in state_dict.items():
            if "_quantizer" in k:
                modelopt_weights[k] = v.cpu()
        modelopt_state["modelopt_state_weights"] = modelopt_weights

    torch.save(modelopt_state, modelopt_state_path)


def restore_modelopt_state_with_weights(model, modelopt_state_path):
    """Restore the modelopt weights for fsdp2 models."""
    _modelopt_state = torch.load(modelopt_state_path, weights_only=False)
    modelopt_weights = _modelopt_state.pop("modelopt_state_weights", None)
    restore_from_modelopt_state(model, _modelopt_state)
    if modelopt_weights is not None:
        model.load_state_dict(modelopt_weights, strict=False)


@contextmanager
def reset_distillation_model(model):
    """Reset the distillation model to student model."""
    if not hasattr(model, "hide_teacher_model"):
        yield
        return
    with (
        model.hide_teacher_model(enable=True),
        model.hide_loss_modules(enable=True),
        model.only_student_forward(enable=True),
    ):
        yield


class QATTrainer(Trainer):
    """Trainer for quantization aware training."""

    def __init__(
        self,
        *args,
        quant_args=None,
        quant_cfg=None,
        **kwargs,
    ):
        """Initialize the trainer with modelopt states."""
        self.quant_args = quant_args
        if quant_cfg is None and quant_args.quant_cfg is not None:
            quant_cfg = getattr(mtq, quant_args.quant_cfg)
        self.quant_cfg = quant_cfg
        self._eval_without_training = False

        super().__init__(*args, **kwargs)
        self._is_fsdp2 = self.is_fsdp_enabled and (
            getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        )
        self._modelopt_state_path = os.path.join(self.args.output_dir, "modelopt_state_train.pth")
        # FSDP1 requires pre-restoring the quantized model if the modelopt state exists.
        if os.path.exists(self._modelopt_state_path) and not self._is_fsdp2:
            self._quantize_model()

    def _get_quantize_forward_loop(self, data_loader, use_eval_loop=True):
        def forward_loop(_model):
            print_rank_0("Calibrating...")
            if use_eval_loop:
                return self.evaluation_loop(
                    data_loader,
                    description="Calibration",
                    prediction_loss_only=True,
                    ignore_keys=None,
                    metric_key_prefix="calibration",
                )
            else:
                for batch in data_loader:
                    batch = self._prepare_inputs(batch)
                    _model(**batch)
            print_rank_0("Calibration done!")

        return forward_loop

    def _quantize_model(self, use_eval_loop=True):
        """Quantize the model. Restore the quantization state if it exists."""
        model = self.accelerator.unwrap_model(self.model)

        if os.path.exists(self._modelopt_state_path):
            print_rank_0(f"Restoring modelopt state from {self._modelopt_state_path}...")
            restore_modelopt_state_with_weights(self.model, self._modelopt_state_path)
            print_rank_0("Restored model from modelopt state.")
        else:
            dataset = torch.utils.data.Subset(
                self.eval_dataset, list(range(self.quant_args.calib_size))
            )
            data_loader = self.get_eval_dataloader(dataset)
            forward_loop = self._get_quantize_forward_loop(data_loader, use_eval_loop)

            print_rank_0("Quantizing the model...")
            mtq.quantize(model, self.quant_cfg, forward_loop)
            print_rank_0("Quantization done!")
            print_rank_0(f"Saving modelopt state to {self._modelopt_state_path}")
            save_modelopt_state_with_weights(model, self._modelopt_state_path, save_weights=True)
            torch.cuda.empty_cache()
            if use_eval_loop:
                self.callback_handler.on_evaluate(self, self.state, self.control, metrics=None)

    def _evaluate(self, *args, **kwargs):
        """Quantize the model before evaluation.

        Note that we do not force to run the evaluation if the `eval_on_start` is False.
        """
        if self.quant_cfg is not None and not is_quantized(self.model):
            self._quantize_model()
            metrics = None
            if self._original_evaluate_on_start:
                metrics = super()._evaluate(*args, **kwargs)
        else:
            metrics = super()._evaluate(*args, **kwargs)
        # used for eval without training
        if self._eval_without_training:
            metrics = get_metrics_with_perplexity(metrics)
            print_rank_0(f"Evaluation results: \n{metrics}")
            raise EvalOnlyError()
        return metrics

    def train(self, *args, eval_only=False, **kwargs):
        """Train the model with quantization."""
        self._eval_without_training = eval_only
        self._original_evaluate_on_start = (
            self.args.eval_on_start if not self._eval_without_training else True
        )
        if self.quant_args.quant_cfg is not None and not is_quantized(self.model):
            self.args.eval_on_start = True
        with suppress(EvalOnlyError):
            super().train(*args, **kwargs)
        self.args.eval_on_start = self._original_evaluate_on_start


class QADTrainer(QATTrainer, KDTrainer):
    """Trainer for quantization aware distillation."""

    def __init__(
        self,
        *args,
        distill_config=None,
        **kwargs,
    ):
        """Initialize the trainer with modelopt states."""
        assert distill_config is not None, "`distill_config` is required for QAD."
        self.distill_config = distill_config

        super().__init__(*args, **kwargs)

        # Note: QAD doesn't work with FSDP wrapped model. We quantize model before the wrapper.
        # The drawback is that we can't train a model that is bigger than a single GPU memory.
        # And memory efficient loading doesn't work.
        self.model.cuda()
        if self.quant_cfg is not None and not is_quantized(self.model):
            self._quantize_model(use_eval_loop=False)
        self._convert_to_distillation_model()

    def _convert_to_distillation_model(self):
        """Convert the model to a distillation model."""
        # We don't need any save/restore feature of the distallation mode, so we skip it here.
        _convert_for_kd(self.model, KDLossConfig(**self.distill_config))
        print_rank_0("Distillation model created.")

    def train(self, *args, **kwargs):
        """Train the model with QAD."""
        self.compute_loss_func = lambda *args, **kwargs: self.model.compute_kd_loss()
        return super().train(*args, **kwargs)

    def save_model(self, output_dir: str, export_student: bool = False, *args, **kwargs):
        """Dumps model to disk without teacher model and loss modules.

        Args:
            output_dir: The directory to save the model and ModelOpt states.
        """
        model = self.accelerator.unwrap_model(self.model)
        # DDP and FSDP2 uses the original save_model method.
        if not self.is_fsdp_enabled or self._is_fsdp2:
            with reset_distillation_model(model):
                return QATTrainer.save_model(self, output_dir, *args, **kwargs)

        return super().save_model(output_dir, export_student, *args, **kwargs)
