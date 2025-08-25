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

import gc
import os
from contextlib import suppress
from dataclasses import dataclass, field

import torch
import torch.distributed.checkpoint as dist_cp
from accelerate.utils import save_fsdp_model

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill import KDLossConfig
from modelopt.torch.distill.mode import _convert_for_kd
from modelopt.torch.distill.plugins.huggingface import KDTrainer
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.opt.plugins import ModelOptHFTrainer
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.utils import (
    calibrate_with_adapters,
    disable_lora_quantizers_in_config,
    is_quantized,
)
from modelopt.torch.utils import print_rank_0

# TODO: Enable documentation rendering for this class


@dataclass
class QuantizationArguments:
    """Quantization arguments for quantization aware training.

    This classes is intended to be used with ModelOpt's QAT/QAD trainers for HuggingFace models.
    This class can also be used to parse the quantization arguments
    from the command line to the taining script.
    """

    quant_cfg: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
        },
    )
    calib_size: int = field(
        default=512,
        metadata={
            "help": (
                "Specify the calibration size for quantization. The calibration dataset is used to"
                " setup the quantization scale parameters for PTQ/QAT."
            )
        },
    )
    compress: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compress the model weights after quantization for QLoRA. "
                "This is useful for reducing the model size."
            )
        },
    )


class QuantizationArgumentsWithConfig(QuantizationArguments):
    """Quantization arguments for quantization aware training with config.

    This class is intended to be used with ModelOpt's QAT/QAD trainers for HuggingFace models,
    however, it cannot be used for command line parsing.
    """

    quant_cfg: str | QuantizeConfig | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
        },
    )


class EvalOnlyError(Exception):
    """Exception to raise when evaluation is only needed."""


def check_awq_smoothquant(quant_cfg):
    # TODO: Remove this once deepspeed for AWQ and SmoothQuant is added
    """Get the quantization type from the configuration."""
    if quant_cfg is None:
        return False
    algorithm = quant_cfg.get("algorithm", {})
    is_awq_smoothquant = False
    # Check SmoothQuant and AWQ
    if algorithm and ("smoothquant" in algorithm or "awq" in algorithm):
        is_awq_smoothquant = True

    return is_awq_smoothquant


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    metrics = {"perplexity": float(torch.exp(torch.tensor(metrics["eval_loss"]))), **metrics}
    return metrics


def convert_sharded_model_to_hf_format(
    model, model_path, modelopt_state_name="modelopt_state.pth", output_path=None
):
    """Convert a sharded model to HF format.

    Args:
        model: The original HF model.
        model_path: The path to the sharded model with pytorch_model_fsdp_0 directory.
        modelopt_state_name: The name of the modelopt state file. If not provided, the default name
            "modelopt_state.pth" will be used.
        output_path: The path to save the converted model. If not provided, the model will be saved
            to the same directory as the sharded model.
    """
    if output_path is None:
        output_path = model_path
    os.makedirs(output_path, exist_ok=True)
    state_dict = {"model": model.state_dict()}
    sharded_model_path = os.path.join(model_path, "pytorch_model_fsdp_0")
    modelopt_state_path = os.path.join(model_path, modelopt_state_name)
    if not os.path.exists(sharded_model_path):
        print_rank_0(f"Sharded model path does not exist: {sharded_model_path}")
        return model
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(sharded_model_path),
        no_dist=True,
    )
    model.load_state_dict(state_dict["model"])
    restore_modelopt_state_with_weights(model, modelopt_state_path)
    model.save_pretrained(output_path)
    return model


def restore_modelopt_state_with_weights(model, modelopt_state_path):
    """Restore the modelopt weights for fsdp2 models."""
    _modelopt_state = torch.load(modelopt_state_path, weights_only=False)
    modelopt_weights = _modelopt_state.pop("modelopt_state_weights", None)
    restore_from_modelopt_state(model, _modelopt_state)
    if modelopt_weights is not None:
        model.load_state_dict(modelopt_weights, strict=False)


class QATTrainer(ModelOptHFTrainer):
    """A drop-in replacement of HuggingFace's Trainer for quantization aware training with ModelOpt.

    This class takes an additional optional argument `quant_args` of type
    :class:`QuantizationArgumentsWithConfig <QuantizationArgumentsWithConfig>`
    to specify the quantization arguments.
    """

    def __init__(
        self,
        *args,
        quant_args: QuantizationArgumentsWithConfig | QuantizationArguments | None = None,
        **kwargs,
    ):
        """Initialize the trainer with modelopt states."""
        super().__init__(*args, **kwargs)

        self.quant_args = quant_args
        quant_cfg = None
        if quant_args is not None and getattr(quant_args, "quant_cfg", None):
            quant_cfg = (
                getattr(mtq, quant_args.quant_cfg)
                if isinstance(quant_args.quant_cfg, str)
                else quant_args.quant_cfg
            )
        self.quant_cfg = quant_cfg
        self._eval_without_training = False

        self._is_fsdp2 = self.is_fsdp_enabled and (
            getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        )
        self.fsdp_state_dict_type = (
            str(self.accelerator.state.fsdp_plugin.state_dict_type) if self.is_fsdp_enabled else ""
        )
        self._modelopt_state_path = os.path.join(self.args.output_dir, "modelopt_state_train.pth")

        # Add lora adapter before quantizing the model
        if getattr(self.args, "lora_config", None) is not None and not hasattr(
            self.model, "peft_config"
        ):
            # TODO: use get_peft_model here instead of add_adapter
            self.model.add_adapter(self.args.lora_config, adapter_name="adapter")
            print_rank_0("Lora adapter added.")

        if hasattr(self.model, "peft_config") and self.quant_cfg is not None:
            target_modules = (
                self.args.lora_config.target_modules if hasattr(self.args, "lora_config") else []
            )
            disable_lora_quantizers_in_config(self.quant_cfg, target_modules)

        if self.is_deepspeed_enabled:
            assert not check_awq_smoothquant(self.quant_cfg), (
                f"QAT DeepSpeed does not currently support AWQ or SmoothQuant: {self.quant_cfg}"
            )

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

    def _save_modelopt_state_with_weights(self, model, modelopt_state_path, save_weights=False):
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

        if self.args.should_save:
            torch.save(modelopt_state, modelopt_state_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _quantize_model(self, use_eval_loop=True):
        """Quantize the model. Restore the quantization state if it exists."""
        model = self.accelerator.unwrap_model(self.model)
        if os.path.exists(self._modelopt_state_path):
            print_rank_0(f"Restoring modelopt state from {self._modelopt_state_path}...")
            restore_modelopt_state_with_weights(self.model, self._modelopt_state_path)
            print_rank_0("Restored model from modelopt state.")
        else:
            dataset = torch.utils.data.Subset(
                self.eval_dataset,
                list(range(min(self.quant_args.calib_size, len(self.eval_dataset)))),  # type: ignore [union-attr]
            )
            data_loader = self.get_eval_dataloader(dataset)
            forward_loop = self._get_quantize_forward_loop(data_loader, use_eval_loop)
            with calibrate_with_adapters(model, self.args):
                print_rank_0("Quantizing the model...")
                mtq.quantize(model, self.quant_cfg, forward_loop)  # type: ignore [arg-type]
                print_rank_0("Quantization done!")

            if getattr(self.quant_args, "compress", False):
                print_rank_0("Compressing model after calibration")
                mtq.compress(model)

            # Force garbage collection to free up memory
            gc.collect()

            print_rank_0(f"Saving modelopt state to {self._modelopt_state_path}")
            self._save_modelopt_state_with_weights(
                model, self._modelopt_state_path, save_weights=True
            )
            torch.cuda.empty_cache()
            if use_eval_loop:
                self.callback_handler.on_evaluate(self, self.state, self.control, metrics=None)

        if self.accelerator.is_main_process:
            mtq.print_quant_summary(model)

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
        if getattr(self.quant_args, "quant_cfg", None) is not None and not is_quantized(self.model):
            self.args.eval_on_start = True
        train_result = None
        with suppress(EvalOnlyError):
            train_result = super().train(*args, **kwargs)
        self.args.eval_on_start = self._original_evaluate_on_start
        return train_result

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False, *args, **kwargs
    ):
        """Save the quantized model."""
        dict_type = (
            str(self.accelerator.state.fsdp_plugin.state_dict_type) if self.is_fsdp_enabled else ""
        )
        if not _internal_call and self.is_fsdp_enabled and "SHARDED_STATE_DICT" in dict_type:
            # The default save_model in Trainer doesn't save checkpoint with SHARDED_STATE_DICT + FSDP.
            # We save the model manually at the end of the training in order to convert the last
            # checkpoint from distcp to HF compatible format.
            if output_dir is None:
                output_dir = self.args.output_dir
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                output_dir,
            )
            self.processing_class.save_pretrained(output_dir)
            self.model.config.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call, *args, **kwargs)


class QADTrainer(QATTrainer, KDTrainer):
    """A drop-in replacement of HuggingFace's Trainer for quantization aware distillation with ModelOpt.

    This class takes additional optional argument `distill_config` to specify the distillation
    arguments in addition to the `quant_args` argument.
    For details on `quant_args` see
    :class:`QATTrainer <QATTrainer>`.
    """

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
        if getattr(self.args, "lora_config", None) is not None:
            self.model.add_adapter(self.args.lora_config, adapter_name="adapter")
            print_rank_0("Lora adapter added.")
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

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
        export_student: bool = False,
        *args,
        **kwargs,
    ):
        """Dumps model to disk without teacher model and loss modules.

        Args:
            output_dir: The directory to save the model and ModelOpt states.
            export_student: Whether to export the student model.
        """
        if "SHARDED_STATE_DICT" in self.fsdp_state_dict_type and self._is_fsdp2:
            if export_student:
                model = self.accelerator.unwrap_model(self.model)
                model = model.export()
            return QATTrainer.save_model(self, output_dir, _internal_call, *args, **kwargs)
        return KDTrainer.save_model(
            self, output_dir, _internal_call, export_student, *args, **kwargs
        )
