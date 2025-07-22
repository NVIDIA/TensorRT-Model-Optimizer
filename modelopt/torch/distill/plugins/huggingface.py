# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ModelOpt plugin to train HuggingFace models with knowledge distillation."""

import torch
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType
from transformers import Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto


class KDTrainer(Trainer):
    """Distillation trainer for HuggingFace models."""

    def compute_loss(self, model, inputs, *args, **kwargs):
        """Compute loss for distillation.

        Change the training loss to distillation loss and keep the original validation loss.

        Args:
            model: The model to compute loss for.
            inputs: The inputs to the model.
        """
        if not model.training:
            _compute_loss_func = self.compute_loss_func
            self.compute_loss_func = None

        loss = super().compute_loss(model, inputs, *args, **kwargs)
        if not model.training:
            self.compute_loss_func = _compute_loss_func

        return loss

    def save_model(self, output_dir: str, export_student: bool = False, *args, **kwargs):
        """Dumps model and ModelOpt states to disk.

        Args:
            output_dir: The directory to save the model and ModelOpt states.
            export_student: Whether to export the student model.
        """
        model = self.accelerator.unwrap_model(self.model)
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_cfg
        ):
            cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                if export_student:
                    student_model = self.model.export()
                    # remove teacher model from state dict since FSDP forces
                    # expose_minimal_state_dict to be False
                    cpu_state_dict = {
                        k: v for k, v in cpu_state_dict.items() if "_teacher_model" not in k
                    }
                    student_model.save_pretrained(
                        output_dir,
                        state_dict=cpu_state_dict,
                        safe_serialization=self.args.save_safetensors,
                    )
                    self.processing_class.save_pretrained(output_dir)
                else:
                    self._save(output_dir, state_dict=cpu_state_dict)
                # ModelOpt state
                modelopt_state = mto.modelopt_state(model)
                modelopt_state["modelopt_state_dict"] = [
                    state
                    for state in modelopt_state["modelopt_state_dict"]
                    if "kd_loss" not in state and "export_student" not in state
                ]
                torch.save(modelopt_state, f"{output_dir}/modelopt_state.pth")

    def train(self, *args, **kwargs):
        """Train the model."""
        self.compute_loss_func = lambda *args, **kwargs: self.model.compute_kd_loss()
        return super().train(*args, **kwargs)


class LMLogitsLoss(mtd.LogitsDistillationLoss):
    """Logits loss for knowledge distillation."""

    def forward(self, out_student: CausalLMOutputWithPast, out_teacher: CausalLMOutputWithPast):
        """Forward pass for logits distillation loss.

        Args:
            out_student: The student model output.
            out_teacher: The teacher model output.
        """
        return super().forward(out_student.logits, out_teacher.logits)
