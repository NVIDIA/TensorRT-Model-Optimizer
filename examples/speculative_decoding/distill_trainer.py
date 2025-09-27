# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from abc import abstractmethod

import torch
import torch.distributed as dist
from tqdm import tqdm

import modelopt.torch.opt as mto

mto.enable_huggingface_checkpointing()

# Hyperparameters for profiling
EPOCHS = 1
LOG_INTERVAL = 1
SAVE_INTERVAL = 20000
# VALIDATE_INTERVAL = 20

# We define the distill signal from teacher as the map of variable name to its shape and dtype.
DistillMetadata = dict[str, tuple[torch.Size, torch.dtype]]


class BaseDistillTrainer:
    """
    Base class for distillation trainer. Initalized and called on every rank.
    Args:
        rank: rank of the current process
        args: arguments
        teacher_step: teacher step function.
        student_step: student step function.
    """

    def __init__(self, rank, args, tokenizer, distill_metadata: DistillMetadata):
        self.rank = rank
        args.teacher_pgroup = dist.new_group(ranks=args.teacher_ranks)
        args.student_pgroup = dist.new_group(ranks=args.student_ranks)
        self.args = args
        self.tokenizer = tokenizer
        self.distill_metadata = distill_metadata

    def _print_model_placement(self, module):
        for name, param in module.named_parameters():
            print(f"(Rank {self.rank}) {name}  --->  {param.device} ")

    @property
    def current_rank_device(self):
        pass

    def _reset_all_mem_stats(self):
        torch.cuda.reset_max_memory_allocated(self.current_rank_device)

    def _print_mem_stats(self):
        max_mem = torch.cuda.max_memory_allocated(self.current_rank_device)
        print(f"GPU {self.current_rank_device}: Max memory allocated: {max_mem / 1024**3:.2f} GB")

    @abstractmethod
    def load_teacher_model(self):
        pass

    @abstractmethod
    def load_student_model(self):
        pass

    @abstractmethod
    def teacher_step(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def student_step(self, *args, **kwargs):
        pass

    def save_pretrained(self, path=None):
        if self.rank == self.args.student_ranks[0]:
            path = self.args.out_path if path is None else path
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Pretrained model saved to {path}")

    def _check_valid_message(self, message: dict[str, torch.Tensor]):
        # Check if keys and length match between message and distill_metadata
        if set(message.keys()) != set(self.distill_metadata.keys()):
            raise ValueError(
                f"Message keys: {set(message.keys())} \n"
                f"do not match expected keys {set(self.distill_metadata.keys())}"
            )
        if len(message) != len(self.distill_metadata):
            raise ValueError(
                f"Message length: {len(message)} \n"
                f"does not match expected {len(self.distill_metadata)}"
            )
        for k, v in message.items():
            if v.shape != self.distill_metadata[k][0] or v.dtype != self.distill_metadata[k][1]:
                raise ValueError(
                    f"Invalid message. {k} has shape {v.shape} and dtype {v.dtype}, \n"
                    f"expected {self.distill_metadata[k]}"
                )

    def _init_student_recv_buffer(self):
        self.student_recv_buffer = {
            k: torch.empty(v[0], device=self.current_rank_device, dtype=v[1])
            for k, v in self.distill_metadata.items()
        }

    def _recv_from_teacher(self):
        reqs = [
            dist.irecv(buffer, src=self.args.teacher_ranks[0])
            for buffer in self.student_recv_buffer.values()
        ]
        for req in reqs:
            req.wait()

    def _get_distill_kwargs(self):
        return {k: v.clone().detach() for k, v in self.student_recv_buffer.items()}

    def _send_to_student(self, teacher_outputs):
        if self.rank != self.args.teacher_ranks[0]:
            return
        # TODO: use broadcast
        assert len(teacher_outputs) == len(self.args.student_ranks), (
            f"Number of teacher outputs {len(teacher_outputs)} does not \
            match number of student ranks {len(self.args.student_ranks)}"
        )
        for s in self.args.student_ranks:
            self._check_valid_message(teacher_outputs[s])
            reqs = [dist.isend(buffer, dst=s) for buffer in teacher_outputs[s].values()]
            for req in reqs:
                req.wait()

    # def _validate_ar(self, steps=3, osl=20, num_samples=20):
    #     if self.rank != self.args.student_rank:
    #         return
    #     # Load MT-Bench prompts from HuggingFace
    #     ds = load_dataset("HuggingFaceH4/mt_bench_prompts")["train"]
    #     self.model.eval()
    #     self.model.to(self.args.student_device)
    #     ars = validate_ar(
    #         self.model, self.tokenizer, ds, steps, osl, num_samples, self.args.student_device
    #     )
    #     # Print results
    #     avg_ar = sum(ars) / len(ars)
    #     print("\n==== AR Validation Results on MT-Bench ====")
    #     print(f"Number of samples: {len(ars)}")
    #     print(f"Output Sequence Length: {osl}")
    #     print(f"Steps: {steps}")
    #     print(f"Average AR: {avg_ar:.4f}")
    #     self.model.train()

    def train(self, dataloader):
        """Main training entrance of the composed model."""
        self._reset_all_mem_stats()

        if self.rank in self.args.student_ranks:
            import wandb

            wandb.login()

            with wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                config={"epochs": EPOCHS, "lr": self.args.lr, "batch_size": self.args.batch_size},
            ) as run:
                self.model, self.optimizer = self.load_student_model()
                self._init_student_recv_buffer()
                wandb.watch(self.model, log="all")

                for epoch in range(EPOCHS):
                    pbar = tqdm(dataloader)
                    for i, batch in enumerate(pbar):
                        global_step = epoch * len(dataloader) + i
                        inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                        self._recv_from_teacher()
                        loss, train_acc = self.student_step(inputs, **self._get_distill_kwargs())
                        pbar.set_description(f"Epoch {epoch} Loss:{loss} Acc:{train_acc}")

                        if global_step % LOG_INTERVAL == 0:
                            run.log(
                                {
                                    "loss": loss,
                                    "train_acc_step0": train_acc[0],
                                    "train_acc_step1": train_acc[1],
                                    "train_acc_step2": train_acc[2],
                                    "train_acc_step3": train_acc[3],
                                },
                                step=global_step,
                            )

                        # This is not working for some reason.
                        # if global_step > 0 and global_step % VALIDATE_INTERVAL == 0:
                        #     self._validate_ar()

                        if global_step > 0 and global_step % SAVE_INTERVAL == 0:
                            self.save_pretrained(
                                f"{self.args.out_path}/epoch_{epoch}_step_{global_step}"
                            )

        else:
            self.model = self.load_teacher_model()
            # Inference Loop
            for epoch in range(EPOCHS):
                for i, batch in enumerate(dataloader):
                    global_step = epoch * len(dataloader) + i
                    inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                    inputs["position_ids"] = None
                    with torch.inference_mode():
                        teacher_outputs = self.teacher_step(self.model, inputs)
                        self._send_to_student(teacher_outputs)

        self._print_mem_stats()
        # Makesure all processes finished before destroy.
        dist.barrier()
        # clean up processess
        dist.destroy_process_group()
