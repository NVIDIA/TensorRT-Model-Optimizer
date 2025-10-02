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
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

mto.enable_huggingface_checkpointing()

# Hyperparameters for profiling
EPOCHS = 1
LOG_INTERVAL = 100
SAVE_INTERVAL = 20000
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DRAFT_VOCAB_SIZE = 32000
# VALIDATE_INTERVAL = 20

# Shape and dtype description of the distillation signal
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
        """Return a copy of received buffer for student training."""
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
                self.model, self.optimizer, self.scheduler = self.load_student_model()
                self._init_student_recv_buffer()
                wandb.watch(self.model, log="all")

                for epoch in range(EPOCHS):
                    pbar = (
                        tqdm(dataloader) if self.rank == self.args.student_ranks[0] else dataloader
                    )
                    for i, batch in enumerate(pbar):
                        global_step = epoch * len(dataloader) + i
                        inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                        self._recv_from_teacher()
                        loss, train_acc = self.student_step(inputs, **self._get_distill_kwargs())

                        if self.rank != self.args.student_ranks[0]:
                            continue

                        pbar.set_description(f"Epoch {epoch} Loss:{loss} Acc:{train_acc}")
                        if global_step % LOG_INTERVAL == 0:
                            run.log(
                                {
                                    "loss": loss,
                                    "train_acc_step0": train_acc[0],
                                    "train_acc_step1": train_acc[1],
                                    "train_acc_step2": train_acc[2],
                                    "train_acc_step3": train_acc[3],
                                    "lr": self.optimizer.param_groups[0]["lr"],
                                },
                                step=global_step,
                            )
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


class EagleTPTrainer(BaseDistillTrainer):
    @property
    def current_rank_device(self):
        if self.rank in self.args.student_ranks:
            return self.args.student_devices[self.rank]
        else:
            return self.args.teacher_devices[self.rank - len(self.args.student_ranks)]

    def load_teacher_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            tp_plan="auto",
            device_mesh=DeviceMesh.from_group(self.args.teacher_pgroup, "cuda"),
        )
        self.args.eagle_config["eagle_architecture_config"].update(
            {
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
                "draft_vocab_size": DRAFT_VOCAB_SIZE,
            }
        )
        mtsp.convert(model, [("eagle", self.args.eagle_config)])
        model.eval()
        self._print_model_placement(model)
        return model

    def load_student_model(self):
        """Load student model on a single device and keep needed modules from teacher."""
        # Load to CPU first to avoid OOM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype="auto", device_map="cpu"
        )
        # Hidden size and vocab size must match base model
        self.args.eagle_config["eagle_architecture_config"].update(
            {
                "hidden_size": model.config.hidden_size,
                "vocab_size": model.config.vocab_size,
                "draft_vocab_size": DRAFT_VOCAB_SIZE,
            }
        )
        mtsp.convert(
            model,
            [("eagle", self.args.eagle_config)],
        )
        if model.config.vocab_size > DRAFT_VOCAB_SIZE:
            model_name = os.path.basename(os.path.normpath(MODEL_PATH))
            vocab_cache_path = os.path.join("draft_vocab_cache", model_name, "d2t.pt")
            try:
                vocab_cache = torch.load(vocab_cache_path)
                assert len(vocab_cache) == DRAFT_VOCAB_SIZE
                model.eagle_module.d2t = vocab_cache
                print(f"Loaded draft vocab cache from {vocab_cache_path}.")
            except Exception as e:
                raise e

        # TODO:copy needed modules and del the rest
        model.model._modules.pop("layers")
        model.to(self.current_rank_device)

        model.train()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.current_rank_device],
            process_group=self.args.student_pgroup,
            find_unused_parameters=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=117380
        )
        self._print_model_placement(model)
        return model, optimizer, scheduler

    def teacher_step(self, model, inputs):
        base_model_hidden_states, base_model_logits, _, _ = model._base_model_forward(
            **inputs,
            freeze_base_model=True,
            past_key_values=None,
        )
        # aux_hidden_states could be on multiple devices. Gather them and cat.
        aux_hidden_states = torch.cat(
            [t.to(base_model_logits.device) for t in model.pop_aux_hidden_states()], dim=-1
        )
        base_model_hidden_states = base_model_hidden_states.chunk(len(self.args.student_ranks))
        base_model_logits = base_model_logits.chunk(len(self.args.student_ranks))
        aux_hidden_states = aux_hidden_states.chunk(len(self.args.student_ranks))

        return [
            {
                "base_model_hidden_states": base_model_hidden_states[i],
                "aux_hidden_states": aux_hidden_states[i],
                "base_model_logits": base_model_logits[i],
            }
            for i in range(len(self.args.student_ranks))
        ]

    def student_step(
        self,
        inputs,
        base_model_hidden_states,
        aux_hidden_states,
        base_model_logits,
    ):
        self.optimizer.zero_grad()
        # Second stage forward using the unified model
        inputs = {k: v.chunk(len(self.args.student_ranks))[self.rank] for k, v in inputs.items()}
        output = self.model(
            **inputs,
            # providing base model outputs to bypass the base model forward.
            base_model_outputs={
                "base_model_hidden_states": base_model_hidden_states,
                "aux_hidden_states": aux_hidden_states.clone().detach(),
                "base_model_logits": base_model_logits.clone().detach(),
            },
        )
        loss = output.loss
        # print(f"Rank {self.rank} loss: {loss.item()}")
        train_acc = output.train_acc

        # Backward
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return round(loss.item(), 3), train_acc


# class EagleMPTrainer(EagleTPTrainer, BaseDistillTrainer):
#     @property
#     def current_rank_devices(self):
#         if self.rank == self.args.student_rank:
#             return [self.args.student_device]
#         else:
#             return self.args.teacher_devices

#     def load_teacher_model(self):
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             torch_dtype="auto",
#             device_map="sequential",
#             max_memory=dict.fromkeys(
#                 self.args.teacher_devices, "999GiB"
#             ),  # To use only given devices
#         )
#         self.args.eagle_config["eagle_architecture_config"].update(
#             {
#                 "hidden_size": model.config.hidden_size,
#                 "vocab_size": model.config.vocab_size,
#                 "draft_vocab_size": DRAFT_VOCAB_SIZE,
#             }
#         )
#         mtsp.convert(model, [("eagle", self.args.eagle_config)])

#         if model.config.vocab_size > DRAFT_VOCAB_SIZE:
#             model_name = os.path.basename(os.path.normpath(MODEL_PATH))
#             vocab_cache_path = os.path.join("draft_vocab_cache", model_name, "d2t.pt")
#             try:
#                 vocab_cache = torch.load(vocab_cache_path)
#                 assert len(vocab_cache) == DRAFT_VOCAB_SIZE
#                 model.eagle_module.d2t = vocab_cache
#                 print(f"Loaded draft vocab cache from {vocab_cache_path}.")
#             except Exception as e:
#                 raise e

#         model.eval()
#         self._print_model_placement(model)
#         return model
