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

import logging
import os
from dataclasses import dataclass

import datasets
import torch
import torch.distributed
import transformers
from accelerate.logging import get_logger
from transformers import AutoTokenizer
from trl import SFTTrainer

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
from modelopt.torch.distill.plugins.huggingface import KDTrainer, LMLogitsLoss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelArguments:
    teacher_name_or_path: str | None = None
    student_name_or_path: str | None = None
    single_model: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_train: bool = True
    do_eval: bool = True
    save_strategy: str = "no"
    max_length: int = 1024
    optim: str = "adamw_torch"
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    dataloader_drop_last: bool = True
    dataset_num_proc: int = 8
    bf16: bool = True
    tf32: bool = True


def llama_text_format_func(sample):
    p, q, r = sample["system_prompt"], sample["question"], sample["response"]
    if not p:
        return f"<s>[INST] {q}[/INST]\n{r}</s>"
    else:
        return f"<s>[INST] <<SYS>>{p}<</SYS>>\n{q}[/INST]\n{r}</s>"


class KDSFTTrainer(SFTTrainer, KDTrainer):
    pass


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Enable automatic save/load of modelopt state huggingface checkpointing
    # modelopt state will be saved automatically to "modelopt_state.pth"
    mto.enable_huggingface_checkpointing()

    # Set total batch size across all ranks to equal 64
    total_batch_size = 64
    num_accum_steps = total_batch_size / (
        training_args.per_device_train_batch_size * torch.distributed.get_world_size()
    )
    if not num_accum_steps.is_integer():
        raise ValueError(
            f"`per_device_train_batch_size` * `world_size` must be a factor of {total_batch_size}"
        )
    training_args.gradient_accumulation_steps = int(num_accum_steps)
    logger.info(
        f"Using {int(num_accum_steps)} grad accumulation steps for effective batchsize of {total_batch_size}."
    )

    logger.info("Loading dataset...")
    dset = datasets.load_dataset("Open-Orca/OpenOrca", split="train")
    dset_splits = dset.train_test_split(train_size=25600, test_size=1700, seed=420)
    dset_train, dset_eval = dset_splits["train"], dset_splits["test"]
    logger.info("Dataset loaded.")

    logger.info("Loading tokenizer...")
    model_path = model_args.teacher_name_or_path or model_args.student_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    if model_args.single_model:
        logger.info("Loading single model only...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16 if training_args.bf16 else None
        )
        logger.info("Model loaded.")
    else:
        logger.info("Loading student model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.student_name_or_path, dtype=torch.bfloat16 if training_args.bf16 else None
        )
        logger.info("Student loaded.")
        # Load checkpoint
        logger.info("Loading teacher model and converting to Distillation model...")
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_name_or_path, dtype=torch.bfloat16 if training_args.bf16 else None
        )
        kd_config = {
            "teacher_model": teacher_model,
            "criterion": LMLogitsLoss(),
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])
        logger.info("Models converted.")

    # Fix problematic settings that logger.info excessive warnings
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    # Trainer
    trainer_cls = SFTTrainer if model_args.single_model else KDSFTTrainer
    trainer = trainer_cls(
        model,
        training_args,
        train_dataset=dset_train,
        eval_dataset=dset_eval,
        formatting_func=llama_text_format_func,
        processing_class=tokenizer,
    )

    # Do training
    if training_args.do_train:
        logger.info("Beginning training...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("Training done.")

    # Do evaluation
    if training_args.do_eval:
        logger.info("Evaluating...")
        eval_results = trainer.evaluate()
        logger.info(eval_results)
        logger.info("Evaluation complete.")

    # Save checkpoint
    logger.info("Saving checkpoint...")
    trainer.save_state()
    kwargs = {"export_student": True} if not model_args.single_model else {}
    trainer.save_model(trainer.args.output_dir, **kwargs)
    logger.info("Checkpoint saved.")


if __name__ == "__main__":
    train()
