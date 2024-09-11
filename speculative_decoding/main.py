# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from utils import make_supervised_data_module

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.utils import print_rank_0


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)


@dataclass
class MedusaArguments:
    medusa_only_heads: Optional[bool] = field(default=True)
    medusa_num_heads: Optional[int] = field(default=1)
    medusa_num_layers: Optional[int] = field(default=1)
    medusa_lm_head: Optional[str] = field(default="")


def get_metrics_with_perplexity(metrics):
    metrics = {"perplexity": float(torch.exp(torch.tensor(metrics["eval_loss"]))), **metrics}
    return metrics


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, MedusaArguments)
    )
    model_args, data_args, training_args, medusa_args = parser.parse_args_into_dataclasses()
    print_rank_0(f"arguments: {model_args}, {training_args}, {medusa_args}")

    mto.enable_huggingface_checkpointing()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    modelopt_state_path = os.path.join(training_args.output_dir, "modelopt_state.pt")
    if os.path.isfile(modelopt_state_path):
        print_rank_0(f"Loading modelopt state from {modelopt_state_path}")
        modelopt_state = torch.load(modelopt_state_path)
        mto.restore_from_modelopt_state(model, modelopt_state)
    else:
        config = {
            "medusa_num_heads": medusa_args.medusa_num_heads,
            "medusa_num_layers": medusa_args.medusa_num_layers,
        }
        mtsp.convert(model, [("medusa", config)])
    if medusa_args.medusa_lm_head:
        mtsp.plugins.transformers.load_medusa_head(model, medusa_args.medusa_lm_head)
    model.generation_config.do_sample = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_rank_0("Loading dataset...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer._move_model_to_device(model, trainer.args.device)
    mtsp.plugins.transformers.replace_medusa_compute_loss(
        trainer, medusa_only_heads=medusa_args.medusa_only_heads
    )

    print_rank_0("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
