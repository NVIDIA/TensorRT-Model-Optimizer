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
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import Trainer, default_data_collator
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    get_lora_config,
    make_supervised_data_module,
    monkey_patch_training_step_to_fix_memory_leak,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.utils import print_rank_0

CUSTOM_QUANT_CFG = {
    "INT4_WEIGHT_INT8_ACTIVATIONS": {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
            "*lm_head*": {"enable": False},
            "default": {"enable": False},
        },
        "algorithm": "max",
    }
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


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
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add LoRA (Low-Rank Adaptation) adapter before training. When using real quantization, "
                "the LoRA adapter must be set, as quantized weights will be frozen during training."
            )
        },
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default="samsum",
        metadata={"help": "Specify the dataset.", "choices": ["samsum", "Daring-Anteater"]},
    )


@dataclass
class QuantizationArguments:
    quant_cfg: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
            "choices": mtq.config.choices | CUSTOM_QUANT_CFG.keys(),
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
                "Whether to compress the model weights after quantization. "
                "This is useful for reducing the model size."
            )
        },
    )


def get_metrics_with_perplexity(metrics):
    metrics = {"perplexity": float(torch.exp(torch.tensor(metrics["eval_loss"]))), **metrics}
    return metrics


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, QuantizationArguments)
    )
    model_args, training_args, data_args, quant_args = parser.parse_args_into_dataclasses()
    print_rank_0(f"arguments: {model_args}, {training_args}, {quant_args}")

    # Enable automatic save/load of modelopt state huggingface checkpointing
    # modelopt state will be saved automatically to "modelopt_state.pt"
    mto.enable_huggingface_checkpointing()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.do_sample = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_rank_0("Loading dataset...")
    data_module = make_supervised_data_module(dataset=data_args.dataset, tokenizer=tokenizer)

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Torch >= 2.4 throws an error if `use_reentrant` is not set explicitly
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer._move_model_to_device(model, trainer.args.device)

    if checkpoint is None:
        if quant_args.quant_cfg is not None:
            calib_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=default_data_collator,
            )

            if "AWQ" in quant_args.quant_cfg:
                print_rank_0(
                    "\n####\nAWQ calibration could take longer than other calibration methods. "
                    "Consider reducing calib_size to reduce calibration time.\n####\n"
                )

            def calibrate_loop(model):
                print_rank_0("Calibrating model...")
                for i, data in enumerate(calib_dataloader):
                    if i >= quant_args.calib_size // training_args.per_device_eval_batch_size:
                        break
                    data = {k: v.to(trainer.args.device) for k, v in data.items()}
                    model(**data)

            quant_cfg = (
                CUSTOM_QUANT_CFG[quant_args.quant_cfg]
                if quant_args.quant_cfg in CUSTOM_QUANT_CFG
                else getattr(mtq, quant_args.quant_cfg)
            )
            model = mtq.quantize(model, quant_cfg, calibrate_loop)
            torch.cuda.empty_cache()  # Lets make sure to free up the memory for training
    else:
        assert not training_args.lora, "Does not support LoRA resuming training yet!"

    # add lora adapter
    if training_args.lora:
        model.add_adapter(get_lora_config(), adapter_name="adapter")

    # compress model weights after lora adapter inserted to prevent training error
    if checkpoint is None and quant_args.compress:
        mtq.compress(model)

    # There could be GPU memory leak during QAT causing OOM. This is a workaround to fix it.
    monkey_patch_training_step_to_fix_memory_leak(trainer)

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics = get_metrics_with_perplexity(metrics)
        print_rank_0(f"Evaluation results: \n{metrics}")

    if training_args.do_train or quant_args.quant_cfg is not None:
        trainer.save_state()
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
