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

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
from eagle_utils import make_eagle_supervised_data_module
from medusa_utils import make_medusa_supervised_data_module
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.utils import calibrate_frequent_vocab
from modelopt.torch.utils import print_rank_0

torch.manual_seed(0)
mto.enable_huggingface_checkpointing()


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = True
    draft_vocab_cache_dir: str = field(
        default="draft_vocab_cache",
        metadata={"help": "Path to the d2t cache directory."},
    )
    calibrate_size: int = field(
        default=None,
        metadata={"help": "Size of the calibration data. If None, use entire training set."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    training_seq_len: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    mode: Literal["eagle", "medusa"] = "medusa"


@dataclass
class MedusaArguments:
    medusa_num_heads: int | None = field(default=1)
    medusa_num_layers: int | None = field(default=1)


@dataclass
class EagleArguments:
    eagle_num_layers: int | None = field(default=1)
    use_input_layernorm_in_first_layer: bool | None = field(default=True)
    use_last_layernorm: bool | None = field(default=True)
    use_aux_hidden_state: bool | None = field(default=True)
    draft_vocab_size: int | None = field(default=32000)


def train():
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            MedusaArguments,
            EagleArguments,
        )
    )
    model_args, data_args, training_args, medusa_args, eagle_args = (
        parser.parse_args_into_dataclasses()
    )
    print_rank_0(f"arguments: {model_args}, {training_args}, {medusa_args}, {eagle_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint:
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, torch_dtype="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.training_seq_len,
        )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{%- for message in messages %}"
                "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
                "{%- endfor %}"
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if training_args.mode == "medusa":
            config = {
                "medusa_num_heads": medusa_args.medusa_num_heads,
                "medusa_num_layers": medusa_args.medusa_num_layers,
            }
            mtsp.convert(model, [("medusa", config)])
        elif training_args.mode == "eagle":
            config = {
                "eagle_num_layers": eagle_args.eagle_num_layers,
                "use_input_layernorm_in_first_layer": eagle_args.use_input_layernorm_in_first_layer,
                "use_last_layernorm": eagle_args.use_last_layernorm,
                "use_aux_hidden_state": eagle_args.use_aux_hidden_state,
                "draft_vocab_size": eagle_args.draft_vocab_size,
            }

            mtsp.convert(model, [("eagle", config)])

            if eagle_args.draft_vocab_size > 0 and (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            ):
                model_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))

                vocab_cache_path = os.path.join(
                    data_args.draft_vocab_cache_dir, model_name, "d2t.pt"
                )
                if os.path.exists(vocab_cache_path):
                    vocab_cache = torch.load(vocab_cache_path)
                    if len(vocab_cache) == eagle_args.draft_vocab_size:
                        model.eagle_module.d2t = vocab_cache
                        print_rank_0(f"Loaded draft vocab cache from {vocab_cache_path}.")
                else:
                    print_rank_0(
                        "No matching draft vocab cache found, calibrating vocab using training set..."
                    )
                    with open(data_args.data_path) as f:
                        calibrate_conversations = [json.loads(line)["conversations"] for line in f]
                        if data_args.calibrate_size:
                            calibrate_conversations = calibrate_conversations[
                                : data_args.calibrate_size
                            ]
                        calibrate_conversations = [
                            item for sublist in calibrate_conversations for item in sublist
                        ]

                    model.eagle_module.d2t = calibrate_frequent_vocab(
                        tokenizer, calibrate_conversations, eagle_args.draft_vocab_size
                    )
        else:
            raise Exception(f"{training_args.mode} is not supported!")

    print_rank_0("Loading dataset...")
    if training_args.mode == "medusa":
        data_module = make_medusa_supervised_data_module(tokenizer, data_args)
    elif training_args.mode == "eagle":
        data_module = make_eagle_supervised_data_module(tokenizer, data_args)

    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    trainer._move_model_to_device(model, trainer.args.device)

    # Manually enable this to return loss in eval
    trainer.can_return_loss = True
    # Make sure label_smoother is None
    assert trainer.label_smoother is None, (
        "label_smoother is not supported in speculative decoding!"
    )

    print_rank_0("Start training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        print_rank_0(f"Evaluation results: \n{metrics}")


if __name__ == "__main__":
    train()
