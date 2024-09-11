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

import gc
import types
from functools import partial
from typing import Dict

import datasets
import transformers
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import default_data_collator

IGNORE_INDEX = -100


class ConcatDataset(Dataset):
    def __init__(self, dataset, max_length=4096):
        self.dataset = dataset
        self.samples = []
        buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        for sample in self.dataset:
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            while len(next(iter(buffer.values()))) > max_length:
                self.samples.append({k: v[:max_length] for k, v in buffer.items()})
                buffer = {k: v[max_length:] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_preprocessed_samsum(tokenizer, split, max_length=4096):
    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(
            sample["summary"] + tokenizer.eos_token, add_special_tokens=False
        )
        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [IGNORE_INDEX] * len(prompt) + summary,
        }
        return sample

    dataset = datasets.load_dataset("samsum", split=split, trust_remote_code=True)
    prompt = "Summarize this dialog:\n{dialog}\n---\nSummary:\n"
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    dataset = ConcatDataset(dataset, max_length)
    return dataset


def get_daring_anteater(tokenizer: transformers.AutoTokenizer, split="train", max_length=4096):
    # sample = {
    #     'system': '{system message}',
    #     'conversations': [
    #         {'from': 'User', 'value': '{turn 1 user message}', 'label': None},
    #         {'from': 'Assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
    #         {'from': 'User', 'value': '{turn 2 user message}', 'label': None},
    #         {'from': 'Assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
    #     ],
    #     "mask": "User",
    #     "type": "VALUE_TO_TEXT",
    # }

    def process_and_tokenize(sample):
        conversations = sample["conversations"]
        all_input_ids = [tokenizer.bos_token_id]
        all_labels = [IGNORE_INDEX]

        for conversation in conversations:
            role = conversation["from"]
            input_ids = tokenizer.encode(conversation["value"] + "\n", add_special_tokens=False)
            labels = input_ids if role == "Assistant" else [IGNORE_INDEX] * len(input_ids)

            all_input_ids.extend(input_ids)
            all_labels.extend(labels)

            if len(all_input_ids) > max_length:
                break

        all_input_ids.append(tokenizer.eos_token_id)
        all_labels.append(IGNORE_INDEX)
        all_attention_mask = [1] * len(all_input_ids)

        cur_seq_length = len(all_input_ids)
        if cur_seq_length < max_length:
            pad_token = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            all_input_ids += [pad_token] * (max_length - cur_seq_length)
            all_attention_mask += [0] * (max_length - cur_seq_length)
            all_labels += [IGNORE_INDEX] * (max_length - cur_seq_length)

        return {
            "input_ids": all_input_ids[:max_length],
            "attention_mask": all_attention_mask[:max_length],
            "labels": all_labels[:max_length],
        }

    if hasattr(get_daring_anteater, "cached_dataset"):
        dataset = get_daring_anteater.cached_dataset
    else:
        dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
        dataset = dataset.map(process_and_tokenize, remove_columns=list(dataset.features))
        dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=42)
        get_daring_anteater.cached_dataset = dataset  # type: ignore[attr-defined]
    return dataset[split]


def make_supervised_data_module(
    dataset="samsum", tokenizer: transformers.PreTrainedTokenizer = None
) -> Dict:
    """Make dataset and collmtor for supervised fine-tuning."""
    if dataset == "samsum":
        train_dataset = get_preprocessed_samsum(tokenizer, "train", tokenizer.model_max_length)
        val_dataset = get_preprocessed_samsum(tokenizer, "validation", tokenizer.model_max_length)
    elif dataset == "Daring-Anteater":
        train_dataset = get_daring_anteater(tokenizer, "train", tokenizer.model_max_length)
        val_dataset = get_daring_anteater(tokenizer, "test", tokenizer.model_max_length)
    return dict(
        train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=default_data_collator
    )


def get_lora_config():

    return LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def monkey_patch_training_step_to_fix_memory_leak(trainer):
    def new_func(original_f_name, trainer, *args, **kwargs):
        gc.collect()
        return getattr(trainer, original_f_name)(*args, **kwargs)

    for f_name in ["training_step", "prediction_step", "_load_best_model"]:
        setattr(trainer, "_original_" + f_name, getattr(trainer, f_name))
        setattr(
            trainer, f_name, types.MethodType(partial(new_func, "_original_" + f_name), trainer)
        )
