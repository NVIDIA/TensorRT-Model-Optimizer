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

from typing import Dict

import datasets
import transformers
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

    dataset = datasets.load_dataset("samsum", split=split)
    prompt = "Summarize this dialog:\n{dialog}\n---\nSummary:\n"
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    dataset = ConcatDataset(dataset, max_length)
    return dataset


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collmtor for supervised fine-tuning."""
    train_dataset = get_preprocessed_samsum(tokenizer, "train", tokenizer.model_max_length)
    val_dataset = get_preprocessed_samsum(tokenizer, "validation", tokenizer.model_max_length)
    return dict(
        train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=default_data_collator
    )
