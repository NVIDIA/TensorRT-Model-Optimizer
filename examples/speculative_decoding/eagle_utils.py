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
from typing import Any

import torch
import transformers
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def preprocess(examples, tokenizer):
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "loss_mask": [],
        "labels": [],
    }
    roles = ["user", "assistant"]
    for i in range(len(examples)):
        messages = []
        source = examples[i]["conversations"]
        if source[0]["from"].lower() != "user":
            # Skip the first one if it is not from human
            source = source[1:]
        for j, sentence in enumerate(source):
            assert sentence["from"].lower() == roles[j % 2], f"{i}"
            messages.append({"role": sentence["from"].lower(), "content": sentence["value"]})
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        output = tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = output.input_ids[0]
        attention_mask = output.attention_mask[0]
        offset_mapping = output.offset_mapping[0]
        loss_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_TOKEN_ID)

        for turn in messages:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = conversation.index(content.strip())
                stop = start + len(content)
                indices = []
                for tok_index, (tok_start, tok_stop) in enumerate(offset_mapping):
                    if tok_start >= start and tok_stop <= stop:
                        indices.append(tok_index)
                labels[indices] = input_ids[indices]
                loss_mask[indices] = 1

        # Shift loss_mask and labels to the left by 1 token
        loss_mask = torch.cat([loss_mask[1:], torch.zeros(1, dtype=loss_mask.dtype)])
        labels = torch.cat([labels[1:], torch.tensor([IGNORE_TOKEN_ID], dtype=labels.dtype)])

        new_examples["input_ids"].append(input_ids)
        new_examples["attention_mask"].append(attention_mask)
        new_examples["loss_mask"].append(loss_mask)
        new_examples["labels"].append(labels)

    return new_examples


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        print_rank_0("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.loss_mask = data_dict["loss_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            loss_mask=self.loss_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        print_rank_0("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            loss_mask=ret["loss_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_eagle_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    print_rank_0("Loading data...")

    if data_args.data_path.endswith("jsonl"):
        with open(data_args.data_path, "r") as f:
            data_json = [json.loads(line) for line in f]
    else:
        data_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(data_json[: int(len(data_json) * 0.95)], tokenizer=tokenizer)
    eval_dataset = dataset_cls(data_json[int(len(data_json) * 0.95) :], tokenizer=tokenizer)

    data_collator = DataCollatorWithPadding()

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class DataCollatorWithPadding:
    def paddingtensor2d(self, intensors, length):
        n, dim = intensors.shape
        padding_tensor = torch.zeros(length - n, dim, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor))
        return outtensors

    def paddingtensor(self, intensors, length):
        padding_tensor = torch.zeros(length - intensors.shape[0], dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor))
        return outtensors

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(item["input_ids"].shape[0] for item in features)
        batch_input_ids = torch.stack(
            [self.paddingtensor(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.stack(
            [self.paddingtensor(item["attention_mask"], max_length) for item in features]
        )
        batch_loss_mask = torch.stack(
            [self.paddingtensor(item["loss_mask"], max_length) for item in features]
        )

        batch_labels = torch.stack(
            [self.paddingtensor(item["labels"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "labels": batch_labels,
        }

        return batch
