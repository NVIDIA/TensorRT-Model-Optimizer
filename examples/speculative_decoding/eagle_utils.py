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
from pathlib import Path
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

        # Detect format: either role/content or from/value
        def get_role_content(item):
            if "role" in item and "content" in item:
                return item["role"], item["content"]
            elif "from" in item and "value" in item:
                return item["from"], item["value"]
            else:
                raise ValueError(f"Unknown conversation format: {item}")

        first_role, _ = get_role_content(source[0])
        if first_role.lower() != "user":
            # Skip the first one if it is not from human
            source = source[1:]
        for j, sentence in enumerate(source):
            role, content = get_role_content(sentence)
            assert role.lower() == roles[j % 2], f"{i}"
            messages.append({"role": role.lower(), "content": content})
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
        super().__init__()

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
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
            "loss_mask": self.loss_mask[i],
        }


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
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
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["labels"][0],
            "attention_mask": ret["attention_mask"][0],
            "loss_mask": ret["loss_mask"][0],
        }
        self.cached_data_dict[i] = ret

        return ret


class OfflineSupervisedDataset(Dataset):
    """Lazy offline dataset for supervised fine-tuning.

    This dataset loads data on-the-fly from pre-processed .pt data files as well as
    input conversations in JSON format.

    Args:
        data_entries (list): A list of tuples (raw_data_example, file_path).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, data_entries, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        print_rank_0("Formatting inputs...Skip in offline mode")
        self.tokenizer = tokenizer
        self.data_entries = data_entries

        # Does not cache the hidden states, as those have an extremely large memory footprint.
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # Load the conversational data, using the cache
        raw_data, offline_file_path = self.data_entries[i]
        if i in self.cached_data_dict:
            preprocessed_base = self.cached_data_dict[i]
        else:
            ret = preprocess([raw_data], self.tokenizer)
            preprocessed_base = {
                "input_ids": ret["input_ids"][0],
                "labels": ret["labels"][0],
                "attention_mask": ret["attention_mask"][0],
                "loss_mask": ret["loss_mask"][0],
            }
            self.cached_data_dict[i] = preprocessed_base

        # Extend the data sample with the hidden states from the .pt file
        max_length = self.tokenizer.model_max_length
        offline_data = torch.load(offline_file_path)
        offline_data["input_ids"] = offline_data["input_ids"][:max_length]
        offline_data["hidden_states"] = offline_data["hidden_states"][:max_length, :]
        offline_data["aux_hidden_states"] = offline_data["aux_hidden_states"][:max_length, :]

        # Make sure the input_ids have the same shape
        if preprocessed_base["input_ids"].shape != offline_data["input_ids"].shape:
            msg = f"""Input IDs from offline data do not match the preprocessed input IDs
                                for offline data sample at {offline_file_path}."""
            raise ValueError(msg)

        ret = {**preprocessed_base}  # Shallow copy so we don't accidentally modify the cache
        ret["input_ids"] = offline_data["input_ids"]
        ret["kwargs"] = {
            "base_model_outputs": {
                "base_model_hidden_states": offline_data["hidden_states"],
                "aux_hidden_states": offline_data["aux_hidden_states"],
            }
        }
        return ret


def make_eagle_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, use_offline_training: bool
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    # Load the conversations from the source file
    with open(data_args.data_path) as f:
        if data_args.data_path.endswith("jsonl"):
            data_json = [json.loads(line) for line in f]
        else:
            data_json = json.load(f)

    if use_offline_training:
        print_rank_0("Loading pre-processed data for offline training...")
        dataset_cls = OfflineSupervisedDataset

        # Glob for all .pt files in the data_path directory
        assert data_args.offline_data_path is not None, (
            "offline_data_path must be provided for offline training."
        )
        offline_data_path = Path(data_args.offline_data_path)
        all_files = {str(p) for p in offline_data_path.glob("*.pt")}
        if not all_files:
            raise ValueError(f"No .pt files found in {data_args.offline_data_path}")

        # Filter to conversations that exist in the offline data and in the provided json
        valid_entries = []
        for idx, entry in enumerate(data_json):
            conv_id = entry.get("conversation_id")
            if conv_id is None:
                conv_id = entry.get("id")
            if conv_id is None:
                conv_id = "{:08d}".format(idx)
            file_path = str(offline_data_path / f"{conv_id}.pt")
            if file_path in all_files:
                valid_entries.append((entry, file_path))

        if len(valid_entries) == 0:
            msg = """No valid files found in the offline data path that match the conversation IDs
            in the provided data json. Please ensure that the offline data path is correct and
            contains .pt files named after the conversation IDs, and that the input conversations
            json has the correct format (with 'conversation_id' or 'id' fields)."""
            raise ValueError(msg)
        elif len(valid_entries) < len(data_json):
            print_rank_0(
                f"Warning: Only {len(valid_entries)} out of {len(data_json)} conversations"
                " have corresponding .pt files in the offline data path. Continuing..."
            )

        num_train = int(len(valid_entries) * 0.95)
        train_dataset = dataset_cls(valid_entries[:num_train], tokenizer=tokenizer)
        eval_dataset = dataset_cls(valid_entries[num_train:], tokenizer=tokenizer)

        data_collator = DataCollatorForOffline()
    else:
        print_rank_0("Loading input conversations...")
        dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset

        train_dataset = dataset_cls(data_json[: int(len(data_json) * 0.95)], tokenizer=tokenizer)
        eval_dataset = dataset_cls(data_json[int(len(data_json) * 0.95) :], tokenizer=tokenizer)

        data_collator = DataCollatorWithPadding()

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


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


class DataCollatorForOffline(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        base_batch = super().__call__(features)
        if "kwargs" not in features[0]:
            raise ValueError("No kwargs found in batch features. Offline data required.")

        features = [item["kwargs"]["base_model_outputs"] for item in features]
        max_hs_length = max(item["base_model_hidden_states"].shape[0] for item in features)

        batch_hidden_states = torch.stack(
            [
                self.paddingtensor2d(item["base_model_hidden_states"], max_hs_length)
                for item in features
            ]
        )
        batch_aux_hidden_states = torch.stack(
            [self.paddingtensor2d(item["aux_hidden_states"], max_hs_length) for item in features]
        )

        batch = {
            **base_batch,
            "base_model_outputs": {
                "base_model_hidden_states": batch_hidden_states,
                "aux_hidden_states": batch_aux_hidden_states,
            },
        }

        return batch
