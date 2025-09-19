"""Processing large data to tokenize for pretraining."""

import torch
import transformers
from datasets import load_dataset

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def _sharegpt_to_openai_conversations(data):
    role_mapping = {
        "user": "user",
        "User": "user",
        "human": "user",
        "assistant": "assistant",
        "Assistant": "assistant",
        "gpt": "assistant",
        "system": "system",
        "System": "system",
    }
    processed_data = {"conversations": []}
    for msg in data["conversations"]:
        role = role_mapping[msg["from"]]
        content = msg["value"]
        processed_data["conversations"].append({"role": role, "content": content})
    return processed_data


class LanguageDataset(torch.utils.data.Dataset):
    """Tem."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        data_path: str,
        subset: str | None = None,
        split: str = "train",
        max_length: int = 4096,
        num_shards: int = 1,
        shard_index: int = 0,
        chat_template: str | None = None,
    ):
        """Tem."""
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_shards = num_shards
        self.shard_index = shard_index

        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        else:
            self._post_process_chat_template()

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

        self._raw_samples = load_dataset(
            self.data_path,
            self.subset,
            split=self.split,
            num_proc=4,  # TODO: Make this configurable
            desc="Loading dataset...",
        )
        self._raw_samples = self._raw_samples.shard(
            num_shards=self.num_shards, index=self.shard_index
        )

    def __len__(self):
        return len(self._raw_samples)

    def __getitem__(self, index):
        index = index // self.num_shards
        index = index % len(self._raw_samples)
        return self._process_and_pack_example(self._raw_samples[index])

    def _post_process_chat_template(self):
        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

    def _pre_process_example(self, example):
        if not isinstance(example, dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))

    def _process_and_pack_example(self, example):
        pass


# class VisionLanguageDataset(LanguageDataset):
