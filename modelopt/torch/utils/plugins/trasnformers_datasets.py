"""Processing large data to tokenize for pretraining."""

import warnings

import torch
import transformers
from datasets import load_dataset

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def _sharegpt_to_openai_messages(conversations: list[dict]):
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
    messages = []
    for msg in conversations:
        role = role_mapping[msg["from"]]
        content = msg["value"]
        messages.append({"role": role, "content": content})
    return messages


class LanguageDataset(torch.utils.data.Dataset):
    """LanguageDataset with on-the-fly tokenization over HuggingFace datasets.

    Args:
        tokenizer: The tokenizer to use.
        data_path: The path to the dataset.
        subset: The subset to use.
        split: The split to use.
        max_length: The maximum length of the sequence.
        num_shards: The number of shards to use.
        shard_index: The index of the shard to use.
        chat_template: The chat template to use.
        add_generation_prompt: Whether to add a generation prompt.
        json_key: The key to use to get the text from the dataset.
        answer_only_loss: Whether to use answer-only loss.
        streaming: Whether to use streaming.
    """

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
        add_generation_prompt: bool = False,
        json_key: str = "text",
        answer_only_loss: bool = False,
        streaming: bool = True,
    ):
        """Initialize the LanguageDataset."""
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError(
                "The tokenizer must be a transformers.PreTrainedTokenizerBase but got {}".format(
                    type(tokenizer)
                )
            )

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.subset = subset
        self.split = split
        self.max_length = max_length
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.add_generation_prompt = add_generation_prompt
        self.json_key = json_key
        self.answer_only_loss = answer_only_loss
        self.streaming = streaming
        self._post_process_tokenizer()

        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        else:
            self._post_process_chat_template()

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

        if self.answer_only_loss and "{% generation %}" not in self.tokenizer.chat_template:
            warnings.warn(
                "The chat template does not contain the {% generation %} tag. OVERWRITE answer_only_loss to False!"
            )
            self.answer_only_loss = False

        if self.tokenizer.padding_side != "right":
            warnings.warn(
                "The tokenizer padding side is {}. OVERWRITE to right!".format(
                    self.tokenizer.padding_side
                )
            )
            self.tokenizer.padding_side = "right"

        self._load_dataset()

    def __len__(self):
        return len(self._raw_samples)

    def __getitem__(self, index):
        index = index // self.num_shards

        if self.streaming:
            while index >= len(self._raw_samples):
                self._raw_samples.append(next(self._iterator))

        raw_sample = self._raw_samples[index]
        return self._process_and_pack_sample(raw_sample)

    def _post_process_tokenizer(self):
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token == "<|eot_id|>":  # nosec
                self.tokenizer.pad_token = "<|end_of_text|>"  # nosec
            else:
                raise ValueError("The tokenizer has no pad_token!")

    def _post_process_chat_template(self):
        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

    def _load_dataset(self):
        self._raw_samples = []
        self._stream_samples = load_dataset(
            self.data_path,
            self.subset,
            split=self.split,
            # num_proc=4,  # TODO: Make this configurable
            streaming=True,
        )
        self._steam_samples = self._stream_samples.shard(
            num_shards=self.num_shards, index=self.shard_index
        )
        self._iterator = iter(self._stream_samples)

    def _process_chat_sample(self, example):
        messages = example.get("messages", None)
        if messages is None:
            conversations = example.get("conversations", None)
            if conversations is None:
                raise ValueError(
                    "The sample must in either OpenAI messages format or ShareGPT conversations format."
                )
            else:
                messages = _sharegpt_to_openai_messages(conversations)

        tokenized_messages = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=self.add_generation_prompt,
            return_assistant_tokens_mask=self.answer_only_loss,
        )
        return tokenized_messages

    def _process_text_sample(self, text: str):
        tokenized_sample = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized_sample

    def _process_and_pack_sample(self, example):
        if not isinstance(example, dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))
        text = example.get(self.json_key, None)
        if isinstance(text, str):
            return self._process_text_sample(text)
        return self._process_chat_sample(example)


class VisionLanguageDataset(LanguageDataset):
    """VisionLanguageDataset with on-the-fly tokenization over HuggingFace datasets.

    Args:
        processor: The processor to use.
        data_path: The path to the dataset.
        subset: The subset to use.
        split: The split to use.
        max_length: The maximum length of the sequence.
        num_shards: The number of shards to use.
        shard_index: The index of the shard to use.
        chat_template: The chat template to use.
        add_generation_prompt: Whether to add a generation prompt.
        answer_only_loss: Whether to use answer-only loss.
        streaming: Whether to use streaming.
        local_image_path: The path to the local images.
    """

    def __init__(
        self,
        processor: transformers.ProcessorMixin,
        data_path: str,
        subset: str | None = None,
        split: str = "train",
        max_length: int = 4096,
        num_shards: int = 1,
        shard_index: int = 0,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        streaming: bool = True,
        local_image_path: str | None = None,
    ):
        """Initialize the VisionLanguageDataset."""
        if not isinstance(processor, transformers.ProcessorMixin):
            raise ValueError(
                "The processor must be a transformers.ProcessorMixin but got {}".format(
                    type(processor)
                )
            )

        self.processor = processor
        self.local_image_path = local_image_path
        if self.local_image_path is None:
            self.local_image_path = data_path + "/images"

        super().__init__(
            tokenizer=self.processor.tokenizer,
            data_path=data_path,
            subset=subset,
            split=split,
            max_length=max_length,
            num_shards=num_shards,
            shard_index=shard_index,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            answer_only_loss=answer_only_loss,
            streaming=streaming,
        )

    def _process_multimodal_sample(self, example):
        messages = example.get("messages", None)
        if messages is None:
            print(example)
            conversations = example.get("conversations", None)
            if conversations is None:
                raise ValueError(
                    "The sample must in either OpenAI messages format or ShareGPT conversations format."
                )
            else:
                messages = _sharegpt_to_openai_messages(conversations)

        for msg in messages:
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            for ctn in msg["content"]:
                if ctn["type"] == "image" and "path" in ctn:
                    ctn["path"] = self.local_image_path + "/" + ctn["path"]

        tokenized_messages = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=self.add_generation_prompt,
            return_assistant_tokens_mask=self.answer_only_loss,
        )
        return tokenized_messages

    def _process_and_pack_sample(self, example):
        if not isinstance(example, dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))

        return self._process_multimodal_sample(example)
