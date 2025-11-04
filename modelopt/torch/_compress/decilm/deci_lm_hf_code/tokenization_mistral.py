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

# Based on https://github.com/vllm-project/vllm/blob/739e03b3449a7f3b0a81ebc30b9555305d914e2d/vllm/transformers_utils/tokenizers/mistral.py
# mypy: ignore-errors

import os
import re
import sys
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

if TYPE_CHECKING:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

logger = logging.get_logger(__name__)


def _called_from_vllm() -> bool:
    frame = sys._getframe(1)
    while frame:
        mod = frame.f_globals.get("__name__", "")
        if mod == "vllm" or mod.startswith("vllm."):
            return True
        frame = frame.f_back
    return False


class HFAdaptedMistralTokenizer(PreTrainedTokenizer):
    """
    In order to save the tokenizer, do the following:
    ```
    # from <path_to_tokenization_mistral> import HFAdaptedMistralTokenizer
    # from mistral_common.tokens.tokenizers.base import SpecialTokens
    HFAdaptedMistralTokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer = HFAdaptedMistralTokenizer("<PATH_TO_TEKKEN_JSON>", chat_template="dummy")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [v.value for _, v in SpecialTokens.__members__.items()]}
    )
    tokenizer.save_pretrained("<PATH_TO_SAVE>")
    ```
    """

    vocab_files_names = {"path_indicator": "tokenizer_config.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        path_indicator: str,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        path_indicator: Path = Path(path_indicator)
        if path_indicator.name == "tokenizer_config.json":
            path_indicator = path_indicator.parent
        if path_indicator.is_dir():
            tokenizer_file_name = _find_tokenizer_file(os.listdir(path_indicator))
            tokenizer_file = str(path_indicator / tokenizer_file_name)
        else:
            tokenizer_file = path_indicator
        self._mistral_tokenizer_path = str(tokenizer_file)

        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MistralTokenizer

        self._mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)
        self._instruct_tokenizer = self._mistral_tokenizer.instruct_tokenizer

        # Copied from https://github.com/patrickvonplaten/vllm/blob/6cca3d8c330e169bbf386561c441ca5f3879cf85/vllm/transformers_utils/tokenizers/mistral.py
        self.version: int = int(
            self._instruct_tokenizer.tokenizer.version.value.split("v")[-1].split("m")[0]
        )

        tokenizer_ = self._instruct_tokenizer.tokenizer
        from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer

        self.is_tekken = isinstance(tokenizer_, Tekkenizer)
        from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer

        self.is_spm = isinstance(tokenizer_, SentencePieceTokenizer)
        if self.is_tekken:
            # Make sure special tokens will not raise
            tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE
        elif self.is_spm:
            pass
        else:
            raise TypeError(f"Unsupported tokenizer: {type(tokenizer_)}")

        self._vocab = tokenizer_.vocab()
        # Convert to a Dict[str, int] to match protocol, but this is a lossy
        # conversion. There may be multiple token ids that decode to the same
        # string due to partial UTF-8 byte sequences being converted to �
        self._vocab_dict = {token: idx for idx, token in enumerate(self._vocab)}
        self._tokenizer = tokenizer_
        self._max_token_id = self.vocab_size - 1
        self.vocab = self._vocab_dict

        bos_token = (
            bos_token
            if bos_token
            else AddedToken(
                self._tokenizer._vocab[self._tokenizer.bos_id],
                normalized=False,
                special=True,
            )
        )
        eos_token = (
            eos_token
            if eos_token
            else AddedToken(
                self._tokenizer._vocab[self._tokenizer.eos_id],
                normalized=False,
                special=True,
            )
        )
        unk_token = (
            unk_token
            if unk_token
            else AddedToken(
                self._tokenizer._vocab[self._tokenizer.unk_id],
                normalized=False,
                special=True,
            )
        )
        pad_token = (
            pad_token
            if pad_token
            else AddedToken(
                self._tokenizer._vocab[self._tokenizer.pad_id],
                normalized=False,
                special=True,
            )
        )

        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token

        self._in_vllm = _called_from_vllm()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self._tokenizer.n_words

    def get_vocab(self):
        """Returns vocab as a dict"""
        return self._vocab_dict

    def tokenize(
        self,
        text: str,
        pair: str | None = None,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[str]:
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        if add_special_tokens is None:
            bos = self._add_bos_token
            eos = self._add_eos_token
        else:
            bos = add_special_tokens
            eos = add_special_tokens

        input_ids = []
        parts = self.tokens_trie.split(text)

        in_vllm_chat_completion_mode = False
        if (
            self._in_vllm
            and len(parts) > 1
            and parts[0] == SpecialTokens.bos.value
            and parts[1] == SpecialTokens.begin_inst.value
        ):
            # This is a dangerous hack to make the tokenizer work with vLLM.
            # It means we are in chat completion mode.
            bos = False
            eos = False
            in_vllm_chat_completion_mode = True

        if os.environ.get("HF_TOKENIZE_FORCE_NO_SPECIAL_TOKENS", "0") == "1":
            bos = False
            eos = False

        if not self._in_vllm or in_vllm_chat_completion_mode:
            for part in parts:
                if part in self.additional_special_tokens and part in self._vocab_dict:
                    input_ids.append(self._convert_token_to_id(part))
                else:
                    input_ids.extend(self._tokenizer.encode(part, bos=bos, eos=eos))
        else:
            # Doesn't tokenize special tokens properly, but this is the behavior of vLLM when we are in completion mode.
            input_ids = self._tokenizer.encode(text, bos=bos, eos=eos)

        if os.environ.get("HF_TOKENIZE_ABUSE", "1") == "1":
            # A lot faster than the other option
            return input_ids
        else:
            return [self._convert_id_to_token(token_id) for token_id in input_ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if len(tokens) > 0 and isinstance(tokens[0], int):
            return tokens
        return super().convert_tokens_to_ids(tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._vocab_dict[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        piece = self._tokenizer.id_to_piece(index)
        return piece if isinstance(piece, str) else piece.value

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        from mistral_common.tokens.tokenizers.base import SpecialTokens

        if self.is_tekken:
            tokens = [
                t
                for t in tokens
                if (t is SpecialTokens.tool_calls or t not in self._tokenizer._all_special_tokens)
            ]

            if any(isinstance(t, bytes) for t in tokens):
                # we need to encode and decode all tokens again
                shift = self._tokenizer.num_special_tokens

                def _token_to_id(t: str):
                    t_bytes = t.encode("utf-8") if not isinstance(t, bytes) else t
                    try:
                        return shift + self._tokenizer._tekken_token2id_nospecial[t_bytes]
                    except KeyError:
                        logger.warning(
                            "Failed to convert token %s to id, replacing with <unk>",
                            t_bytes,
                        )
                        return self._tokenizer.unk_id

                ids = [_token_to_id(t) for t in tokens]
                decoded = self._tokenizer.decode(ids)
            else:
                decoded = "".join(tokens)
        else:
            # make sure certain special tokens like Tool calls are
            # not decoded
            special_tokens = {SpecialTokens.tool_calls}
            regular_tokens: list[str] = []
            decoded_list = []

            for token in tokens:
                if token in special_tokens:
                    if regular_tokens:
                        decoded_list.append(self._tokenizer.decode(regular_tokens))
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                decoded_list.append(self._tokenizer.decode(regular_tokens))  # type: ignore[no-untyped-call]

            decoded = "".join(decoded_list)

        return decoded

    def save_vocabulary(self, save_directory, filename_prefix: str | None = None) -> tuple[str]:
        """
        Use this method to save the full tokenizer file.
        """

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(save_directory, "tekken.json")

        if os.path.abspath(self._mistral_tokenizer_path) != os.path.abspath(out_vocab_file):
            copyfile(self._mistral_tokenizer_path, out_vocab_file)

        return (out_vocab_file,)

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        request = _make_mistral_chat_completion_request(conversation, tools)
        encoded = self._mistral_tokenizer.encode_chat_completion(request)
        if tokenize:
            # encode-decode to get clean prompt
            return encoded.tokens
        else:
            return encoded.text


def _find_tokenizer_file(files: list[str]):
    file_pattern = re.compile(r"^tokenizer\.model\.v.*$|^tekken\.json$|^tokenizer\.mm\.model\.v.*$")

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure only one Mistral "
            f"tokenizer is present in {files}."
        )
    elif len(matched_files) == 0:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure that a Mistral "
            f"tokenizer is present in {files}."
        )

    return matched_files[0]


def _make_mistral_chat_completion_request(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> "ChatCompletionRequest":
    last_message = messages[-1]
    if last_message["role"] == "assistant":
        last_message["prefix"] = True

    # mistral-common requires AssistantMessage content to be string [1].
    #
    # [1]: https://github.com/mistralai/mistral-common/blob/f4a06998b75ed78bbf5aaf569590b772ea26c9f6/src/mistral_common/protocol/instruct/messages.py#L80
    for message in messages:
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(chunk.get("text") for chunk in content)
                message["content"] = content

    # The Mistral client, in comparison to the OpenAI client, requires the
    # "parameters" dict to be present, even if it's empty.
    if tools:
        for function in [tool["function"] for tool in tools if tool["type"] == "function"]:
            if function.get("parameters") is None:
                function["parameters"] = {}

    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    return ChatCompletionRequest(messages=messages, tools=tools)  # type: ignore[type-var]
