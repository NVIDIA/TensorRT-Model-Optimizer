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

"""The utils to support Nemo models."""

import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Tokenizer, PreTrainedTokenizer, T5Tokenizer

try:
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    sentence_piece_tokenizer_available = True
except Exception:
    sentence_piece_tokenizer_available = False
    SentencePieceTokenizer = PreTrainedTokenizer


class CustomSentencePieceTokenizer(SentencePieceTokenizer):
    """Custom tokenizer based on Nemo SentencePieceTokenizer.

    This extension of SentencePieceTokenizer is to make API consistent with HuggingFace tokenizers
    in order to run evaluation tools in examples/tensorrt_llm/scripts/nemo_example.sh script.
    """

    def __init__(self, *args, **kwargs):
        """Constructor method with extra check for non-legacy SentencePieceTokenizer variant."""
        super().__init__(*args, **kwargs)
        assert not self.legacy, "Only non-legacy tokenizer is supported"

    @property
    def pad_token(self):
        """pad_token."""
        return self.pad_id

    @property
    def eos_token(self):
        """eos_token."""
        return self.eos_id

    @property
    def pad_token_id(self):
        """pad_token_id."""
        return self.pad_id

    @property
    def eos_token_id(self):
        """eos_token_id."""
        return self.eos_id

    def encode(self, text, return_tensors=None, max_length=None, **kwargs):
        """Method introduced for HF tokenizers API consistency for evaluation scripts.

        Note: kwargs other than return_tensors and max_length are ignored.
        """
        output = self.tokenizer.encode_as_ids(text)
        if max_length is not None:
            if isinstance(text, str):
                output = output[:max_length]
            if isinstance(text, list):
                output = [x[:max_length] for x in output]
        if return_tensors == "pt":
            # Only plain text input is supported since for list of strings some padding needs to be introduced
            assert isinstance(text, str), (
                "Returning 'pt' tensors is only supported for simple text input"
            )
            output = torch.LongTensor(output).reshape((1, -1))
        return output

    def batch_encode_plus(self, texts, **kwargs):
        """Method introduced for HF tokenizers API consistency for evaluation scripts.

        Note: kwargs are ignored.
        """
        assert isinstance(texts, list), f"Expected list of texts got {texts}"
        return {"input_ids": self.tokenizer.encode_as_ids(texts)}

    def decode(self, ids, **kwargs):
        """MMethod introduced for HF tokenizers API consistency for evaluation scripts.

        Note: kwargs are ignored.
        """
        if isinstance(ids, np.ndarray) or torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def batch_decode(self, ids, **kwargs):
        """Method introduced for HF tokenizers API consistency for evaluation scripts."""
        return self.decode(ids, **kwargs)


def _build_tokenizer(tokenizer_config: dict):
    if tokenizer_config["library"] == "sentencepiece":
        # Model Optimizer modification.
        # Turn off legacy model by default: See https://github.com/huggingface/transformers/pull/24622
        tokenizer = T5Tokenizer(tokenizer_config["model"], extra_ids=0, legacy=False)
    elif "GPT2" in tokenizer_config["type"]:
        tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
    else:
        raise ValueError(f"Tokenizer type {tokenizer_config['library']} not handled")

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": "<s>"})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    return tokenizer


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    # TODO: Remove this function sometime in the future
    warnings.warn(
        (
            "Function get_tokenzier is deprecated and may be removed soon. "
            "Please consider using get_nemo_tokenizer instead."
        ),
        DeprecationWarning,
    )
    model_path = (
        tokenizer_dir_or_path / "tokenizer.model"
        if tokenizer_dir_or_path.is_dir()
        else tokenizer_dir_or_path
    )
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return _build_tokenizer(tokenizer_config)


def get_nemo_tokenizer(tokenizer_cfg_path: str):
    """Build tokenizer from Nemo tokenizer config.

    Refer to the logic of get_nmt_tokenizer function on how to instantiate tokenizers in Nemo, see
    https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/tokenizer_utils.py.
    """
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from omegaconf import OmegaConf

    print(f"Initializing tokenizer from tokenizer config {tokenizer_cfg_path}")
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)

    library = tokenizer_cfg.library
    legacy = tokenizer_cfg.get("sentencepiece_legacy", library == "sentencepiece")
    special_tokens_dict = tokenizer_cfg.get("special_tokens", {})

    if library == "huggingface":
        print(f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_cfg.type}")
        tokenizer = AutoTokenizer(
            pretrained_model_name=tokenizer_cfg.type,
            vocab_file=tokenizer_cfg.get("vocab_file", None),
            merges_file=tokenizer_cfg.get("merge_file", None),
            use_fast=tokenizer_cfg.get("use_fast", False),
            **special_tokens_dict,
        )
        print("Unwrapping HuggingFace tokenizer from Nemo AutoTokenizer")
        # TODO: Either Nemo tokenizer API alignment is needed https://github.com/NVIDIA/NeMo/pull/8818
        #  or alternatively we can use only HuggingFace tokenizers in future. Current workaround
        #  proposed here is to unwrap the HuggingFace tokenizer from Nemo AutoTokenizer class:
        tokenizer = tokenizer.tokenizer
    elif library == "sentencepiece":
        print(f"Getting SentencePiece with model: {tokenizer_cfg.model}")
        if not sentence_piece_tokenizer_available:
            warnings.warn("Cannot import nemo package, falling back to HF PreTrainedTokenizer!")
        tokenizer = CustomSentencePieceTokenizer(model_path=tokenizer_cfg.model, legacy=legacy)
    else:
        raise NotImplementedError(
            "Currently we only support 'huggingface' and 'sentencepiece' tokenizer libraries."
        )

    return tokenizer
