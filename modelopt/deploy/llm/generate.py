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

"""A wrapper over the TensorRT-LLM high level API runner."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tensorrt_llm
import torch
from packaging.version import Version
from tensorrt_llm import SamplingParams
from tensorrt_llm.bindings.executor import DecodingConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.llmapi.llm import LLM as TRT_LLM
from tensorrt_llm.llmapi.tokenizer import TokenizerBase, TransformersTokenizer


def _sanitize_temperature_and_top_p(temperature, top_p):
    assert temperature >= 0.0, "Temperature must be greater than 0.0."

    # TRT LLM acccepts temperature values only greater than 0.0
    temperature = max(temperature, 0.001)

    kwargs = {"temperature": temperature}
    if top_p is not None:
        # cpp executor only supports topP.value() > 0.f
        top_p = 1e-4 if top_p == 0 else top_p
        kwargs["top_p"] = top_p

    return kwargs


class LLM(TRT_LLM):
    """A wrapper over the ``tensorrt_llm.llmapi.llm.LLM`` for LLM profiling and validation."""

    def _build_trt_llm_from_config(
        self, config, engine_dir, tokenizer, kv_cache_config, medusa_choices
    ):
        build_config = config["build_config"]
        world_size = config.get("pretrained_config", {}).get("mapping", {}).get("world_size", 1)
        max_tokens_kv_cache = build_config["max_seq_len"] * build_config["max_batch_size"]

        trt_kv_cache_config = TRT_KvCacheConfig(enable_block_reuse=False)

        # If not specified, free_gpu_memory_fraction is set to the default TRT LLM value 0.9
        trt_kv_cache_config.free_gpu_memory_fraction = kv_cache_config.get(
            "free_gpu_memory_fraction", 0.9
        )

        # If not specified, max_tokens is set to the max value calculated above.
        trt_kv_cache_config.max_tokens = kv_cache_config.get("max_tokens", max_tokens_kv_cache)

        kwargs = {}
        if medusa_choices is not None:
            decoding_config = DecodingConfig()
            decoding_config.medusa_choices = medusa_choices
            kwargs["decoding_config"] = decoding_config
            assert world_size == 1, "decoding_config does not support multi TP in HLAPI."

        if tokenizer is None:
            # Assume the tokenizer is stored in the engine_dir if not specified.
            tokenizer = engine_dir

        # CustomSentencePieceTokenizer will not be recognized by llmapi, wrapping it around TransformersTokenizer
        if type(tokenizer).__name__ in ["CustomSentencePieceTokenizer"]:
            tokenizer = TransformersTokenizer(tokenizer)

        super().__init__(
            model=engine_dir,
            tokenizer=tokenizer,
            kv_cache_config=trt_kv_cache_config,
            **kwargs,
        )

    def _build_torch_llm_from_config(self, checkpoint_dir, tokenizer, tp, trust_remote_code):
        kwargs = {}
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer

        if tp < 1:
            tp = torch.cuda.device_count()

        # Sometimes 90% of the GPU memory is not enough for the TRT LLM torch engine.
        trt_kv_cache_config = TRT_KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=0.85
        )

        super().__init__(
            backend="pytorch",
            model=checkpoint_dir,
            tensor_parallel_size=tp,
            trust_remote_code=trust_remote_code,
            enable_chunked_prefill=True,
            kv_cache_config=trt_kv_cache_config,
            # pytorch backend configs
            use_cuda_graph=True,
            cuda_graph_padding_enabled=True,
            **kwargs,
        )

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tokenizer: "str | Path | TokenizerBase | None" = None,
        kv_cache_config: dict[str, int | float] = {},
        medusa_choices: Any = None,
        tp: int = 0,
        trust_remote_code: bool = False,
    ):
        """Initializes the LLM runner class.

        Args:
            engine_dir: the directory path of the TensorRT-LLM engine.
            tokenizer: the tokenizer. For example, a tokenizer from the Huggingface model.
            kv_cache_config: the kv cache config as a dict. Please refer to
                https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
            medusa_choices: The medusa choices for the decoding config.
            tp: the tensor parallel size (for the torch backend). If 0, it will be set to the number of GPUs.
            trust_remote_code: whether to trust the remote code (for the torch backend).
        """
        assert Version(tensorrt_llm.__version__) >= Version("0.17.0")

        with open(Path(checkpoint_dir) / "config.json") as config_file:
            config = json.load(config_file)

            if "build_config" in config:
                self._build_trt_llm_from_config(
                    config, checkpoint_dir, tokenizer, kv_cache_config, medusa_choices
                )

                self._is_torch = False
                self._max_seq_len = self.args.build_config.max_seq_len
                self._max_beam_width = self.args.build_config.max_beam_width
                self._gather_context_logits = self.args.build_config.gather_context_logits
            else:
                assert medusa_choices is None, (
                    "medusa_choices is not supported with the torch llmapi"
                )

                self._build_torch_llm_from_config(checkpoint_dir, tokenizer, tp, trust_remote_code)
                self._is_torch = True
                self._max_seq_len = config["max_position_embeddings"]
                self._max_beam_width = 1
                self._gather_context_logits = False

    @property
    def max_seq_len(self):
        """Get the max sequence length from the LLM instance."""
        return self._max_seq_len

    @property
    def max_beam_width(self):
        """Get the max beam width from the LLM instance."""
        return self._max_beam_width

    @property
    def gather_context_logits(self):
        """Returns whether the context_logits can be returned from the LLM instance."""
        return self._gather_context_logits

    def _generate(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ):
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        # TODO: Remove this once torch backend supports stop words
        if self._is_torch:
            stop_words = None

        beam_width = self.max_beam_width
        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        sampling_config = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=True,
            best_of=beam_width,
            stop=stop_words,
            **kwargs,
        )

        return self.generate(prompts, sampling_params=sampling_config, use_tqdm=False)

    def generate_tokens(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ) -> list[list[int]] | list[list[list[int]]]:
        """Generates the tokens based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            stop_words: A list of words that the generate stops on.

        Returns:
            a list of output token lists if max_beam_width is 1 or a 3D list with shape [batch, beam, sequence_len].
        """
        outputs = self._generate(prompts, max_new_tokens, temperature, top_p, stop_words)
        output_tokens = []

        beam_width = self.max_beam_width
        for request_output in outputs:
            token_ids = [
                completion_output.token_ids for completion_output in request_output.outputs
            ]
            if beam_width == 1:
                # each output_text is a single list of tokens.
                output_tokens += token_ids
            else:
                # each output_text is a list of beam searched token lists.
                output_tokens.append(token_ids)

        return output_tokens

    def generate_text(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ) -> list[str] | list[list[str]]:
        """Generates the text based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            stop_words: A list of words the generate will stop on.

        Returns:
            a list of output text strings if max_beam_width is 1 or a 2D list with shape [batch, beam].
        """
        outputs = self._generate(prompts, max_new_tokens, temperature, top_p, stop_words)
        output_texts = []

        beam_width = self.max_beam_width
        for request_output in outputs:
            texts = [completion_output.text for completion_output in request_output.outputs]
            if beam_width == 1:
                # each output_text is a single text string.
                output_texts += texts
            else:
                # each output_text is a list of beam searched texts
                output_texts.append(texts)

        return output_texts

    def generate_context_logits(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> list[torch.tensor]:
        """Generates the context logits based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.

        Returns:
            a tensor list of the context_logits.
        """
        assert self.gather_context_logits, (
            "Please enable gather_context_logits flag when building the engine."
        )
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        kwargs["return_context_logits"] = True

        sampling_config = SamplingParams(max_tokens=1, use_beam_search=True, best_of=1, **kwargs)

        outputs = self.generate(prompts, sampling_params=sampling_config, use_tqdm=False)

        return [output.context_logits for output in outputs]
