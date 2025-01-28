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
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import tensorrt_llm
import torch
from packaging.version import parse
from tensorrt_llm.bindings.executor import DecodingConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.llmapi import SamplingParams
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

    def __init__(
        self,
        engine_dir: Union[str, Path],
        tokenizer: Optional[Union[str, Path, TokenizerBase]] = None,
        kv_cache_config: dict[str, Union[int, float]] = {},
        medusa_choices: Any = None,
    ):
        """Initializes the LLM runner class.

        Args:
            engine_dir: the directory path of the TensorRT-LLM engine.
            tokenizer: the tokenizer. For example, a tokenizer from the Huggingface model.
            kv_cache_config: the kv cache config as a dict. Please refer to
                https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md
        """
        assert parse(tensorrt_llm.__version__) >= parse("0.15.0")

        with open(Path(engine_dir) / "config.json", "r") as engine_config_file:
            engine_config = json.load(engine_config_file)
            build_config = engine_config["build_config"]
            world_size = (
                engine_config.get("pretrained_config", {}).get("mapping", {}).get("world_size", 1)
            )
            max_tokens_kv_cache = build_config["max_seq_len"] * build_config["max_batch_size"]
            self.gather_context_logits = build_config.get("gather_context_logits", False)

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

    @property
    def max_input_len(self):
        """Get the max input length from the LLM instance."""
        return self.args.build_config.max_input_len

    @property
    def max_beam_width(self):
        """Get the max beam width from the LLM instance."""
        return self.args.build_config.max_beam_width

    def generate_tokens(
        self,
        prompts: Union[Iterable[str], Iterable[list[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = None,
        keep_input_prompt: bool = True,
        stop_words: list[str] = None,
    ) -> Union[list[list[int]], list[list[list[int]]]]:
        """Generates the tokens based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            keep_input_prompt: Set to include input prommpts in the outputs.
            stop_words: A list of words that the generate stops on.

        Returns:
            a list of output token lists if max_beam_width is 1 or a 3D list with shape [batch, beam, sequence_len].
        """
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        beam_width = self.max_beam_width
        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        sampling_config = SamplingParams(
            max_tokens=max_new_tokens, beam_width=beam_width, stop=stop_words, **kwargs
        )

        prompt_ids = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        outputs = self.generate(prompt_ids, sampling_params=sampling_config, use_tqdm=False)

        def _process_output_token_id(output_token_id, prompt_id, with_input, keep_input_prompt):
            if with_input == keep_input_prompt:
                return output_token_id

            elif with_input:  # and not keep_input_prompt
                return output_token_id[len(prompt_id) :]

            else:  # not with_input and keep_input_prompt:
                return prompt_id + output_token_id

        with_input = False
        output_tokens = []
        for prompt_id, output in zip(prompt_ids, outputs):
            output_token_ids = [out.token_ids for out in output.outputs]

            for output_token_id in output_token_ids:
                output_tokens.append(
                    _process_output_token_id(
                        output_token_id, prompt_id, with_input, keep_input_prompt
                    )
                )

        return (
            output_tokens
            if beam_width == 1
            else [
                output_tokens[i : i + beam_width] for i in range(0, len(output_tokens), beam_width)
            ]
        )

    def generate_text(
        self,
        prompts: Union[Iterable[str], Iterable[list[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = None,
        keep_input_prompt: bool = True,
        stop_words: list[str] = None,
    ) -> Union[list[str], list[list[str]]]:
        """Generates the text based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            keep_input_prompt: Set to include input prommpts in the outputs.
            stop_words: A list of words the generate will stop on.

        Returns:
            a list of output text strings if max_beam_width is 1 or a 2D list with shape [batch, beam].
        """
        beam_width = self.max_beam_width
        output_tokens = self.generate_tokens(
            prompts,
            max_new_tokens,
            temperature,
            keep_input_prompt=keep_input_prompt,
            top_p=top_p,
            stop_words=stop_words,
        )
        if beam_width == 1:
            output_text = [self.tokenizer.decode(batch) for batch in output_tokens]
        else:
            output_text = [
                [self.tokenizer.decode(beam) for beam in batch] for batch in output_tokens
            ]
        return output_text

    def generate_context_logits(
        self,
        prompts: Union[Iterable[str], Iterable[list[int]]],
        temperature: float = 1.0,
        top_p: float = None,
    ) -> list[torch.tensor]:
        """Generates the context logits based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            keep_input_prompt: Set to include input prommpts in the outputs.

        Returns:
            a tensor list of the context_logits.
        """
        assert (
            self.gather_context_logits
        ), "Please enable gather_context_logits flag when building the engine."
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        kwargs["return_context_logits"] = True

        sampling_config = SamplingParams(max_tokens=1, beam_width=1, **kwargs)

        prompt_ids = [
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            for prompt in prompts
        ]
        outputs = self.generate(prompt_ids, sampling_params=sampling_config, use_tqdm=False)

        return [output.context_logits for output in outputs]
