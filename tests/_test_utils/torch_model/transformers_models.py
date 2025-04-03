# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

import modelopt.torch.opt as mto


def get_tiny_llama(**config_kwargs) -> LlamaForCausalLM:
    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(**config_kwargs)
    tiny_llama = LlamaForCausalLM(LlamaConfig(**kwargs))

    return tiny_llama


def create_tiny_llama_dir(tmp_path: Path, with_tokenizer: bool = False, **config_kwargs) -> Path:
    if with_tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.save_pretrained(tmp_path / "tiny_llama")
        config_kwargs["vocab_size"] = tokenizer.vocab_size

    tiny_llama = get_tiny_llama(**config_kwargs)
    tiny_llama = tiny_llama.to(torch.bfloat16)  # Use same dtype as TinyLlama-1.1B-Chat-v1.0
    tiny_llama.save_pretrained(tmp_path / "tiny_llama")
    return tmp_path / "tiny_llama"


def create_tiny_bert_dir(tmp_path: Path) -> Path:
    model = BertForQuestionAnswering(
        BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(tmp_path / "tiny_bert")
    return tmp_path / "tiny_bert"


def tf_output_tester(model_ref, model_test):
    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs)
    output_test = model_test(**inputs)
    if hasattr(output_ref, "logits"):
        assert torch.allclose(output_ref.logits, output_test.logits)
    else:
        assert torch.allclose(output_ref.start_logits, output_test.start_logits)
        assert torch.allclose(output_ref.end_logits, output_test.end_logits)


def tf_modelopt_state_and_output_tester(model_ref, model_test):
    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for module in model_test.modules():
        if hasattr(module, "_is_hf_initialized"):
            delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    tf_output_tester(model_ref, model_test)
