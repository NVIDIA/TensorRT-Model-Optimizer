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

import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, T5Tokenizer

DEFAULT_HF_MODEL_DIRS = {
    "BaichuanForCausalLM": "baichuan-inc/Baichuan-13B-Chat",
    "BloomForCausalLM": "bigscience/bloom-560m",
    "ChatGLMForCausalLM": "THUDM/chatglm3-6b",
    "FalconForCausalLM": "tiiuae/falcon-rw-1b",
    "GPTForCausalLM": "gpt2-medium",
    "GPTJForCausalLM": "EleutherAI/gpt-j-6b",
    "GPTNeoXForCausalLM": "EleutherAI/gpt-neox-20b",
    "InternLMForCausalLM": "internlm/internlm-chat-7b",
    "LlamaForCausalLM": "meta-llama/Llama-2-7b-hf",
    "MPTForCausalLM": "mosaicml/mpt-7b",
    "PhiForCausalLM": "microsoft/phi-2",
    "OPTForCausalLM": "facebook/opt-350m",
    "QWenForCausalLM": "Qwen/Qwen-7B",
}

DEFAULT_PROMPT_TEMPLATES = {
    "InternLMForCausalLM": "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    "QWenForCausalLM": (
        "<|im_start|>system\nYou are a helpful"
        " assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    ),
}


def read_model_name(engine_dir: str):
    with open(Path(engine_dir) / "config.json", "r") as f:
        config = json.load(f)

    # MODELOPT modification.
    if "builder_config" in config:
        return config["builder_config"]["name"], None

    model_arch = config["pretrained_config"]["architecture"]
    model_version = None
    if model_arch == "ChatGLMForCausalLM":
        model_version = config["pretrained_config"]["chatglm_version"]
    if model_arch == "QWenForCausalLM":
        model_version = config["pretrained_config"]["qwen_type"]
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(
    tokenizer_dir: Optional[str] = None,
    vocab_file: Optional[str] = None,
    model_name: str = "GPTForCausalLM",
    model_version: Optional[str] = None,
    tokenizer_type: Optional[str] = None,
):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            tokenizer_type=tokenizer_type,
            use_fast=use_fast,
        )
    elif vocab_file.endswith(".yaml"):
        # Model Optimizer modification
        # For Nemo models, tokenizer is instantiated based on its config
        from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

        tokenizer = get_nemo_tokenizer(vocab_file)
    elif model_name == "GemmaForCausalLM" or model_name == "RecurrentGemmaForCausalLM":
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(
            vocab_file=vocab_file, padding_side="left", truncation_side="left", legacy=False
        )
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(
            vocab_file=vocab_file, padding_side="left", truncation_side="left", legacy=False
        )

    # Model Optimizer modification
    if (
        model_name == "QWenForCausalLM"
        and model_version == "qwen"
        and tokenizer_dir
        and (Path(tokenizer_dir) / "generation_config.json").exists()
    ):
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config["pad_token_id"]
        end_id = gen_config["eos_token_id"]
    elif model_name == "ChatGLMForCausalLM" and model_version == "glm":
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id
