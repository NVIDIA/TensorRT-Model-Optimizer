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

import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LEN = 2048

MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt2",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
    "RecurrentGemma": "recurrentgemma",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "phi3small": "phi3small",
    "phi3": "phi3",
    "phi": "phi",
    "TLGv4ForCausalLM": "phi",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
    "StarCoder": "gptnext",
    "Dbrx": "dbrx",
    "T5": "t5",
    "GLM": "glm",
    "InternLM2ForCausalLM": "internlm",
    "ExaoneForCausalLM": "exaone",
}


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_mode_type_from_engine_dir(engine_dir_str):
    # Split the path by '/' and get the last part
    last_part = os.path.basename(engine_dir_str)

    # Split the last part by '_' and get the first segment
    model_type = last_part.split("_")[0]

    return model_type


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, model_type=None):
    print(f"Initializing tokenizer from {ckpt_path}")
    if "vila" in ckpt_path.lower():
        ckpt_path += "/llm"
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    if model_type and model_type == "qwen":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_dtype(dtype):
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    return dtype


def get_model(ckpt_path, device="cuda", gpu_mem_percentage=0.8):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}

    if "vila" in ckpt_path:
        sys.path.append(os.path.join(ckpt_path, "..", "VILA"))
        from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
        from transformers import AutoModel

        hf_vila = AutoModel.from_pretrained(
            ckpt_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = hf_vila.llm
    else:
        hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

        if hf_config.model_type == "llava":
            from transformers import LlavaForConditionalGeneration

            hf_llava = LlavaForConditionalGeneration.from_pretrained(
                ckpt_path, device_map=device_map, **model_kwargs
            )
            model = hf_llava.language_model
        elif hf_config.model_type == "t5":
            from transformers import T5ForConditionalGeneration

            model = T5ForConditionalGeneration.from_pretrained(
                ckpt_path, device_map=device_map, **model_kwargs
            )
        elif hf_config.model_type == "glm":
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_path, device_map="cuda", **model_kwargs, trust_remote_code=True
            )
        else:
            from accelerate import infer_auto_device_map, init_empty_weights
            from accelerate.utils import get_max_memory

            with init_empty_weights():
                # When computing the device_map, assuming half precision by default,
                # unless specified by the hf_config.
                torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
                model = AutoModelForCausalLM.from_config(
                    hf_config, torch_dtype=torch_dtype, trust_remote_code=True
                )

            max_memory = get_max_memory()
            inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

            on_cpu = "cpu" in inferred_device_map.values()

            if on_cpu:
                for device in max_memory.keys():
                    if isinstance(device, int):
                        max_memory[device] *= gpu_mem_percentage

                print(
                    "Model does not fit to the GPU mem. "
                    f"We apply the following memmory limit for calibration: \n{max_memory}\n"
                    "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                    "reduce the calibration `batch_size` manually."
                )
                model_kwargs["max_memory"] = max_memory

            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                **model_kwargs,
                trust_remote_code=True,
            )
    model.eval()
    if device == "cuda":
        if not is_model_on_gpu(model):
            print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())
