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

import PIL
import PIL.Image
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
    "Gemma": "gemma",
    "phi": "phi",
    "TLGv4ForCausalLM": "phi",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
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


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    if "vila" in ckpt_path:
        register_vila(ckpt_path)

    model_kwargs = {"torch_dtype": dtype}
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration

        hf_llava = LlavaForConditionalGeneration.from_pretrained(
            ckpt_path, device_map=device_map, **model_kwargs
        )
        model = hf_llava.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, device_map=device_map, **model_kwargs, trust_remote_code=True
        )
    model.eval()
    if device == "cuda":
        if not all("cuda" in str(param.device) for param in model.parameters()):
            raise RuntimeError(
                "Some parameters are not on a GPU. Ensure there is sufficient GPU memory or set the device to 'cpu'."
            )

    return model


def register_vila(ckpt_path: str):
    """
    Imports model class and model config from VILA source code until it is added to HF model zoo

    Args: huggingface model directory
    """
    directory_path = ckpt_path + "/../VILA"
    if os.path.isdir(directory_path):
        sys.path.append(directory_path)
        from llava.model import LlavaConfig, LlavaLlamaForCausalLM
    else:
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")

    AutoConfig.register("llava_llama", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


def image_process_vila(image: PIL.Image.Image, ckpt_path: str):
    """
    Processes input image using VILA image processor

    Args:
        image: the image to be processed using VILA image processor
        ckpt_path: the huggingface model directory

    Returns:
        LlavaLlamaForCausalLM: instance of pretrained model class
        Tensor: processed image using VILA image processor
    """
    register_vila(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    image = image_processor(images=image, return_tensors="pt")["pixel_values"]
    return model, image
