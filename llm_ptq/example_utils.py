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
import os
import sys
from os.path import isfile, join

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


def get_model(ckpt_path, device="cuda"):
    print(f"Initializing model from {ckpt_path}")

    if "vila" in ckpt_path:
        register_vila(ckpt_path)

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}
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
    elif hf_config.model_type == "t5":
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(
            ckpt_path, device_map=device_map, **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, device_map=device_map, **model_kwargs, trust_remote_code=True
        )
    model.eval()
    if device == "cuda":
        if not is_model_on_gpu(model):
            print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


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


def combine_medusa_head(base_model: str, medusa_model: str):
    """
    Combines medusa heads with base model into one model state dict

    Args:
        base_model: path to the base model dir
        medusa_model: path to the medusa head dir
    """
    medusa_weight = join(medusa_model, "medusa_lm_head.pt")
    assert isfile(medusa_weight), "Medusa head not found at " + medusa_weight

    index_file_list = [
        join(base_model, f)
        for f in os.listdir(base_model)
        if isfile(join(base_model, f)) and f.endswith(".bin.index.json")
    ]
    if len(index_file_list) == 1:
        base_model_bin_index = index_file_list[0]
    else:
        base_model_bin_index = None

    bin_file_list = [
        join(base_model, f)
        for f in os.listdir(base_model)
        if isfile(join(base_model, f)) and f.endswith(".bin")
    ]
    if len(bin_file_list) == 1:
        base_model_weight = bin_file_list[0]
    else:
        base_model_weight = None

    assert not (
        base_model_weight is None and base_model_bin_index is None
    ), "Base model state dict not detected."

    with open(join(medusa_model, "config.json")) as f:
        medusa_config = json.load(f)
    with open(join(base_model, "config.json")) as f:
        base_config = json.load(f)

    # Copy the medusa config to base model config
    for key in medusa_config.keys():
        base_config[key] = medusa_config[key]
    with open(join(base_model, "config.json"), "w") as f:
        json.dump(base_config, f, indent=4)

    medusa_modules = torch.load(medusa_weight)
    # The bin.index file has higher priority. If it presents, we will update base_model_weight
    if base_model_bin_index is not None:
        base_model_weight = bin_file_list[0]

        medusa_size = os.path.getsize(medusa_weight)
        with open(base_model_bin_index) as f:
            bin_index = json.load(f)
        bin_index["metadata"]["total_size"] += medusa_size
        for module in medusa_modules.keys():
            bin_index["weight_map"]["medusa_head." + module] = base_model_weight.split("/")[-1]

        with open(base_model_bin_index, "w") as f:
            json.dump(bin_index, f, indent=4)

    # Store the medusa head state dict to the base model state dict
    base_model_modules = torch.load(base_model_weight)
    for module in medusa_modules.keys():
        base_model_modules["medusa_head." + module] = medusa_modules[module]

    torch.save(base_model_modules, base_model_weight)
