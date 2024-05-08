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

import argparse
import os
import random
import time
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts

DEFAULT_PAD_TOKEN = "[PAD]"


def get_calib_dataloader(
    data="cnn_dailymail", tokenizer=None, batch_size=1, calib_size=512, block_size=512, device=None
):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train"
        )
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(
        dataset, return_tensors="pt", padding=True, truncation=True, max_length=block_size
    )
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(ckpt_path: str, model_max_length: int):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )

    return tokenizer


def get_model(ckpt_path, dtype="fp16"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")
    model_kwargs = {"torch_dtype": dtype}

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, device_map="auto", **model_kwargs, trust_remote_code=True
    )
    model.eval()

    return model


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(1234)
    np.random.seed(1234)

    model = get_model(args.model_name_or_path, args.dtype)
    tokenizer = get_tokenizer(args.model_name_or_path, args.model_max_length)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

    calib_size = args.calib_size

    # Get calibration dataloader
    calib_dataloader = get_calib_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        calib_size=calib_size,
        device=args.device,
    )

    # Sparsify the model
    print("Starting sparsifying...")
    total_time = -time.time()
    model = mts.sparsify(
        model,
        args.sparsity_fmt,
        config={"data_loader": calib_dataloader, "collect_func": lambda x: x},
    )
    total_time += time.time()
    print(f"Sparsification done. Total time used: {total_time:.2f}s")

    # Save the sparsity modelopt state
    os.makedirs(args.output_dir, exist_ok=True)
    modelopt_state_path = os.path.join(args.output_dir, "pts_modelopt_state.pth")
    print(f"Saving sparsity modelopt state to {modelopt_state_path}")
    mto.save(model, modelopt_state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path", help="Specify where the PyTorch checkpoint path is", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="fp16")
    parser.add_argument(
        "--model_max_length",
        default=2048,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
    )
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument(
        "--sparsity_fmt", type=str, default="sparsegpt", choices=["sparsegpt", "sparse_magnitude"]
    )
    parser.add_argument(
        "--calib_size", help="Number of samples for calibration.", type=int, default=512
    )
    parser.add_argument("--output_dir", default="output_dir")
    parser.add_argument(
        "--inference_tensor_parallel",
        help="Number of tensor parallel groups for inference.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--inference_pipeline_parallel",
        help="Number of pipeline parallel groups for inference.",
        type=int,
        default=1,
    )
    parser.add_argument("--naive_quantization", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
