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
from typing import Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from utils import get_model_type

import modelopt.torch.opt as mto
import modelopt.torch.sparsity as mts
from modelopt.torch.export import export_tensorrt_llm_checkpoint

DEFAULT_PAD_TOKEN = "[PAD]"


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
    print(f"Initializing model from {ckpt_path} using dtype {dtype}.")
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

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # Export the sparse model to trt-llm checkpoint
    model_type = get_model_type(model)
    if args.modelopt_restore_path:
        print(f"Loading sparsity state from {args.modelopt_restore_path}")
        if not os.path.isfile(args.modelopt_restore_path):
            raise FileNotFoundError(f"Sparsity state file {args.modelopt_restore_path} not found.")

        mto.restore(model, args.modelopt_restore_path)

    print(f"Exporting trt-llm checkpoint to {args.output_dir}")
    with torch.inference_mode():
        model = mts.export(model)
        export_tensorrt_llm_checkpoint(
            model,
            model_type,
            torch.float16,
            export_dir=args.output_dir,
            inference_tensor_parallel=args.inference_tensor_parallel,
            inference_pipeline_parallel=args.inference_pipeline_parallel,
            naive_fp8_quantization=args.naive_quantization,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path", help="Specify where the PyTorch checkpoint path is", required=True
    )
    parser.add_argument(
        "--modelopt_restore_path",
        help="Path to the pruned modelopt checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="fp16")
    parser.add_argument(
        "--model_max_length",
        default=2048,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
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
