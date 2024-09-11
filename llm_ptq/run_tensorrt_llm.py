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

"""An example script to run the tensorrt_llm engine."""

import argparse
from pathlib import Path

import torch
from summarize.utils import load_tokenizer, read_model_name
from transformers import (
    PreTrainedTokenizerBase,
)

from modelopt.deploy.llm import LLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=100)
    parser.add_argument("--engine_dir", type=str, default="/tmp/modelopt")
    parser.add_argument(
        "--input_texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )

    return parser.parse_args()


def run(args):
    if not args.tokenizer:
        # Assume the tokenizer files are saved in the engine_dr.
        args.tokenizer = args.engine_dir

    if isinstance(args.tokenizer, PreTrainedTokenizerBase):
        tokenizer = args.tokenizer
    else:
        pad_id, end_id = None, None
        model_name, _ = read_model_name(args.engine_dir)
        tokenizer_path = Path(args.tokenizer)

        if tokenizer_path.exists() and tokenizer_path.is_file():
            tokenizer, pad_id, end_id = load_tokenizer(
                vocab_file=args.tokenizer, model_name=model_name
            )
        else:
            tokenizer, pad_id, end_id = load_tokenizer(
                tokenizer_dir=args.tokenizer, model_name=model_name
            )

        if pad_id and end_id:
            try:
                if hasattr(tokenizer, "pad_token"):
                    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)
                if hasattr(tokenizer, "eos_token"):
                    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(end_id)
            except AttributeError:
                print("pad_token/eos_token is read-only and cannot be set.")

    input_texts = args.input_texts.split("|")
    assert input_texts, "input_text not specified"

    free_memory_before = torch.cuda.mem_get_info()

    print("TensorRT-LLM example outputs:")

    llm = LLM(args.engine_dir, tokenizer=tokenizer)
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate_text(input_texts, args.max_output_len)
    torch.cuda.cudart().cudaProfilerStop()

    free_memory_after = torch.cuda.mem_get_info()
    print(
        f"Use GPU memory: {(free_memory_before[0] - free_memory_after[0]) / 1024 / 1024 / 1024} GB"
    )

    print(f"Generated outputs: {outputs}")

    if llm.gather_context_logits:
        logits = llm.generate_context_logits(input_texts)
        print(f"Generated logits: {logits}")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
