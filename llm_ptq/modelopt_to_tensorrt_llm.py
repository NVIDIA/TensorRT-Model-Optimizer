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

"""An example to convert an Model Optimizer exported model to tensorrt_llm."""

import argparse

from run_tensorrt_llm import run

from modelopt.deploy.llm import build_tensorrt_llm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=512)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_num_beams", type=int, default=1)
    parser.add_argument("--engine_dir", type=str, default="/tmp/modelopt")
    parser.add_argument("--refit_engine_dir", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument(
        "--input_texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument("--num_build_workers", type=int, default="1")
    parser.add_argument("--enable_sparsity", type=str2bool, default=False)
    parser.add_argument(
        "--max_prompt_embedding_table_size",
        "--max_multimodal_len",
        type=int,
        default=0,
        help="Setting to a value > 0 enables support for prompt tuning or multimodal input.",
    )

    return parser.parse_args()


def main(args):
    build_tensorrt_llm(
        pretrained_config=args.model_config,
        engine_dir=args.engine_dir,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_num_beams,
        num_build_workers=args.num_build_workers,
        enable_sparsity=args.enable_sparsity,
        max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
    )

    if args.model_config is not None and all(
        model_name not in args.model_config for model_name in ("vila", "llava")
    ):
        # Reduce output_len for the inference run example.
        args.max_output_len = 100
        run(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
