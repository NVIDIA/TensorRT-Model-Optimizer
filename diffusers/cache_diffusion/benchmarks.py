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
import time
from pathlib import Path

import torch
from cache_diffusion import cachify
from cache_diffusion.utils import SDXL_DEFAULT_CONFIG
from diffusers import DiffusionPipeline
from pipeline.deploy import compile, teardown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-iter", type=int, default=8)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            print("Parsed args -- {}: {}".format(key, value))
    return args


def main(args):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    prompt = "A random person with a head that is made of flowers, photo by James C. Leyendecker, \
        Afrofuturism, studio portrait, dynamic pose, national geographic photo, retrofuturism, \
            biomorphicy"

    compile(
        pipe.unet,
        onnx_path=Path("./onnx"),
        engine_path=Path("./engine"),
        batch_size=args.batch_size,
    )

    cachify.prepare(pipe, args.num_inference_steps, SDXL_DEFAULT_CONFIG)

    generator = torch.Generator(device="cuda").manual_seed(2946901)
    total_time = 0
    cachify.disable(pipe)
    for _ in range(args.num_iter):
        start_time = time.time()
        _ = pipe(
            prompt=[prompt] * args.batch_size,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        end_time = time.time()
        total_time += end_time - start_time
    total_time = total_time / args.num_iter
    latency = total_time / args.batch_size
    print(f"TRT Disabled Cache: {latency}")

    generator = torch.Generator(device="cuda").manual_seed(2946901)
    total_time = 0
    cachify.enable(pipe)
    for _ in range(args.num_iter):
        start_time = time.time()
        _ = pipe(
            prompt=[prompt] * args.batch_size,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        end_time = time.time()
        total_time += end_time - start_time
    total_time = total_time / args.num_iter
    latency = total_time / args.batch_size
    print(f"TRT Enabled Cache: {latency}")
    teardown(pipe.unet)


if __name__ == "__main__":
    args = parse_args()
    main(args)
