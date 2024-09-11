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
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
)
from e2e_pipeline import deploy
from quantize import MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl-1.0",
        choices=["sdxl-1.0", "sd3-medium", "flux-dev"],
        help="The Model ID that you want to test with",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="./backbone.plan",
        help="The target engine file path. If it doesn't exist, the script will create one for you.",
    )
    parser.add_argument(
        "--inf-img-size",
        type=int,
        default=2,
        help="The img size (batch size) to use during inference (inf-batch-size) should not exceed the batch \
            size specified in maxShapes. For more details, you can refer to the \
            https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/\
            OptimizationProfile.html#ioptimizationprofile Additionally, please \
            note that Flux-dev is a distilled model, meaning that one image \
            corresponds to one batch size.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-iter", type=int, default=8)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            print("Parsed args -- {}: {}".format(key, value))
    return args


def main():
    args = parse_args()

    if args.model == "sd3-medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16
        )
    elif args.model == "flux-dev":
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    deploy.load(
        pipe,
        args.model,
        Path(args.engine_path),
        args.inf_img_size if args.model == "flux-dev" else args.inf_img_size * 2,
    )

    prompt = "An astronaut riding a green horse"
    _num_img = args.inf_img_size // (2 if args.model != "flux-dev" else 1)
    if _num_img > 1:
        prompt = [prompt] * _num_img
    # warmup
    _ = pipe(prompt=prompt, num_inference_steps=args.num_inference_steps).images[0]

    start_time = time.time()
    for i in range(args.num_iter):
        _ = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
    end_time = time.time()

    img_per_sec = _num_img / ((end_time - start_time) / args.num_iter)
    print(f"Img per second: {img_per_sec}")


if __name__ == "__main__":
    main()
