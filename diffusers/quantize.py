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

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from utils import (
    get_fp8_config,
    get_int8_config,
    load_calib_prompts,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
        ).images


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--exp_name", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        choices=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
            "runwayml/stable-diffusion-v1-5",
        ],
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=30,
        help="Number of denoising steps, for SDXL-turbo, use 1-4 steps",
    )

    # Calibration and quantization parameters
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "fp8"])
    parser.add_argument("--percentile", type=float, default=1.0, required=False)
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    if args.model == "runwayml/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model, torch_dtype=torch.float16, safety_checker=None
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    # This is a list of prompts
    cali_prompts = load_calib_prompts(args.batch_size, "./calib/calib_prompts.txt")
    extra_step = (
        1 if args.model == "runwayml/stable-diffusion-v1-5" else 0
    )  # Depending on the scheduler. some schedulers will do n+1 steps
    if args.format == "int8":
        # Making sure to use global_min in the calibrator for SD 1.5
        if args.model == "runwayml/stable-diffusion-v1-5":
            args.collect_method = "global_min"
        quant_config = get_int8_config(
            pipe.unet,
            args.quant_level,
            args.alpha,
            args.percentile,
            args.n_steps + extra_step,
            collect_method=args.collect_method,
        )
    else:
        if args.collect_method == "default":
            quant_config = mtq.FP8_DEFAULT_CFG
        else:
            quant_config = get_fp8_config(
                pipe.unet,
                args.percentile,
                args.n_steps + extra_step,
                collect_method=args.collect_method,
            )

    def forward_loop(unet):
        pipe.unet = unet
        do_calibrate(
            pipe=pipe,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

    mtq.quantize(pipe.unet, quant_config, forward_loop)
    mto.save(pipe.unet, f"./unet.state_dict.{args.exp_name}.pt")


if __name__ == "__main__":
    main()
