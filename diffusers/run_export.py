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
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import filter_func, quantize_lvl

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        choices=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sdxl-turbo",
        ],
    )
    parser.add_argument(
        "--quantized-ckpt",
        type=str,
        default="./base.unet.state_dict.fp8.0.25.384.percentile.all.pt",
    )
    parser.add_argument("--format", default="int8", choices=["int8", "fp8"])
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )
    args = parser.parse_args()

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

    # Lets restore the quantized model
    mto.restore(pipe.unet, args.quantized_ckpt)

    quantize_lvl(pipe.unet, args.quant_level)
    mtq.disable_quantizer(pipe.unet, filter_func)

    # QDQ needs to be in FP32
    pipe.unet.to(torch.float32).to("cpu")
    if args.format == "fp8":
        generate_fp8_scales(pipe.unet)
    modelopt_export_sd(pipe, f"{str(args.onnx_dir)}", args.model)


if __name__ == "__main__":
    main()
