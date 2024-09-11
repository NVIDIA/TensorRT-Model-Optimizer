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
from config import (
    FP8_FP16_DEFAULT_CONFIG,
    FP8_FP32_DEFAULT_CONFIG,
    get_int8_config,
    set_stronglytyped_precision,
)
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import check_lora, filter_func, load_calib_prompts, quantize_lvl, set_fmha

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

MODEL_ID = {
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd2.1": "stabilityai/stable-diffusion-2-1",
    "sd2.1-base": "stabilityai/stable-diffusion-2-1-base",
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
}

# You can include the desired arguments for calibration at this point.
ADDTIONAL_ARGS = {
    "flux-dev": {
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
    },
}


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        common_args = {
            "prompt": prompts,
            "num_inference_steps": kwargs["n_steps"],
        }
        other_args = (
            ADDTIONAL_ARGS[kwargs["model_id"]]
            if kwargs["model_id"] in ADDTIONAL_ARGS.keys()
            else {}
            # Also, you can add the negative_prompt when doing the calibration if the model allows
        )
        pipe(**common_args, **other_args).images


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument(
        "--quantized-torch-ckpt-save-path",
        default=None,
        help="The file path for the quantized Torch checkpoint ends with a .pt extension.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl-1.0",
        choices=[
            "sdxl-1.0",
            "sdxl-turbo",
            "sd2.1",
            "sd2.1-base",
            "sd3-medium",
            "flux-dev",
        ],
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n-steps",
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
        help=(
            "Ways to collect the amax of each layers, for example, min-max means min(max(step_0),"
            " max(step_1), ...)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=None, help="Will export the ONNX if not None"
    )

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    if args.model == "sd2.1" or args.model == "sd2.1-base":
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch.float16, safety_checker=None
        )
    elif args.model == "sd3-medium":
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

    backbone = pipe.unet if args.model not in ["sd3-medium", "flux-dev"] else pipe.transformer

    if args.quant_level == 4.0:
        assert args.format != "int8", "We only support fp8 for Level 4 Quantization"
        assert args.model == "sdxl-1.0", "We only support fp8 for SDXL on Level 4"
        set_fmha(backbone)
    if not args.restore_from:
        # This is a list of prompts
        cali_prompts = load_calib_prompts(
            args.batch_size,
            "./calib/calib_prompts.txt",
        )
        extra_step = (
            1 if args.model == "sd2.1" or args.model == "sd2.1-base" else 0
        )  # Depending on the scheduler. some schedulers will do n+1 steps
        if args.format == "int8":
            # Making sure to use global_min in the calibrator for SD 2.1
            assert args.collect_method != "default"
            if args.model == "sd2.1" or args.model == "sd2.1-base":
                args.collect_method = "global_min"
            quant_config = get_int8_config(
                backbone,
                args.quant_level,
                args.alpha,
                args.percentile,
                args.n_steps + extra_step,
                collect_method=args.collect_method,
            )
        elif args.format == "fp8":
            if args.collect_method == "default":
                quant_config = (
                    FP8_FP32_DEFAULT_CONFIG if args.model == "sd2.1" else FP8_FP16_DEFAULT_CONFIG
                )
            else:
                raise NotImplementedError

        def forward_loop(backbone):
            if args.model not in ["sd3-medium", "flux-dev"]:
                pipe.unet = backbone
            else:
                pipe.transformer = backbone
            do_calibrate(
                pipe=pipe,
                calibration_prompts=cali_prompts,
                calib_size=args.calib_size,
                n_steps=args.n_steps,
                model_id=args.model,
            )

        # All the LoRA layers should be fused
        check_lora(backbone)
        if args.model == "flux-dev":
            set_stronglytyped_precision(quant_config, "BFloat16")
        mtq.quantize(backbone, quant_config, forward_loop)
        mto.save(backbone, f"{args.quantized_torch_ckpt_save_path}")
    else:
        mto.restore(backbone, args.restore_from)
    quantize_lvl(backbone, args.quant_level)
    mtq.disable_quantizer(backbone, filter_func)

    # if you want to export the model on CPU, move the dummy input and the model to cpu and float32
    if args.onnx_dir is not None:
        if args.format == "fp8":
            generate_fp8_scales(backbone)
        pipe.to("cpu")
        torch.cuda.empty_cache()
        # to save GPU memory
        backbone.to("cuda")
        modelopt_export_sd(backbone, f"{str(args.onnx_dir)}", args.model, args.format)


if __name__ == "__main__":
    main()
