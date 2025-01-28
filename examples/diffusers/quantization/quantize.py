# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from config import FP8_DEFAULT_CONFIG, NVFP4_FP8_MHA_CONFIG, get_int8_config, set_quant_config_attr
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import check_lora, filter_func, fp8_mha_disable, load_calib_prompts, quantize_lvl

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
    parser.add_argument("--model-dtype", type=str, default="Half", choices=["Half", "BFloat16"])
    parser.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="Half",
        choices=["Half", "BFloat16", "Float"],
    )

    # Calibration and quantization parameters
    parser.add_argument(
        "--format", type=str, default="int8", choices=["fp16", "int8", "fp8", "fp4"]
    )
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
        default=4.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )
    parser.add_argument(
        "--onnx-dir", type=str, default=None, help="Will export the ONNX if not None"
    )

    args = parser.parse_args()
    if args.model not in ["flux-dev"]:
        assert args.model_dtype != "BFloat16", "Only Flux-dev can be BF16 precision"
    args.calib_size = args.calib_size // args.batch_size
    torch_dtype = torch.float16 if args.model_dtype != "BFloat16" else torch.bfloat16

    if args.model == "sd2.1" or args.model == "sd2.1-base":
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch_dtype, safety_checker=None
        )
    elif args.model == "sd3-medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID[args.model], torch_dtype=torch_dtype
        )
    elif args.model in ["flux-dev"]:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch_dtype,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID[args.model],
            torch_dtype=torch.float16,  # Hardcoded into FP16
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    backbone = pipe.unet if args.model not in ["sd3-medium", "flux-dev"] else pipe.transformer

    if args.quant_level == 4.0:
        assert args.format != "int8", "We only support fp8 for Level 4 Quantization"
    if not args.restore_from and args.format != "fp16":
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
                quant_config = FP8_DEFAULT_CONFIG
            else:
                raise NotImplementedError
        elif args.format == "fp4":
            assert args.model == "flux-dev", "In this example, only FP4 is supported for Flux."
            quant_config = mtq.NVFP4_DEFAULT_CFG  # type: ignore[attr-defined]
            if args.quant_level == 4:
                quant_config = NVFP4_FP8_MHA_CONFIG
        else:
            raise NotImplementedError
        set_quant_config_attr(quant_config, args.trt_high_precision_dtype)

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
        mtq.quantize(backbone, quant_config, forward_loop)
        mto.save(backbone, f"{args.quantized_torch_ckpt_save_path}")
    elif args.restore_from and args.format != "fp16":
        mto.restore(backbone, args.restore_from)

    quantize_lvl(args.model, backbone, args.quant_level)
    mtq.disable_quantizer(backbone, filter_func)

    # if you want to export the model on CPU, move the dummy input and the model to cpu and float32
    if args.onnx_dir is not None:
        if args.format == "fp8" and args.model != "flux-dev":
            generate_fp8_scales(backbone)
        pipe.to("cpu")
        torch.cuda.empty_cache()
        # to save GPU memory
        backbone.to("cuda")
        if args.quant_level == 4.0:
            fp8_mha_disable(backbone, quantized_mha_output=False)
        backbone.eval()
        with torch.no_grad():
            modelopt_export_sd(
                backbone, f"{str(args.onnx_dir)}", args.model, args.format, args.model_dtype
            )


if __name__ == "__main__":
    main()
