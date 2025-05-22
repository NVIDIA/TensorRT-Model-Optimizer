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
from typing import Any, Optional

import torch
from config import (
    FP8_DEFAULT_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    NVFP4_FP8_MHA_FLUX_CONFIG,
    get_int8_config,
    set_quant_config_attr,
)
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import check_lora, filter_func, fp8_mha_disable, load_calib_prompts, quantize_lvl

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

MODEL_ID: dict[str, str] = {
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}

# Additional model-specific arguments for calibration
ADDITIONAL_ARGS: dict[str, dict[str, Any]] = {
    "flux-dev": {
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
    },
    "flux-schnell": {
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
    },
}


def create_pipeline(
    model_name: str,
    model_dtype: str,
    override_model_path: Optional[str] = None,
) -> DiffusionPipeline:
    """Create and return an appropriate pipeline based on the model_name provided."""
    # Convert string to torch.dtype
    if model_dtype == "Half":
        torch_dtype = torch.float16
    elif model_dtype == "BFloat16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "Float":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown model dtype {model_dtype}.")

    if override_model_path:
        model_path = override_model_path
    else:
        model_path = MODEL_ID[model_name]

    if model_name == "sd3-medium":
        return StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    elif model_name in ["flux-dev", "flux-schnell"]:
        return FluxPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        # Example for stable-diffusion-based models
        return DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )


def do_calibrate(
    pipe: DiffusionPipeline,
    calibration_prompts: list[str],
    model_id: str,
    calib_size: int,
    n_steps: int,
) -> None:
    """
    Run calibration steps on the pipeline using the given prompts.
    """
    for i, prompts in enumerate(calibration_prompts):
        if i >= calib_size:
            break
        common_args = {
            "prompt": prompts,
            "num_inference_steps": n_steps,
        }
        extra_args = ADDITIONAL_ARGS.get(model_id, {})
        # If needed, add negative_prompt or other custom logic here
        pipe(**common_args, **extra_args).images


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Quantization and Calibration Script")

    # Model hyperparameters
    parser.add_argument(
        "--quantized-torch-ckpt-save-path",
        default=None,
        help="File path for the quantized Torch checkpoint (should end with .pt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=["sdxl-1.0", "sdxl-turbo", "sd3-medium", "flux-dev", "flux-schnell"],
        help="Which model to load and quantize.",
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="CPU offloading calibration during inference for GPUs with limited VRAM.",
    )
    parser.add_argument(
        "--override-model-path",
        type=str,
        default=None,
        help="Path to the model if not using default paths in MODEL_ID mapping.",
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to a previously quantized checkpoint to restore from.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="Half",
        choices=["Half", "BFloat16", "Float"],
        help="Precision used to load the model.",
    )
    parser.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="Half",
        choices=["Half", "BFloat16", "Float"],
        help="Precision used in TensorRT high-precision layers.",
    )
    parser.add_argument(
        "--quant-algo",
        type=str,
        default="max",
        choices=["max", "svdquant", "smoothquant"],
        help="Quantization algo",
    )

    # Calibration and quantization parameters
    parser.add_argument(
        "--format",
        type=str,
        default="int8",
        choices=["int8", "fp8", "fp4"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Percentile used for calibration (relevant in some calibrators).",
    )
    parser.add_argument(
        "--collect-method",
        type=str,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
        help="How to collect amax values during calibration.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for calibration.")
    parser.add_argument("--calib-size", type=int, default=128, help="Number of calibration steps.")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="SmoothQuant alpha hyperparameter."
    )
    parser.add_argument("--lowrank", type=int, default=32, help="SVDQuant lowrank hyperparameter.")
    parser.add_argument(
        "--quant-level",
        type=float,
        default=3.0,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Directory to export ONNX models. If None, no export is done.",
    )

    return parser.parse_args()

def quantize_diffusion_model(pipe: DiffusionPipeline,
                             backbone: torch.nn.Module,
                             model_name: str = "flux-dev",
                             precision: str = "fp8",
                             quant_level: float = 3.0,
                             calib_size: int = 128, 
                             batch_size: int = 2, 
                             calibration_source: str = "./calib/calib_prompts.txt",
                             collect_method : str = "default",
                             quant_algo : str = "smoothquant", 
                             percentile: float = 1.0, 
                             num_denoising_steps: int = 30, 
                             trt_high_precision_dtype: str = "Half",
                             alpha: float = 1.0,
                             lowrank: int = 32) -> torch.nn.Module:
    """
    Quantize a diffusion model using the given quant_config.
    """
    if quant_level == 4.0 and format == "int8":
        raise ValueError("Level 4 quantization is only supported for fp8, not int8.")
    
    # Validate model_name exists in one of the supported models
    if model_name not in ["sd3-medium", "flux-dev", "flux-schnell"]:
        raise ValueError(f"model_name must be one of ["sd3-medium", "flux-dev", "flux-schnell"], got {model_name}")

    # Adjust calibration steps to be number of batches
    calib_size = calib_size // batch_size

    calibration_prompts = load_calib_prompts(
            batch_size,
            calibration_source,
        )

    # Build quant_config based on format
    if precision == "int8":
        if collect_method == "default":
                raise ValueError(
                    "You must specify an explicit --collect-method (e.g., 'global_min') for int8."
                )
        if quant_algo != "smoothquant":
            raise ValueError(
                "INT8 quantization only works well when combined with SmoothQuant;"
                "otherwise, it will produce very poor quality results."
            )
        quant_config = get_int8_config(
            backbone,
            quant_level,
            percentile,
            num_denoising_steps,
            collect_method=collect_method,
        )
    elif precision == "fp8":
        if collect_method != "default":
            raise NotImplementedError("Only 'default' collect method is implemented for fp8.")
        quant_config = FP8_DEFAULT_CONFIG
    elif precision == "fp4":
        if model_name.startswith("flux"):
            quant_config = NVFP4_FP8_MHA_FLUX_CONFIG
        else:
            quant_config = NVFP4_DEFAULT_CONFIG
    else:
        raise NotImplementedError(f"Unknown precision specified for quantization: {precision}.")
    
    # Set the quant config
    set_quant_config_attr(
        quant_config,
        trt_high_precision_dtype,
        quant_algo,
        alpha=alpha,
        lowrank=lowrank,
    )

    def forward_loop(mod):
        if model_name not in ["sd3-medium", "flux-dev", "flux-schnell"]:
            pipe.unet = mod
        else:
            pipe.transformer = mod

        do_calibrate(
            pipe=pipe,
            calibration_prompts=calibration_prompts,
            model_id=model_name,
            calib_size=calib_size,
            n_steps=num_denoising_steps,
        )
    
    # Fuse LoRA layers, then quantize
    check_lora(backbone)
    mtq.quantize(backbone, quant_config, forward_loop)

     # Additional quantization adjustments by level
    quantize_lvl(model_name, backbone, quant_level)

    # Disable some quantizers
    mtq.disable_quantizer(backbone, filter_func)
    
    return backbone 


def main() -> None:
    """
    Main entrypoint for quantizing and calibrating the pipeline, then optionally exporting.
    """
    args = parse_args()

    if args.quant_level == 4.0 and args.format == "int8":
        raise ValueError("Level 4 quantization is only supported for fp8, not int8.")

    # Create pipeline
    pipe = create_pipeline(args.model, args.model_dtype, args.override_model_path)
    pipe.to("cuda") if not args.cpu_offloading else pipe.enable_model_cpu_offload()

    # Choose correct backbone (unet or transformer) for the loaded pipeline
    if args.model not in ["sd3-medium", "flux-dev", "flux-schnell"]:
        backbone = pipe.unet
    else:
        # For SD3 and Flux, apparently your backbone is pipe.transformer
        backbone = pipe.transformer

    # Adjust calibration steps to be number of batches
    args.calib_size = args.calib_size // args.batch_size

    # If restore_from is not specified, run calibration and quantize
    if not args.restore_from:
        backbone = quantize_diffusion_model(pipe, 
                                 backbone, 
                                 model_name=args.model, 
                                 precision=args.format, 
                                 quant_level=args.quant_level, 
                                 calib_size=args.calib_size, 
                                 batch_size=args.batch_size, 
                                 calibration_source="./calib/calib_prompts.txt",
                                 collect_method=args.collect_method,
                                 quant_algo=args.quant_algo,
                                 percentile=args.percentile,
                                 num_denoising_steps=args.n_steps,
                                 trt_high_precision_dtype=args.trt_high_precision_dtype,
                                 alpha=args.alpha,
                                 lowrank=args.lowrank)

        # Save the quantized checkpoint if path is provided
        if args.quantized_torch_ckpt_save_path:
            mto.save(backbone, args.quantized_torch_ckpt_save_path)
    else:
        # Restore the previously quantized model
        mto.restore(backbone, args.restore_from)

        # Additional quantization adjustments by level
        quantize_lvl(args.model, backbone, args.quant_level)

        # Disable some quantizers
        mtq.disable_quantizer(backbone, filter_func)

    # Optional ONNX export
    if args.onnx_dir:
        if args.format == "fp8" and args.model != "flux-dev":
            generate_fp8_scales(backbone)

        pipe.to("cpu")
        torch.cuda.empty_cache()
        backbone.to("cuda")
        if args.quant_level == 4.0:
            fp8_mha_disable(backbone, quantized_mha_output=False)

        backbone.eval()
        with torch.no_grad():
            modelopt_export_sd(backbone, args.onnx_dir, args.model, args.format)

    print("Done!\n")


if __name__ == "__main__":
    main()
