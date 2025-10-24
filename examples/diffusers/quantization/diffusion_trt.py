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
from onnx_utils.export import (
    generate_dummy_inputs_and_dynamic_axes_and_shapes,
    get_io_shapes,
    remove_nesting,
    update_dynamic_axes,
)
from quantize import ModelType, PipelineManager
from tqdm import tqdm

import modelopt.torch.opt as mto
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy._runtime.tensorrt.constants import SHA_256_HASH_LENGTH
from modelopt.torch._deploy._runtime.tensorrt.tensorrt_utils import prepend_hash_to_bytes
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import get_onnx_bytes_and_metadata

MODEL_ID = {
    "sdxl-1.0": ModelType.SDXL_BASE,
    "sdxl-turbo": ModelType.SDXL_TURBO,
    "sd3-medium": ModelType.SD3_MEDIUM,
    "flux-dev": ModelType.FLUX_DEV,
    "flux-schnell": ModelType.FLUX_SCHNELL,
}

dtype_map = {
    "Half": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float": torch.float32,
}


def generate_image(pipe, prompt, image_name):
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save(image_name)
    print(f"Image generated saved as {image_name}")


def benchmark_model(
    pipe, prompt, num_warmup=10, num_runs=50, num_inference_steps=20, model_dtype="Half"
):
    """Benchmark the backbone model inference time."""
    backbone = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    backbone_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def forward_pre_hook(_module, _input):
        start_event.record()

    def forward_hook(_module, _input, _output):
        end_event.record()
        torch.cuda.synchronize()
        backbone_times.append(start_event.elapsed_time(end_event))

    pre_handle = backbone.register_forward_pre_hook(forward_pre_hook)
    post_handle = backbone.register_forward_hook(forward_hook)

    try:
        print(f"Starting warmup: {num_warmup} runs")
        for _ in tqdm(range(num_warmup), desc="Warmup"):
            with torch.amp.autocast("cuda", dtype=dtype_map[model_dtype]):
                _ = pipe(
                    prompt,
                    output_type="pil",
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(42),
                )

        backbone_times.clear()

        print(f"Starting benchmark: {num_runs} runs")
        for _ in tqdm(range(num_runs), desc="Benchmark"):
            with torch.amp.autocast("cuda", dtype=dtype_map[model_dtype]):
                _ = pipe(
                    prompt,
                    output_type="pil",
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(42),
                )
    finally:
        pre_handle.remove()
        post_handle.remove()

    total_backbone_time = sum(backbone_times)
    avg_latency = total_backbone_time / (num_runs * num_inference_steps)
    print(f"Inference latency of the torch backbone: {avg_latency:.2f} ms")
    return avg_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=["sdxl-1.0", "sdxl-turbo", "sd3-medium", "flux-dev", "flux-schnell"],
    )
    parser.add_argument(
        "--override-model-path",
        type=str,
        default=None,
        help="Path to the model if not using default paths in MODEL_ID mapping.",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="Half",
        choices=["Half", "BFloat16", "Float"],
        help="Precision used to load the model.",
    )
    parser.add_argument(
        "--restore-from", type=str, default=None, help="Path to the modelopt quantized checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="Input text prompt for the model",
    )
    parser.add_argument(
        "--onnx-load-path", type=str, default="", help="Path to load the ONNX model"
    )
    parser.add_argument(
        "--trt-engine-load-path", type=str, default=None, help="Path to load the TensorRT engine"
    )
    parser.add_argument(
        "--dq-only", action="store_true", help="Converts the ONNX model to a dq_only model"
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Use the torch pipeline for image generation or benchmarking",
    )
    parser.add_argument("--save-image-as", type=str, default=None, help="Name of the image to save")
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark the model backbone inference time"
    )
    parser.add_argument(
        "--torch-compile", action="store_true", help="Use torch.compile() on the backbone model"
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip image generation")
    args = parser.parse_args()

    image_name = args.save_image_as if args.save_image_as else f"{args.model}.png"

    pipe = PipelineManager.create_pipeline_from(
        MODEL_ID[args.model],
        dtype_map[args.model_dtype],
        override_model_path=args.override_model_path,
    )

    # Save the backbone of the pipeline and move it to the GPU
    add_embedding = None
    backbone = None
    if hasattr(pipe, "transformer"):
        backbone = pipe.transformer
    elif hasattr(pipe, "unet"):
        backbone = pipe.unet
        add_embedding = backbone.add_embedding
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")

    if args.restore_from:
        mto.restore(backbone, args.restore_from)

    if args.torch_compile:
        assert args.model_dtype in ["BFloat16", "Float", "Half"], (
            "torch.compile() only supports BFloat16 and Float"
        )
        print("Compiling backbone with torch.compile()...")
        backbone = torch.compile(backbone, mode="max-autotune")

    if args.torch:
        if hasattr(pipe, "transformer"):
            pipe.transformer = backbone
        elif hasattr(pipe, "unet"):
            pipe.unet = backbone
        pipe.to("cuda")

        if args.benchmark:
            benchmark_model(pipe, args.prompt, model_dtype=args.model_dtype)

        if not args.skip_image:
            generate_image(pipe, args.prompt, image_name)
        return

    backbone.to("cuda")

    # Generate dummy inputs for the backbone
    dummy_inputs, dynamic_axes, dynamic_shapes = generate_dummy_inputs_and_dynamic_axes_and_shapes(
        args.model, backbone
    )

    # Postprocess the dynamic axes to match the input and output names with DeviceModel
    if args.onnx_load_path == "":
        update_dynamic_axes(args.model, dynamic_axes)

    compilation_args = dynamic_shapes

    # We only need to remove the nesting for SDXL models as they contain the nested input added_cond_kwargs
    # which are renamed by the DeviceModel
    ignore_nesting = False
    if args.onnx_load_path != "" and args.model in ["sdxl-1.0", "sdxl-turbo"]:
        remove_nesting(compilation_args)
        ignore_nesting = True

    # Define deployment configuration
    deployment = {
        "runtime": "TRT",
        "precision": "stronglyTyped",
        "onnx_opset": "17",
        "verbose": "false",
    }

    client = RuntimeRegistry.get(deployment)

    # Export onnx model and get some required names from it
    onnx_bytes, metadata = get_onnx_bytes_and_metadata(
        model=backbone,
        dummy_input=dummy_inputs,
        onnx_load_path=args.onnx_load_path,
        dynamic_axes=dynamic_axes,
        onnx_opset=int(deployment["onnx_opset"]),
        remove_exported_model=False,
        dq_only=args.dq_only,
    )

    if not args.trt_engine_load_path:
        # Compile the TRT engine from the exported ONNX model
        compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
        # Save TRT engine for future use
        with open(f"{args.model}.plan", "wb") as f:
            # Remove the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            f.write(compiled_model[SHA_256_HASH_LENGTH:])
    else:
        with open(args.trt_engine_load_path, "rb") as f:
            compiled_model = f.read()
            # Prepend the SHA-256 hash from the compiled model, used to maintain state in the trt_client
            compiled_model = prepend_hash_to_bytes(compiled_model)

    # The output shapes will need to be specified for models with dynamic output dimensions
    device_model = DeviceModel(
        client,
        compiled_model,
        metadata,
        compilation_args,
        get_io_shapes(args.model, args.onnx_load_path, dynamic_shapes),
        ignore_nesting,
    )

    if hasattr(pipe, "unet") and add_embedding:
        setattr(device_model, "add_embedding", add_embedding)

    # Move the backbone back to the CPU and set the backbone to the compiled device model
    backbone.to("cpu")
    if hasattr(pipe, "unet"):
        pipe.unet = device_model
    elif hasattr(pipe, "transformer"):
        pipe.transformer = device_model
    else:
        raise ValueError("Pipeline does not have a transformer or unet backbone")
    pipe.to("cuda")

    if not args.skip_image:
        generate_image(pipe, args.prompt, image_name)
        print(f"Image generated using {args.model} model saved as {image_name}")

    if args.benchmark:
        print(
            f"Inference latency of the TensorRT optimized backbone: {device_model.get_latency()} ms"
        )


if __name__ == "__main__":
    main()
