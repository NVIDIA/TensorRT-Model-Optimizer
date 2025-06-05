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
import time
from pathlib import Path

import torch
from cache_diffusion import cachify
from cache_diffusion.utils import SD3_DEFAULT_CONFIG, SDXL_DEFAULT_CONFIG
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
from pipeline.deploy import compile, teardown

MODEL_IDS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="sdxl", choices=["sdxl", "sd3-medium"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--num-iter", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            print(f"Parsed args -- {key}: {value}")
    return args


def main(args):
    if args.model_id == "sdxl":
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_IDS[args.model_id],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_IDS[args.model_id], torch_dtype=torch.float16
        )
    pipe = pipe.to("cuda")

    prompt = "A dog"

    compile(
        pipe,
        args.model_id,
        onnx_path=Path(args.output_dir, "onnx"),
        engine_path=Path(args.output_dir, "engine"),
        batch_size=args.batch_size,
    )

    cachify.prepare(pipe, SDXL_DEFAULT_CONFIG if args.model_id else SD3_DEFAULT_CONFIG)

    generator = torch.Generator(device="cuda").manual_seed(2946901)
    total_time = 0
    cachify.disable(pipe)
    for _ in range(args.num_iter):
        with torch.autocast("cuda"):
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
        with torch.autocast("cuda"):
            start_time = time.time()
            _ = pipe(
                prompt=[prompt] * args.batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )
            end_time = time.time()
        cachify.reset_status(pipe)
        total_time += end_time - start_time
    total_time = total_time / args.num_iter
    latency = total_time / args.batch_size
    print(f"TRT Enabled Cache: {latency}")
    teardown(pipe)


if __name__ == "__main__":
    args = parse_args()
    main(args)
