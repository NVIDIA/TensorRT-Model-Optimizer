#!/bin/bash
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

set -e
set -x
set -o pipefail

start_time=$(date +%s)
script_dir="$(dirname "$(readlink -f "$0")")"
cuda_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)


# Use the tiny models for faster testing
# NOTE: For tiny sdxl and sd3, we have to use Float dtype instead of Half from transformers v4.49.0
SDXL="--model sdxl-1.0 --override-model-path hf-internal-testing/tiny-sdxl-pipe"
SDXL_DTYPE="Float"
SD3="--model sd3-medium --override-model-path hf-internal-testing/tiny-sd3-pipe"
SD3_DTYPE="Float"
FLUX="--model flux-schnell --override-model-path hf-internal-testing/tiny-flux-pipe"
FLUX_DTYPE="BFloat16"


##### Quantization #####
pushd $script_dir/../../examples/diffusers/quantization
INT8_ARGS="--format int8 --calib-size 8 --collect-method min-mean --percentile 1.0 --alpha 0.8 --quant-level 3.0 --n-steps 20 --quant-algo smoothquant"
FP8_ARGS="--format fp8 --calib-size 8 --quant-level 3.0 --n-steps 20 --batch-size 2"

# SDXL FP8 Level 3 + Restore
python quantize.py $SDXL --model-dtype $SDXL_DTYPE --trt-high-precision-dtype $SDXL_DTYPE $FP8_ARGS --quantized-torch-ckpt-save-path sdxl.fp8.level3.pt --onnx-dir sdxl.fp8.level3.onnx
python quantize.py $SDXL --model-dtype $SDXL_DTYPE --trt-high-precision-dtype $SDXL_DTYPE $FP8_ARGS --restore-from sdxl.fp8.level3.pt --onnx-dir sdxl.fp8.level3.restore.onnx

# SDXL INT8 Level 3
python quantize.py $SDXL --model-dtype $SDXL_DTYPE --trt-high-precision-dtype $SDXL_DTYPE $INT8_ARGS --batch-size 2 --quantized-torch-ckpt-save-path sdxl.int8.level3.pt --onnx-dir sdxl.int8.level3.onnx

# SD3 Medium INT8 Level 3 + Restore
python quantize.py $SD3 --model-dtype $SD3_DTYPE --trt-high-precision-dtype $SD3_DTYPE $INT8_ARGS --batch-size 2 --quantized-torch-ckpt-save-path sd3-medium.int8.level3.pt --onnx-dir sd3-medium.int8.level3.onnx
python quantize.py $SD3 --model-dtype $SD3_DTYPE --trt-high-precision-dtype $SD3_DTYPE $INT8_ARGS --restore-from sd3-medium.int8.level3.pt --onnx-dir sd3-medium.int8.level3.restore.onnx

# Flux INT8 level 3
python quantize.py $FLUX --model-dtype $FLUX_DTYPE --trt-high-precision-dtype $FLUX_DTYPE $INT8_ARGS --batch-size 1 --quantized-torch-ckpt-save-path flux-schnell.int8.level3.pt --onnx-dir flux-schnell.int8.level3.onnx


##### Inference #####
# INT8
python diffusion_trt.py $SDXL --model-dtype $SDXL_DTYPE --restore-from sdxl.int8.level3.pt
rm -rf build

# INT8 DQ only models
python diffusion_trt.py $SDXL --model-dtype $SDXL_DTYPE --onnx-load-path sdxl.int8.level3.onnx/model.onnx --dq-only
python diffusion_trt.py $SD3 --model-dtype $SD3_DTYPE --onnx-load-path sd3-medium.int8.level3.onnx/model.onnx --dq-only
python diffusion_trt.py $FLUX --model-dtype $FLUX_DTYPE --onnx-load-path flux-schnell.int8.level3.onnx/model.onnx --dq-only

# FP8 DQ only models
if [ $cuda_capability -ge 89 ]; then
    python diffusion_trt.py $SDXL --model-dtype $SDXL_DTYPE --onnx-load-path sdxl.fp8.level3.onnx/model.onnx --dq-only
fi

popd


##### Cache Diffusion #####
pushd $script_dir/../../examples/diffusers/cache_diffusion

# SDXL
python -c "
import torch
from cache_diffusion import cachify
from cache_diffusion.utils import SDXL_DEFAULT_CONFIG
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    variant='fp16',
    use_safetensors=True,
).to('cuda')
cachify.prepare(pipe, SDXL_DEFAULT_CONFIG)

prompt = 'A random person with a head that is made of flowers, photo by James C. Leyendecker, \
        Afrofuturism, studio portrait, dynamic pose, national geographic photo, retrofuturism, biomorphicy'
generator = torch.Generator(device='cuda').manual_seed(2946901)
pipe(prompt=prompt, num_inference_steps=30, generator=generator).images[0]
"

# PIXART
python -c "
import torch
from cache_diffusion import cachify
from cache_diffusion.utils import PIXART_DEFAULT_CONFIG
from diffusers import PixArtAlphaPipeline

pipe = PixArtAlphaPipeline.from_pretrained(
    'PixArt-alpha/PixArt-XL-2-1024-MS', torch_dtype=torch.float16
).to('cuda')
cachify.prepare(pipe, PIXART_DEFAULT_CONFIG)

prompt = 'a small cactus with a happy face in the Sahara desert'
generator = torch.Generator(device='cuda').manual_seed(2946901)
pipe(prompt=prompt, generator=generator, num_inference_steps=30).images[0]
"

# SDXL
python benchmarks.py --model-id sdxl

popd


echo "Total wall time: $(($(date +%s) - start_time)) seconds"
