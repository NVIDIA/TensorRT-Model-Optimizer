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
nvidia_gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
cuda_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)


pushd $script_dir/../../examples/vlm_ptq

TASKS="build"
CALIB=32
TYPE=phi
MODEL=microsoft/Phi-3.5-vision-instruct
common_args="--type $TYPE --model $MODEL --task $TASKS --calib $CALIB --trust_remote_code"

pip install -r requirements-phi3.5.txt

# Single GPU case
scripts/huggingface_example.sh --quant fp16 $common_args
scripts/huggingface_example.sh --quant bf16 $common_args

# Model Optimizer quantization
scripts/huggingface_example.sh --quant int8_sq $common_args
scripts/huggingface_example.sh --quant int4_awq $common_args
if [ $cuda_capability -ge 89 ]; then
    scripts/huggingface_example.sh --quant fp8 $common_args
    scripts/huggingface_example.sh --quant w4a8_awq $common_args
fi

# Multi GPU case
if [ $nvidia_gpu_count -gt 1 ] && [ -z "$SINGLE_GPU" ]; then
    # TP
   scripts/huggingface_example.sh --type llava --model llava-hf/llava-1.5-7b-hf --quant fp16 --tp 2 --task $TASKS --calib $CALIB
fi

popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
