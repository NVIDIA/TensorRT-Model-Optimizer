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


pushd $script_dir/../../examples/llm_ptq

CALIB=64
LLAMA_PATH=/tmp/tiny-random-Llama
$script_dir/setup_tiny_llama.sh $LLAMA_PATH
COMMON_LLMA_ARGS="--model $LLAMA_PATH --calib $CALIB"

# Test evals
if [ $cuda_capability -ge 89 ]; then
    (scripts/huggingface_example.sh $COMMON_LLMA_ARGS --quant fp8 --task "mmlu,lm_eval,simple_eval,benchmark" --lm_eval_tasks hellaswag,gsm8k --simple_eval_tasks humaneval --lm_eval_limit 0.1)
    # Force kill llm-serve if leftover
    if [ $? -ne 0 ]; then pkill -f llm-serve; exit 1; fi
fi

popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
