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


pushd $script_dir/../../examples/speculative_decoding

LLAMA_PATH=/tmp/tiny-random-Llama
$script_dir/setup_tiny_llama.sh $LLAMA_PATH

if [ ! -d "Daring-Anteater" ]; then
    git clone https://huggingface.co/datasets/nvidia/Daring-Anteater
fi

COMMON_ARGS="--model $LLAMA_PATH --data Daring-Anteater/train.jsonl --num_epochs 0.005 --lr 1e-5 --save_steps 50 --do_eval False --num_gpu $nvidia_gpu_count"

# Test Medusa
./launch.sh $COMMON_ARGS \
            --mode medusa \
            --output_dir medusa-tinyllama \
            --medusa_num_heads 2 --medusa_num_layers 1


# Test EAGLE
./launch.sh $COMMON_ARGS \
            --mode eagle \
            --output_dir eagle-tinyllama \
            --eagle_num_layers 1


# Test PTQ on Medusa
pushd $script_dir/../../examples/llm_ptq
python hf_ptq.py --pyt_ckpt_path ../speculative_decoding/medusa-tinyllama \
                 --batch_size 1 --calib_size 64 --export_path medusa-tinyllama-trtllm \
                 --qformat fp8 --export_fmt tensorrt_llm
python hf_ptq.py --pyt_ckpt_path ../speculative_decoding/medusa-tinyllama \
                 --batch_size 1 --calib_size 64 --export_path medusa-tinyllama-hf \
                 --qformat fp8 --export_fmt hf

popd


# Test QAT on Medusa
pushd $script_dir/../../examples/llm_qat
./launch.sh --model ../speculative_decoding/medusa-tinyllama \
            --num_epochs 0.1 --lr 1e-5 --save_steps 5 \
            --output_dir medusa-tinyllama-qat-finetune \
            --quant_cfg 'FP8_DEFAULT_CFG' --calib_size 64
popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
