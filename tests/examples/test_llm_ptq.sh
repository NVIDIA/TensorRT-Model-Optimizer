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

# Default to using fast test mode
MODELOPT_FAST_TESTS=${MODELOPT_FAST_TESTS:-true}
if [ "$MODELOPT_FAST_TESTS" = true ]; then
    LLAMA_PATH=/tmp/tiny-random-Llama
    $script_dir/setup_tiny_llama.sh $LLAMA_PATH
else
    LLAMA_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
fi

CALIB=16
CMD="scripts/huggingface_example.sh --calib $CALIB --no-verbose"
CMD_BUILD="$CMD --tasks build"


##### Test Llama #####
$CMD_BUILD --model $LLAMA_PATH --quant fp16
$CMD_BUILD --model $LLAMA_PATH --quant bf16
$CMD_BUILD --model $LLAMA_PATH --quant int8_sq
# sparsegpt test is disable due to some bug when sparsifying models with Pytorch 2.4.0a0+3bcc3cddb5.nv24.7
# $CMD_BUILD --model $LLAMA_PATH --quant int8_sq --sparsity sparsegpt

$CMD_BUILD --model $LLAMA_PATH --quant int4_awq
# Unified checkpoint export path
$CMD_BUILD --model $LLAMA_PATH --quant int4_awq --export_fmt hf

# NVFP4 checkpoint export TRTLLM
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4
# NVFP4 checkpoint export HF
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4 --export_fmt hf

# NVFP4_AWQ checkpoint export TRTLLM
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4_awq
# NVFP4 checkpoint export HF
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4_awq --export_fmt hf

# AutoQ checkpoint export TRTLLM --calib_batch_size
$CMD_BUILD --model $LLAMA_PATH --quant int4_awq,nvfp4,fp8,w4a8_awq --calib_batch_size 4 --effective_bits 6.4
# AutoQ checkpoint export HF
$CMD_BUILD --model $LLAMA_PATH --quant int4_awq,nvfp4,fp8 --calib_batch_size 4 --export_fmt hf --effective_bits 6.4

# Export Regular NVFP4 KV cache
# AutoQ TRTLLM
$CMD_BUILD --model $LLAMA_PATH --quant int4_awq,nvfp4,fp8,w4a8_awq --calib_batch_size 4 --effective_bits 6.4 --kv_cache_quant nvfp4
# AutoQ HF
$CMD_BUILD --model $LLAMA_PATH --quant int4_awq,nvfp4,fp8,w4a8_awq --calib_batch_size 4 --effective_bits 6.4 --kv_cache_quant nvfp4 --export_fmt hf
# Normal quantization
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4_awq --kv_cache_quant nvfp4
# Normal quantization HF
$CMD_BUILD --model $LLAMA_PATH --quant nvfp4_awq --kv_cache_quant nvfp4 --export_fmt hf

# Disable KV cache for fp8
$CMD_BUILD --model $LLAMA_PATH --quant fp8 --kv_cache_quant "none"

if [ $cuda_capability -ge 89 ]; then
    $CMD_BUILD --model $LLAMA_PATH --quant fp8
    # $CMD_BUILD --model $LLAMA_PATH --quant fp8 --sparsity sparsegpt
    # Unified checkpoint export path
    $CMD_BUILD --model $LLAMA_PATH --quant fp8 --export_fmt hf
    $CMD_BUILD --model $LLAMA_PATH --quant w4a8_awq
fi

# Multi GPU case
if [ $nvidia_gpu_count -gt 1 ]; then
    # TP
    $CMD_BUILD --model $LLAMA_PATH --quant fp16 --tp 2
    # $CMD_BUILD --model $LLAMA_PATH --quant fp16 --tp 2 --sparsity sparsegpt

    # TP NVFP4
    $CMD_BUILD --model $LLAMA_PATH --quant nvfp4 --tp 2

    # PP NVFP4
    # $CMD_BUILD --model $LLAMA_PATH --quant nvfp4 --pp 2

    # PP
    # $CMD_BUILD --model $LLAMA_PATH --quant fp16 --pp 2
    # $CMD_BUILD --model $LLAMA_PATH --quant fp16 --tp 1 --pp 2 --sparsity sparsegpt

    $CMD --model $LLAMA_PATH --tasks benchmark --quant fp16 --tp 2
    # $CMD --model $LLAMA_PATH --tasks benchmark --quant fp16 --tp 2 --sparsity sparsegpt
fi


##### Test MOE #####
MIXTRAL_PATH=AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE
$CMD_BUILD --model $MIXTRAL_PATH --quant fp16

if [ $cuda_capability -ge 89 ]; then
    $CMD_BUILD --model $MIXTRAL_PATH --quant fp8
    # Unified checkpoint export path
    $CMD_BUILD --model $MIXTRAL_PATH --quant fp8 --export_fmt hf
fi


##### Test T5 (enc_dec) #####
ENC_DEC_PATH=google-t5/t5-small
# TODO: Enable T5 quantization after the bug is fixed
# $CMD_BUILD --model $ENC_DEC_PATH --quant fp16
# $CMD_BUILD --model $ENC_DEC_PATH --quant fp8


##### Test Bart (enc_dec) #####
ENC_DEC_PATH=facebook/bart-large-cnn
$CMD_BUILD --model $ENC_DEC_PATH --quant fp16
$CMD_BUILD --model $ENC_DEC_PATH --quant fp8


##### Test Whisper (enc_dec) #####
ENC_DEC_PATH=openai/whisper-tiny
pip install -r requirements-whisper.txt
$CMD_BUILD --model $ENC_DEC_PATH --quant fp16
$CMD_BUILD --model $ENC_DEC_PATH --quant fp8


popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
