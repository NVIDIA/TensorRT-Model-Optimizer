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


pushd $script_dir/../../examples/llm_qat

LLAMA_PATH=/tmp/tiny-random-Llama
$script_dir/setup_tiny_llama.sh $LLAMA_PATH

common_args="--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer --num_epochs 0.3 --lr 1e-5 --save_steps 5 --calib_size 64"

# Run PTQ
./launch.sh --model $LLAMA_PATH \
            $common_args \
            --do_train False \
            --quant_cfg INT4_WEIGHT_INT8_ACTIVATIONS \
            --output_dir tinyllama-int4-int8-ptq

# Run QAT on PTQ checkpoint
./launch.sh --model tinyllama-int4-int8-ptq \
            $common_args \
            --do_train True \
            --output_dir tinyllama-int4-int8-qat

# Run LoRA-QAT
./launch.sh --model $LLAMA_PATH \
            $common_args \
            --do_train True \
            --quant_cfg NVFP4_DEFAULT_CFG \
            --lora True \
            --output_dir tinyllama-fp4-lora-qat

# TODO: Fix QLoRa test failure
# Run QLoRA
# NOTE: FSDP is not supported with compression and disabled in launch.sh
# ./launch.sh --model $LLAMA_PATH \
#             $common_args \
#             --do_train True \
#             --quant_cfg NVFP4_DEFAULT_CFG \
#             --lora True \
#             --compress True \
#             --output_dir tinyllama-fp4-qlora

popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
