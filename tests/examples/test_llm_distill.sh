#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


pushd $script_dir/../../examples/llm_distill

LLAMA_PATH=/tmp/tiny-random-Llama
$script_dir/setup_tiny_llama.sh $LLAMA_PATH

SAVE_PATH=/tmp/llm_distill_test_output

accelerate launch --multi_gpu --mixed_precision bf16 main.py \
    --teacher_name_or_path $LLAMA_PATH \
    --student_name_or_path $LLAMA_PATH \
    --output_dir $SAVE_PATH \
    --logging_steps 5 \
    --max_steps 10 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_checkpointing True \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer

popd

echo "Total wall time: $(($(date +%s) - start_time)) seconds"
