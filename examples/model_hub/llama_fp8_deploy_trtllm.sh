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

# Check if enough arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <HF_CKPT_DIR> <TensorRT_LLM_DIR>"
    exit 1
fi

# Set directories based on user input
HF_CKPT_DIR="$1"
TensorRT_LLM_DIR="$2"

echo "Using Hugging Face checkpoint directory: $HF_CKPT_DIR"
echo "Using TensorRT-LLM directory: $TensorRT_LLM_DIR"

CHECKPOINT_DIR="./ckpt"
ENGINE_OUTPUT_DIR="./engine"
CONVERT_SCRIPT="$TensorRT_LLM_DIR/examples/llama/convert_checkpoint.py"
INFERENCE_SCRIPT="$TensorRT_LLM_DIR/examples/run.py"
MAX_BATCH_SIZE=1

# Attempt to grant write permission, if necessary
TARGET_DIR="."
if [ ! -w "$TARGET_DIR" ]; then
    echo "Trying to grant write permission to $TARGET_DIR..."
    chmod u+w "$TARGET_DIR" || {
        echo "Error: Could not grant write permission to $TARGET_DIR. Please check permissions or run the script in a writable directory."
        exit 1
    }
fi

# Convert model checkpoint for TRT-LLM deployment
echo "Converting quantized model checkpoint for TRT-LLM deployment..."
python $CONVERT_SCRIPT --model_dir $HF_CKPT_DIR --output_dir $CHECKPOINT_DIR --use_fp8 || {
    echo "Error during checkpoint conversion."
    exit 1
}

# Build TensorRT-LLM engines
echo "Building TensorRT-LLM engines..."
trtllm-build --checkpoint_dir $CHECKPOINT_DIR --output_dir $ENGINE_OUTPUT_DIR --max_batch_size $MAX_BATCH_SIZE || {
    echo "Error building TensorRT engines."
    exit 1
}

# Run inference with TRT-LLM engine
echo "Running inference with TRT-LLM engine..."
python $INFERENCE_SCRIPT --engine_dir $ENGINE_OUTPUT_DIR --max_output_len=50 --tokenizer_dir $HF_CKPT_DIR || {
    echo "Error running inference."
    exit 1
}
