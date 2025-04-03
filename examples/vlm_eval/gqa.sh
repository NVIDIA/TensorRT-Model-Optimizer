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

# Download dataset
script_dir="$(dirname "$(readlink -f "$0")")"

EVAL_FILE="$script_dir/eval.py"
if [ ! -f $EVAL_FILE ]; then
    echo "$EVAL_FILE does not exist. Downloading this file from https://nlp.stanford.edu/data/gqa/eval.zip."
    wget https://nlp.stanford.edu/data/gqa/eval.zip
    unzip eval.zip "eval.py" -d .
    rm eval.zip

    # Changes to eval.py due to the missing assets in GQA v1.2 release
    sed -i '77s/{tier}_all_questions.json/{tier}_questions.json/' "$EVAL_FILE"
    sed -i '119,120s/^/# /' "$EVAL_FILE"
    sed -i '126,128s/^/# /' "$EVAL_FILE"
    sed -i '367,373s/^/# /' "$EVAL_FILE"
    sed -i '376,379s/^/# /' "$EVAL_FILE"
    sed -i '388s/^/# /' "$EVAL_FILE"
fi

gqa_data=$script_dir/gqa/data
QUESTION=$gqa_data/testdev_balanced_questions.json
if [ ! -f $QUESTION ]; then
    echo "$QUESTION does not exist. Downloading this file from https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip."
    wget -P $gqa_data https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
    unzip $gqa_data/questions1.2.zip "testdev_balanced_questions.json" -d $gqa_data
    rm $gqa_data/questions1.2.zip
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_model)
            HF_MODEL_DIR="$2"
            shift 2
            ;;
        --visual_engine)
            VISUAL_ENGINE_DIR="$2"
            shift 2
            ;;
        --llm_engine)
            LLM_ENGINE_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --quant_cfg)
            QUANT_CFG="$2"
            shift 2
            ;;
        --kv_cache_free_gpu_memory_fraction)
            KV_CACHE_FREE_GPU_MEMORY_FRACTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Set default value for kv_cache_free_gpu_memory_fraction if not provided
if [ -z "$KV_CACHE_FREE_GPU_MEMORY_FRACTION" ]; then
    KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.8
fi

# Verify required arguments are set
if [ -z "$HF_MODEL_DIR" ]; then
    echo "Error: Missing required argument --hf_model"
    exit 1
fi

MODEL_NAME=$(basename $HF_MODEL_DIR | sed 's/[^0-9a-zA-Z\-]/_/g' | tr 'A-Z' 'a-z')

if [[ "$MODEL_NAME" == *"vila"* ]] && [[ -z "$LLM_ENGINE_DIR" ]]; then
    # Install required dependency for VILA
    pip install -r requirements-vila.txt
    # Clone oringinal VILA repo
    if [ ! -d "$(dirname "$HF_MODEL_DIR")/VILA" ]; then
        echo "VILA repository is needed until it is added to HF model zoo. Cloning the repository parallel to $HF_MODEL_DIR..."
        git clone https://github.com/Efficient-Large-Model/VILA.git "$(dirname "$HF_MODEL_DIR")/VILA" && \
	    cd "$(dirname "$HF_MODEL_DIR")/VILA" && \
	    git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5 && \
	    cd -
    fi
fi

# Set batch size defaulted to 20 for VILA and Llava
if [[ -z "$BATCH_SIZE" && ("$MODEL_NAME" == *"vila"* || "$MODEL_NAME" == *"llava"*) ]]; then
    BATCH_SIZE=20
fi

# Check if TRT engine is provided
if [ -z "$VISUAL_ENGINE_DIR" ] || [ -z "$LLM_ENGINE_DIR" ]; then
    echo "Either --visual_engine or --llm_engine not provided, evaluation will be based on Pytorch."
    if [ -z "$QUANT_CFG" ]; then
        ANSWER_DIR="$script_dir/gqa/$MODEL_NAME/llava_gqa_testdev_balanced/answers"
        ANSWERS_FILE="$ANSWER_DIR/merge.jsonl"
    else
        ANSWER_DIR="$script_dir/gqa/${MODEL_NAME}_${QUANT_CFG}/llava_gqa_testdev_balanced/answers"
        ANSWERS_FILE="$ANSWER_DIR/merge.jsonl"
    fi
else
    echo "Both --visual_engine or --llm_engine are provided, evaluation will be based on TRT engine."
    ANSWER_DIR="$script_dir/gqa/$(basename $LLM_ENGINE_DIR)/llava_gqa_testdev_balanced/answers"
    ANSWERS_FILE="$ANSWER_DIR/merge.jsonl"
fi

# Run the Python script with the parsed arguments
if [ ! -f $ANSWERS_FILE ]; then
    python model_gqa_loader.py \
        --answers_file "$ANSWERS_FILE" \
        --hf_model_dir "$HF_MODEL_DIR" \
        ${VISUAL_ENGINE_DIR:+--visual_engine_dir "$VISUAL_ENGINE_DIR"} \
        ${LLM_ENGINE_DIR:+--llm_engine_dir "$LLM_ENGINE_DIR"} \
        ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
        ${QUANT_CFG:+--quant_cfg "$QUANT_CFG"} \
        --kv_cache_free_gpu_memory_fraction "$KV_CACHE_FREE_GPU_MEMORY_FRACTION"
fi

# Convert answer to prediction for evaluation
PREDICTION_FILE="$ANSWER_DIR/testdev_balanced_predictions.json"
if [ ! -f $PREDICTION_FILE ]; then
    python convert_gqa_for_eval.py \
        --src $ANSWERS_FILE \
        --dst $PREDICTION_FILE
fi

# Get evaluation result
python eval.py \
    --tier "$gqa_data/testdev_balanced" \
    --predictions $PREDICTION_FILE
