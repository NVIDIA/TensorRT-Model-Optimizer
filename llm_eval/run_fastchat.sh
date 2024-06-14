#!/bin/bash

# This script can be used to generate responses for MT-Bench using FastChat.
# Ensure that you are logged in to huggingface-cli using: huggingface-cli login.
#
# Usage: bash run_fastchat.sh <HF model folder or model card> <engine_dir>
# model_name: The HuggingFace handle or folder of the model to evaluate.
# engine_dir: The directory where the TRT-LLM engine is stored.
#
# Example commands:
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model:
# bash run_fastchat.sh meta-llama/Meta-Llama-3-8B-Instruct
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model with TRT-LLM engine:
# bash run_fastchat.sh meta-llama/Meta-Llama-3-8B-Instruct /path/to/engine_dir


set -e
set -x

model_name="$1"
engine_dir="$2"

# Clone FastChat repository
if [ ! -d FastChat ]; then
    echo "FastChat repository not found. Cloning FastChat..."
    git clone https://github.com/lm-sys/FastChat.git
    mkdir -p ./data/mt_bench
    cp FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl ./data/mt_bench/question.jsonl
else
    echo "FastChat repository found. Skipping cloning."
fi

# Install FastChat with variants
pushd FastChat
pip install -e ".[model_worker,llm_judge]" -U
popd

PYTHONPATH=FastChat:$PYTHONPATH python gen_model_answer.py \
    --model-path $model_name \
    --model-id $model_name \
    --engine-dir $engine_dir
