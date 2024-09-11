#!/bin/bash

# This script can be used to generate responses for MT-Bench using FastChat.
# Ensure that you are logged in to huggingface-cli using: huggingface-cli login.
# If you are using NIM, ensure that you export the NIM API key using:
# export OPENAI_API_KEY=<NIM_API_KEY>
#
# Usage: bash run_fastchat.sh -h <HF model folder or model card> -e <engine_dir> -n <NIM model model card>
# model_name: The HuggingFace handle or folder of the model to evaluate.
# engine_dir: The directory where the TRT-LLM engine is stored.
# nim_model_name: The handle of the NIM model to be used for evaluation.
#
# Example commands:
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model with TRT-LLM engine:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct -e /path/to/engine_dir
#
# Evaluate "meta-llama/Meta-Llama-3-8B-Instruct" HF model with NIM:
# bash run_fastchat.sh -h meta-llama/Meta-Llama-3-8B-Instruct -n meta-llama/Meta-Llama-3-8B-Instruct


set -e
set -x

hf_model_name=""
engine_dir=""
nim_model_name=""

# Parse command line arguments
while [[ "$1" != "" ]]; do
    case $1 in
        -h | --hf_model_name )
            shift
            hf_model_name=$1
            ;;
        -e | --engine_dir )
            shift
            engine_dir=$1
            ;;
        -n | --nim_model_name )
            shift
            nim_model_name=$1
            ;;
        * )
            usage
            exit 1
    esac
    shift
done

if [ "$hf_model_name" == "" ]; then
    echo "Please provide the HF model name and NIM model name."
    exit 1
fi

if [ "$engine_dir" != "" ]; then
    engine_dir=" --engine-dir $engine_dir "
fi

if [ "$nim_model_name" != "" ]; then
    nim_model_name=" --nim-model $nim_model_name "
fi

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
    --model-path $hf_model_name \
    --model-id $hf_model_name \
    --temperature 0.0001 \
    --top-p 0.0001 \
    $engine_dir \
    $nim_model_name
