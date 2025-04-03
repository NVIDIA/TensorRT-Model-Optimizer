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

script_dir="$(dirname "$(readlink -f "$0")")"

source $script_dir/../../llm_ptq/scripts/parser.sh
parse_options "$@"

set -x

# This will prevent the script from hanging on Selene/EOS due to the MPI support.
echo "********** unset all SLURM_, PMI_, PMIX_ Variables **********"
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

case $MODEL_TYPE in
    llava|phi|vila|mllama)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [llava, phi, vila, mllama]" >&2
        exit 1
esac

if [ -z "$MODEL_PATH" ]; then
    echo "Unsupported model argument: Expected a huggingface model path or model name or a nemo path" >&2
    exit 1
fi

# Check if ENABLE_SPARSITY environment variable is set to "true"
if [ "$SPARSITY_FMT" = "dense" ]; then
    ENABLE_SPARSITY=false
else
    ENABLE_SPARSITY=true
fi

case $SPARSITY_FMT in
    dense|sparsegpt)
        ;;
    *)
        echo "Unknown sparsity argument: Expected one of: [dense, sparsegpt]" >&2
        exit 1
esac

case $QFORMAT in
    fp8|int8_sq|int4_awq|w4a8_awq|fp16|bf16)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, int8_sq, int4_awq, w4a8_awq, fp16, bf16]" >&2
        exit 1
esac

case $TP in
    1|2|4|8)
        ;;
    *)
        echo "Unknown tp argument: Expected one of: [1, 2, 4, 8]" >&2
        exit 1
esac

case $PP in
    1|2|4|8)
        ;;
    *)
        echo "Unknown pp argument: Expected one of: [1, 2, 4, 8]" >&2
        exit 1
esac

GPU_NAME=$(nvidia-smi --id 0 --query-gpu=name --format=csv,noheader,nounits | sed 's/ /_/g')

if [ "${MODEL_TYPE}" = "phi" ]; then
    BUILD_MAX_INPUT_LEN=4096
else
    BUILD_MAX_INPUT_LEN=1024
fi

BUILD_MAX_OUTPUT_LEN=512

if [ "$MODEL_TYPE" = "llava" ] || [ "$MODEL_TYPE" = "vila" ]; then
    BUILD_MAX_BATCH_SIZE=20
else
    BUILD_MAX_BATCH_SIZE=4
fi


echo "Using the following config: max input $BUILD_MAX_INPUT_LEN max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

MODEL_NAME=$(basename $MODEL_PATH | sed 's/[^0-9a-zA-Z\-]/_/g')
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}_${SPARSITY_FMT}_${QFORMAT}_tp${TP}_pp${PP}

if [ $EXPORT_FORMAT != "tensorrt_llm" ]; then
    SAVE_PATH=${SAVE_PATH}_${EXPORT_FORMAT}
fi

MODEL_CONFIG=${SAVE_PATH}/config.json
ENGINE_DIR=${SAVE_PATH}/${MODEL_TYPE}_${TP}x${PP}x${GPU_NAME}_input${BUILD_MAX_INPUT_LEN}_output${BUILD_MAX_OUTPUT_LEN}_batch${BUILD_MAX_BATCH_SIZE}_engine

if [ "${REMOVE_EXISTING_MODEL_CONFIG,,}" = "true" ]; then
    rm -f $MODEL_CONFIG
fi

PTQ_ARGS=""

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    PTQ_ARGS+=" --auto_quantize_bits $AUTO_QUANTIZE_BITS "
fi

if $TRUST_REMOTE_CODE; then
    PTQ_ARGS+=" --trust_remote_code "
fi

if [ "${MODEL_TYPE}" = "mllama" ]; then
    PTQ_ARGS+=" --dataset scienceqa "
fi

case "${MODEL_TYPE}" in
    "vila")
        VISUAL_FEATURE=196
        VLM_ARGS=" --max_multimodal_len=$((BUILD_MAX_BATCH_SIZE * VISUAL_FEATURE)) "
        ;;
    "phi")
        VISUAL_FEATURE=4096
        VLM_ARGS=" --max_multimodal_len=$((BUILD_MAX_BATCH_SIZE * VISUAL_FEATURE)) "
        ;;
    "llava")
        VISUAL_FEATURE=576
        VLM_ARGS=" --max_multimodal_len=$((BUILD_MAX_BATCH_SIZE * VISUAL_FEATURE)) "
        ;;
    "mllama")
        VLM_ARGS=" --max_encoder_input_len=6404 --skip_run"
        ;;
esac

if [ "${MODEL_TYPE}" = "vila" ]; then
    # Install required dependency for VILA
    pip install -r ../vlm_ptq/requirements-vila.txt
    # Clone oringinal VILA repo
    if [ ! -d "$(dirname "$MODEL_PATH")/VILA" ]; then
        echo "VILA repository is needed until it is added to HF model zoo. Cloning the repository parallel to $MODEL_PATH..."
        git clone https://github.com/Efficient-Large-Model/VILA.git "$(dirname "$MODEL_PATH")/VILA" && \
	cd "$(dirname "$MODEL_PATH")/VILA" && \
	git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5 && \
	cd "$script_dir/.."
    fi
fi

if [[ $TASKS =~ "build" ]] || [[ ! -d "$ENGINE_DIR" ]] || [[ ! $(ls -A $ENGINE_DIR) ]]; then
    if ! [ -f $MODEL_CONFIG ]; then
        echo "Quantizing original model..."
        python ../llm_ptq/hf_ptq.py \
            --pyt_ckpt_path=$MODEL_PATH \
            --export_path=$SAVE_PATH \
            --sparsity_fmt=$SPARSITY_FMT \
            --qformat=$QFORMAT \
            --calib_size=$CALIB_SIZE \
            --batch_size=$CALIB_BATCH_SIZE \
            --inference_tensor_parallel=$TP \
            --inference_pipeline_parallel=$PP \
            --export_fmt=$EXPORT_FORMAT \
            --vlm \
            $PTQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi

    if [ $EXPORT_FORMAT != "tensorrt_llm" ]; then
        echo "Please continue deployment with $EXPORT_FORMAT. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi


    echo "Building tensorrt_llm engine from Model Optimizer-quantized model..."

    python ../llm_ptq/modelopt_to_tensorrt_llm.py \
        --model_config=$MODEL_CONFIG \
        --engine_dir=$ENGINE_DIR \
        --tokenizer=$MODEL_PATH \
        --max_input_len=$BUILD_MAX_INPUT_LEN \
        --max_output_len=$BUILD_MAX_OUTPUT_LEN \
        --max_batch_size=$BUILD_MAX_BATCH_SIZE \
        --num_build_workers=$GPUS \
        --enable_sparsity=$ENABLE_SPARSITY \
        $VLM_ARGS
fi


VISUAL_ARGS=""
VISION_ENCODER_DIR=${SAVE_PATH}/vision_encoder
VISUAL_MODEL_TYPE=$MODEL_TYPE
case "${MODEL_TYPE}" in
    "vila")
        VISUAL_ARGS+=" --vila_path ${MODEL_PATH}/../VILA "
        ;;
    "phi")
        VISUAL_MODEL_TYPE="phi-3-vision"
        ;;
esac


VISUAL_MAX_BATCH_SIZE=$BUILD_MAX_BATCH_SIZE

if [[ $TASKS =~ "build" ]] || [[ ! -d "$VISION_ENCODER_DIR" ]] || [[ ! $(ls -A $VISION_ENCODER_DIR) ]]; then
    echo "Build visual engine"
    python vlm_visual_engine.py \
        --model_path $MODEL_PATH \
        --model_type $VISUAL_MODEL_TYPE \
        --output_dir $VISION_ENCODER_DIR \
        --max_batch_size $VISUAL_MAX_BATCH_SIZE \
        $VISUAL_ARGS
fi

VLM_RUN_ARGS=""
case "${MODEL_TYPE}" in
    "mllama")
        VLM_RUN_ARGS+=" --visual_engine_name visual_encoder.engine --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg --input_text \"<|image|><|begin_of_text|>If I had to write a haiku for this one\" --max_new_tokens 50 --batch_size 2 "
        ;;
esac
echo "Run inference example"

mpirun -n $GPUS --allow-run-as-root python vlm_run.py  \
    --hf_model_dir $MODEL_PATH \
    --visual_engine_dir $VISION_ENCODER_DIR \
    --llm_engine_dir $ENGINE_DIR \
    --kv_cache_free_gpu_memory_fraction $KV_CACHE_FREE_GPU_MEMORY_FRACTION \
    $VLM_RUN_ARGS

if [[ $TASKS =~ "gqa" ]]; then
    echo "Evaluating the TensorRT engine of the quantized model using GQA benchmark."
    pushd ../vlm_eval/
    if [[ "$MODEL_PATH" =~ ^/ ]]; then
        # If MODEL_PATH is absolute path
        source gqa.sh --hf_model $MODEL_PATH --llm_engine $ENGINE_DIR --visual_engine $VISION_ENCODER_DIR --kv_cache_free_gpu_memory_fraction $KV_CACHE_FREE_GPU_MEMORY_FRACTION
    else
        # If MODEL_PATH is absolute path
        script_parent_dir=$(dirname "$script_dir")
        source gqa.sh --hf_model $script_parent_dir/$MODEL_PATH --llm_engine $ENGINE_DIR --visual_engine $VISION_ENCODER_DIR --kv_cache_free_gpu_memory_fraction $KV_CACHE_FREE_GPU_MEMORY_FRACTION
    fi

    popd
fi

popd
