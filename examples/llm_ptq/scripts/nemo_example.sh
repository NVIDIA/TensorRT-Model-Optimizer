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

source $script_dir/parser.sh
parse_options "$@"

set -x

# This will prevent the script from hanging on EOS due to the MPI support.
echo "********** unset all SLURM_, PMI_, PMIX_ Variables **********"
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

if [ $DEPLOYMENT != "tensorrt_llm" ]; then
    echo "Only support tensorrt_llm deployment."
    exit 1
fi

case $MODEL_TYPE in
    gpt|llama|gemma)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [gpt, llama, gemma]" >&2
        exit 1
esac

if [[ -z "$MODEL_PATH" ]] || [[ ! -f "$MODEL_PATH" ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "Unsupported model argument: Expect a nemo file or a nemo extracted directory" >&2
    echo "E.g. you can download the 2B model with this URL: https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    exit 1
fi

#Check if provided quantization format/list of comma separated quantization formats provided are valid
IFS=','
for qformat in $QFORMAT; do
    case $qformat in
        bf16|fp16|fp8|int8_sq|int4_awq|w4a8_awq|nvfp4)
            ;;
        *)
            echo "Unknown quant argument: Expected one of: [bf16, fp16, fp8, int8_sq, int4_awq, w4a8_awq, nvfp4]" >&2
            exit 1
    esac
done
IFS=" "
if [ -z "$DTYPE" ]; then
    DTYPE="bf16"
fi

#Does not apply for auto quantize case
if [ "$QFORMAT" == "fp16" ] || [ "$QFORMAT" == "bf16" ]; then
    DTYPE=$QFORMAT
    QFORMAT="null"
fi

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
        echo "Unknown pp argument: Expected one of: [1]" >&2
        exit 1
esac

if [ -z "$CALIB_BATCH_SIZE" ]; then
    CALIB_BATCH_SIZE=64
fi

GPU_NAME=$(nvidia-smi --id 0 --query-gpu=name --format=csv,noheader,nounits | sed 's/ /_/g')

echo "Using the following config: max input $BUILD_MAX_INPUT_LEN max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -f "$MODEL_PATH" ]; then
    # Nemo archive: extract config
    tar -xvf $MODEL_PATH $(tar -tf $MODEL_PATH --no-same-owner | grep model_config.yaml) --no-same-owner
elif [ -d "$MODEL_PATH" ]; then
    # Extracted directory: just copy config
    cp $MODEL_PATH/model_config.yaml .
fi

if [ -z "$CALIB_TP" ]; then
    CALIB_TP=$(nvidia-smi --list-gpus | wc -l)
fi

if [ "$CALIB_BATCH_SIZE" -eq 0 ]; then
    CALIB_BATCH_SIZE=64
fi

PREC=$(cat model_config.yaml | grep precision | awk -F':' '{print $NF}' | awk '{$1=$1};1')
PREC=${PREC:-bf16}

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    AUTO_QUANTIZE_ARGS+="quantization.auto_quantize_bits=$AUTO_QUANTIZE_BITS"
fi

QFORMAT_MODIFIED="${QFORMAT//,/_}"
MODEL_NAME=$(basename $MODEL_PATH | sed 's/\.[^.]*$//')
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}_${QFORMAT_MODIFIED}_${DTYPE}_tp${TP}_pp${PP}
MODEL_CONFIG_PTQ=${SAVE_PATH}/config.json
ENGINE_DIR=${SAVE_PATH}/${TP}x${PP}x${GPU_NAME}_input${BUILD_MAX_INPUT_LEN}_output${BUILD_MAX_OUTPUT_LEN}_batch${BUILD_MAX_BATCH_SIZE}_engine
TOKENIZER_CONFIG=${SAVE_PATH}/tokenizer_config.yaml

if [[ $TASKS =~ "build" ]] || [[ ! -d "$ENGINE_DIR" ]] || [[ ! $(ls -A $ENGINE_DIR) ]]; then

    # If you hit storage space issues unpacking $MODEL_PATH
    # you can unpack it beforehand and use the unpacked ckpt.
    if ! [ -f $MODEL_CONFIG_PTQ ]; then
        echo "Quantizing original model..."
        mpirun -n $CALIB_TP --allow-run-as-root python nemo_ptq.py \
            model_file=$MODEL_PATH \
            tensor_model_parallel_size=$(($CALIB_TP)) \
            pipeline_model_parallel_size=1 \
            trainer.devices=$(($CALIB_TP)) \
            trainer.num_nodes=1 \
            trainer.precision=$PREC \
            quantization.algorithm="'$QFORMAT'" \
            $AUTO_QUANTIZE_ARGS \
            quantization.awq_block_size=$(($AWQ_BLOCK_SIZE)) \
            quantization.num_calib_size=$(($CALIB_SIZE)) \
            quantization.disable_kv_cache_quant=$DISABLE_KV_CACHE_QUANT \
            inference.batch_size=$(($CALIB_BATCH_SIZE)) \
            export.path=$SAVE_PATH \
            export.decoder_type=$MODEL_TYPE \
            export.inference_tensor_parallel=$(($TP)) \
            export.inference_pipeline_parallel=$(($PP)) \
            export.dtype=$DTYPE
    else
        echo "Quantized model config $MODEL_CONFIG_PTQ exists, skipping the quantization stage"
    fi

    # Deployment is skipped for auto quantize
    if [[ -n $AUTO_QUANTIZE_BITS ]]; then
        echo "Please build tensorrt_llm engine with this model from the tensorrt_llm repo for deployment. Checkpoint export_path: $SAVE_PATH"
        exit 1
    fi

    echo "Building tensorrt_llm engine from Model Optimizer-quantized model..."

    # By default, the quantized model is deployed to TensorRT-LLM on a single GPU.
    # The --gpus flag can be added on the following command to deploy on multi GPUs.
    python modelopt_to_tensorrt_llm.py \
        --model_config $MODEL_CONFIG_PTQ \
        --engine_dir $ENGINE_DIR \
        --tokenizer $TOKENIZER_CONFIG \
        --max_input_len=$BUILD_MAX_INPUT_LEN \
        --max_output_len=$BUILD_MAX_OUTPUT_LEN \
        --max_batch_size=$BUILD_MAX_BATCH_SIZE \
        --num_build_workers=$GPUS

fi

if [[ $TASKS =~ "mmlu" ]]; then

    MMLU_RESULT=${ENGINE_DIR}/mmlu.txt
    echo "Evaluating MMLU, result saved to $MMLU_RESULT..."

    pushd ../llm_eval/

    pip install -r requirements.txt

    if [ -z "$MMLU_DATA_PATH" ]; then
        MMLU_DATA_PATH=data/mmlu
    fi
    if [[ ! -d "$MMLU_DATA_PATH" ]] || [[ ! $(ls -A $MMLU_DATA_PATH) ]]; then
        echo "Preparing the MMLU test data"
        wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O /tmp/mmlu.tar
        mkdir -p data
        tar -xf /tmp/mmlu.tar -C data && mv data/data $MMLU_DATA_PATH
    fi

    # For handling Nemo tokenizers you can specify either --vocab_file as yaml tokenizer config (recommended)
    # or --model_path in case of HF-converted models.
    python mmlu.py \
        --engine_dir $ENGINE_DIR \
        --vocab_file $TOKENIZER_CONFIG \
        --data_dir $MMLU_DATA_PATH | tee $MMLU_RESULT
    popd

fi

popd
