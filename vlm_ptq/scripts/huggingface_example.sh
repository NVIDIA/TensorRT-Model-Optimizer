#!/bin/bash
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
    llava|phi|vila)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [llava, phi, vila]" >&2
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
    fp8|fp8_naive|int8_sq|int4_awq|w4a8_awq|fp16)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, fp8_naive, int8_sq, int4_awq, w4a8_awq, fp16]" >&2
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
    BUILD_MAX_OUTPUT_LEN=4608
else
    BUILD_MAX_OUTPUT_LEN=2560
fi

BUILD_MAX_BATCH_SIZE=1

echo "Using the following config: max input $BUILD_MAX_INPUT_LEN max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

MODEL_NAME=$(basename $MODEL_PATH | sed 's/[^0-9a-zA-Z\-]/_/g')
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}_${SPARSITY_FMT}_${QFORMAT}_tp${TP}_pp${PP}

if [ $DEPLOYMENT != "tensorrt_llm" ]; then
    SAVE_PATH=${SAVE_PATH}_${DEPLOYMENT}
fi

MODEL_CONFIG=${SAVE_PATH}/config.json
ENGINE_DIR=${SAVE_PATH}/${MODEL_TYPE}_${TP}x${PP}x${GPU_NAME}_input${BUILD_MAX_INPUT_LEN}_output${BUILD_MAX_OUTPUT_LEN}_batch${BUILD_MAX_BATCH_SIZE}_engine

if [ "${REMOVE_EXISTING_MODEL_CONFIG,,}" = "true" ]; then
    rm -f $MODEL_CONFIG
fi

PTQ_ARGS=""
if [ $QFORMAT == "fp8_naive" ]; then
    QFORMAT=fp8
    PTQ_ARGS+=" --naive_quantization "
fi

if [ -n "$AUTO_QUANTIZE_COMPRESSION" ]; then
    PTQ_ARGS+=" --auto_quantize_compression $AUTO_QUANTIZE_COMPRESSION"
fi

case "${MODEL_TYPE}" in
    "vila" | "phi")
        MULTIMODAL_LEN=$((BUILD_MAX_BATCH_SIZE * 4096))
        ;;
    "llava")
        MULTIMODAL_LEN=$((BUILD_MAX_BATCH_SIZE * 576))
        ;;
esac

if [ "${MODEL_TYPE}" = "vila" ]; then
    # Install required dependency for VILA
    echo "VILA model needs transformers version 4.36.2"
    pip install -r ../vlm_ptq/requirements-vila.txt
    # Clone oringinal VILA repo
    if [ ! -d "VILA" ]; then
        echo "VILA repository is needed until it is added to HF model zoo. Cloning the repository..."
        git clone https://github.com/Efficient-Large-Model/VILA.git && cd VILA && git checkout b2c70791a4239c813d19f5fa949c3e580556f4df && cd ..
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
            --deployment=$DEPLOYMENT \
            --vlm \
            $PTQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi

    if [ $DEPLOYMENT != "tensorrt_llm" ]; then
        echo "Please continue deployment with $DEPLOYMENT. Checkpoint export_path: $SAVE_PATH"
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
        --max_multimodal_len=$MULTIMODAL_LEN
fi

echo "Build visual engine"

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

python vlm_visual_engine.py \
    --model_path $MODEL_PATH \
    --model_type $VISUAL_MODEL_TYPE \
    --output_dir $VISION_ENCODER_DIR \
    $VISUAL_ARGS

echo "Run inference example"

python vlm_run.py  \
    --hf_model_dir $MODEL_PATH \
    --visual_engine_dir $VISION_ENCODER_DIR \
    --llm_engine_dir $ENGINE_DIR \

if [ "${MODEL_TYPE}" = "vila" ]; then
    echo "For VILA model, current transformers version is 4.36.2, higher version transformers may be needed for other model."
fi

popd
