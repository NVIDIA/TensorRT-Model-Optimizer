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
    fp8|fp8_naive|int8_sq|int4_awq|w4a8_awq|fp16|bf16)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, fp8_naive, int8_sq, int4_awq, w4a8_awq, fp16, bf16]" >&2
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

BUILD_MAX_BATCH_SIZE=4

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
if [ $QFORMAT == "fp8_naive" ]; then
    QFORMAT=fp8
    PTQ_ARGS+=" --naive_quantization "
fi

if [ -n "$AUTO_QUANTIZE_BITS" ]; then
    PTQ_ARGS+=" --auto_quantize_bits $AUTO_QUANTIZE_BITS"
fi

case "${MODEL_TYPE}" in
    "vila" | "phi")
        VLM_ARGS=" --max_multimodal_len=$((BUILD_MAX_BATCH_SIZE * 4096)) "
        ;;
    "llava")
        VLM_ARGS=" --max_multimodal_len=$((BUILD_MAX_BATCH_SIZE * 576)) "
        ;;
    "mllama")
        VLM_ARGS=" --max_encoder_input_len=4100 "
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
elif [ "${MODEL_TYPE}" = "llava" ]; then
    echo "LLAVA model needs transformers version 4.42.4."
    pip install -r ../vlm_ptq/requirements-llava.txt
elif [ "${MODEL_TYPE}" = "mllama" ]; then
    echo "Mllama3.2 model requires transformers version 4.46.2 or try the latest version."
    pip install -r ../vlm_ptq/requirements-mllama.txt
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

    if [ $MODEL_TYPE == "mllama" ]; then
        echo "Please continue deployment with the latest TensorRT-LLM main branch. Checkpoint export_path: $SAVE_PATH"
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

VLM_RUN_ARGS=""

if [ "$MODEL_TYPE" == "vila" ]; then
    VLM_RUN_ARGS+=" --image_path https://github.com/NVlabs/VILA/blob/6b941da19e31ddfdfaa60160908ccf0978d96615/demo_images/av.png?raw=true"
fi

python vlm_run.py  \
    --hf_model_dir $MODEL_PATH \
    --visual_engine_dir $VISION_ENCODER_DIR \
    --llm_engine_dir $ENGINE_DIR \
    $VLM_RUN_ARGS

if [ "${MODEL_TYPE}" = "llava" ]; then
    echo "For Llava model, current transformers version is 4.42.4, higher version transformers may be needed for other model."
fi

popd
