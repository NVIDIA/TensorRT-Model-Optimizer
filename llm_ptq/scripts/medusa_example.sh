#!/bin/bash
set -e

script_dir="$(dirname "$(readlink -f "$0")")"

source $script_dir/parser.sh
parse_options "$@"

set -x

# This will prevent the script from hanging on Selene/EOS due to the MPI support.
echo "********** unset all SLURM_, PMI_, PMIX_ Variables **********"
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

case $MODEL_TYPE in
    llama|mixtral)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [llama, mixtral]" >&2
        exit 1
esac

if [ -z "$MODEL_PATH" ]; then
    echo "Unsupported model argument: Expected a medusa model path" >&2
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
    fp8|fp16)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [fp8, fp16]" >&2
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

echo "Using the following config: max input $BUILD_MAX_INPUT_LEN max output $BUILD_MAX_OUTPUT_LEN max batch $BUILD_MAX_BATCH_SIZE"

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

if [ -z "$ROOT_SAVE_PATH" ]; then
    ROOT_SAVE_PATH=$(pwd)
fi

MODEL_NAME=$(basename $MODEL_PATH | sed 's/\.[^.]*$//')
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}_${SPARSITY_FMT}_${QFORMAT}_tp${TP}_pp${PP}_medusa
MODEL_CONFIG=${SAVE_PATH}/config.json
ENGINE_DIR=${SAVE_PATH}/${MODEL_TYPE}_${TP}x${PP}x${GPU_NAME}_input${BUILD_MAX_INPUT_LEN}_output${BUILD_MAX_OUTPUT_LEN}_batch${BUILD_MAX_BATCH_SIZE}_engine
CONFIG_PATH=$MODEL_PATH/config.json
TOKENIZER_PATH=$(jq -r '.base_model_name_or_path' $CONFIG_PATH)

if [ "${REMOVE_EXISTING_MODEL_CONFIG,,}" = "true" ]; then
    rm -f $MODEL_CONFIG
fi

PTQ_ARGS=""
if [ $QFORMAT == "fp8_naive" ]; then
    QFORMAT=fp8
    PTQ_ARGS+=" --naive_quantization "
fi

if [[ $TASKS =~ "build" ]] || [[ ! -d "$ENGINE_DIR" ]] || [[ ! $(ls -A $ENGINE_DIR) ]]; then
    if ! [ -f $MODEL_CONFIG ]; then
        echo "Quantizing original model..."
        python3 hf_ptq.py \
            --pyt_ckpt_path=$MODEL_PATH \
            --export_path=$SAVE_PATH \
            --sparsity_fmt=$SPARSITY_FMT \
            --qformat=$QFORMAT \
            --calib_size=$CALIB_SIZE \
            --inference_tensor_parallel=$TP \
            --inference_pipeline_parallel=$PP \
            --batch_size=1 \
            --medusa \
            $PTQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi

    if [ $MODEL_TYPE == "llava" ]; then
        echo "Please build tensorrt_llm engine with this model from the tensorrt_llm repo."
        exit 0
    fi

    echo "Building tensorrt_llm engine from Model Optimizer-quantized model..."

    python modelopt_to_tensorrt_llm.py \
        --model_config=$MODEL_CONFIG \
        --engine_dir=$ENGINE_DIR \
        --tokenizer=$TOKENIZER_PATH \
        --max_input_len=$BUILD_MAX_INPUT_LEN \
        --max_output_len=$BUILD_MAX_OUTPUT_LEN \
        --max_batch_size=$BUILD_MAX_BATCH_SIZE \
        --num_build_workers=$GPUS \
        --enable_sparsity=$ENABLE_SPARSITY
fi

if [[ $TASKS =~ "summarize" ]]; then

SUMMARIZE_RESULT=${ENGINE_DIR}/summarize.txt

if [ -z "$SUMMARIZE_MAX_ITE" ]; then
    SUMMARIZE_MAX_ITE=20
fi

echo "Evaluating the built TRT engine (ite $SUMMARIZE_MAX_ITE), result saved to $SUMMARIZE_RESULT..."

# specify medusa_choices for the tree-based attention (https://arxiv.org/abs/2401.10774)
mpirun -n $GPUS --allow-run-as-root \
    python3 summarize/summarize.py \
        --engine_dir=$ENGINE_DIR \
        --hf_model_dir=$TOKENIZER_PATH \
        --data_type=fp16 \
        --test_trt_llm \
        --tensorrt_llm_rouge1_threshold=13 \
        --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1],
        [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3],
        [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9],
        [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0],
        [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
        --use_py_session \
        --max_ite=$SUMMARIZE_MAX_ITE 2>&1 | tee $SUMMARIZE_RESULT

fi


if [ -n "$FREE_SPACE" ]; then
    rm -f $SAVE_PATH/*.json
    rm -f $SAVE_PATH/*.safetensors
    rm -f $ENGINE_DIR/*.json
    rm -f $ENGINE_DIR/*.engine
fi

popd
