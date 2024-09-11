#!/bin/bash
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
    gptnext|llama|gpt2|gemma)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [gptnext, gpt2, llama, gemma]" >&2
        exit 1
esac

if [[ -z "$MODEL_PATH" ]] || [[ ! -f "$MODEL_PATH" ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "Unsupported model argument: Expect a nemo file or a nemo extracted directory" >&2
    echo "E.g. you can download the 2B model with this URL: https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo"
    exit 1
fi

case $QFORMAT in
    bf16|fp16|fp8|int8_sq|int4_awq|w4a8_awq)
        ;;
    *)
        echo "Unknown quant argument: Expected one of: [bf16, fp16, fp8, int8_sq, int4_awq, w4a8_awq]" >&2
        exit 1
esac

if [ -z "$DTYPE" ]; then
    DTYPE="bf16"
fi

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

MODEL_NAME=$(basename $MODEL_PATH | sed 's/\.[^.]*$//')
SAVE_PATH=${ROOT_SAVE_PATH}/saved_models_${MODEL_NAME}_${QFORMAT}_${DTYPE}_tp${TP}_pp${PP}
MODEL_CONFIG_PTQ=${SAVE_PATH}/config.json
ENGINE_DIR=${SAVE_PATH}/${MODEL_TYPE}_${TP}x${PP}x${GPU_NAME}_input${BUILD_MAX_INPUT_LEN}_output${BUILD_MAX_OUTPUT_LEN}_batch${BUILD_MAX_BATCH_SIZE}_engine
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
            quantization.algorithm=$QFORMAT \
            quantization.awq_block_size=$(($AWQ_BLOCK_SIZE)) \
            quantization.num_calib_size=$(($CALIB_SIZE)) \
            inference.batch_size=$(($CALIB_BATCH_SIZE)) \
            export.path=$SAVE_PATH \
            export.decoder_type=$MODEL_TYPE \
            export.inference_tensor_parallel=$(($TP)) \
            export.inference_pipeline_parallel=$(($PP)) \
            export.dtype=$DTYPE
    else
        echo "Quantized model config $MODEL_CONFIG_PTQ exists, skipping the quantization stage"
    fi

    if [ "$PP" -ne 1 ] && [[ "$MODEL_TYPE" =~ "gpt" ]]; then
        echo "PP on GPT/GPTNext has not be enabled for the TensorRT-LLM 0.9 release."
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

if [[ $TASKS =~ "summarize" ]]; then

    SUMMARIZE_RESULT=${ENGINE_DIR}/summarize.txt

    if [ -z "$SUMMARIZE_MAX_ITE" ]; then
        SUMMARIZE_MAX_ITE=20
    fi

    echo "Evaluating the built TRT engine (ite $SUMMARIZE_MAX_ITE), result saved to $SUMMARIZE_RESULT..."

    # In order to handle tokenizer for Nemo models you have two options:
    # (1.) specify --vocab_file that is yaml tokenizer config (i.e. given by 'tokenizer' section in model_config.yaml)
    # (2.) specify --hf_model_dir for models converted from HuggingFace (works for e.g. Llama2)
    # We choose more explicit Option (1.) here:
    mpirun -n $GPUS --allow-run-as-root \
        python summarize/summarize.py \
            --engine_dir $ENGINE_DIR \
            --vocab_file $TOKENIZER_CONFIG \
            --test_trt_llm \
            --tensorrt_llm_rouge1_threshold=13 \
            --max_ite=$SUMMARIZE_MAX_ITE \
            --data_type=$DTYPE | tee $SUMMARIZE_RESULT

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
    # or --model_path in case of HF-converted models, see the comment for summarize.py script above.
    python mmlu.py \
        --engine_dir $ENGINE_DIR \
        --vocab_file $TOKENIZER_CONFIG \
        --data_dir $MMLU_DATA_PATH | tee $MMLU_RESULT
    popd

fi

if [[ $TASKS =~ "humaneval" ]]; then

    HUMANEVAL_RESULT=${ENGINE_DIR}/humaneval.txt
    echo "Evaluating humaneval, result saved to $HUMANEVAL_RESULT..."

    pushd ../llm_eval/

    pip install -r requirements.txt

    if [ -z "$HUMANEVAL_CODE_PATH" ]; then
        HUMANEVAL_CODE_PATH=human_eval
    fi
    if [[ ! -d "$HUMANEVAL_CODE_PATH" ]] || [[ ! $(ls -A $HUMANEVAL_CODE_PATH) ]]; then
        echo "Preparing the human eval tests"
        wget https://github.com/declare-lab/instruct-eval/archive/refs/heads/main.zip -O /tmp/instruct-eval.zip
        unzip /tmp/instruct-eval.zip "instruct-eval-main/human_eval/*" -d /tmp/
        cp -r /tmp/instruct-eval-main/human_eval .
    fi

    python humaneval.py \
        --model_name causal \
        --engine_dir $ENGINE_DIR \
        --vocab_file $TOKENIZER_CONFIG \
        --n_sample 1 | tee $HUMANEVAL_RESULT

    mv *.jsonl $ENGINE_DIR/

    popd

fi

popd
