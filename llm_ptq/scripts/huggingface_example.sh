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
    gptj|llama|falcon|baichuan|gpt2|mpt|bloom|chatglm|gemma|recurrentgemma|phi|mixtral|gptnext|qwen|dbrx|enc_dec|glm|internlm|exaone)
        ;;
    *)
        echo "Unsupported type argument: Expected one of: [gpt2, gptj, llama, falcon, baichuan, mpt, bloom, chatglm, gemma, recurrentgemma, phi, mixtral, gptnext, qwen, dbrx, enc_dec, internlm, exaone]" >&2
        exit 1
esac

if [ -z "$MODEL_PATH" ]; then
    echo "Unsupported model argument: Expected a huggingface model path or model name or a nemo path" >&2
    exit 1
fi

#Check if arguments are supported by unified_hf export format
if [ "$EXPORT_FORMAT" = "hf" ]; then
    if [ "$SPARSITY_FMT" != "dense" ]; then
        echo "Unsupported sparsity argument: Expected dense" >&2
        exit 1
    fi

    case $QFORMAT in
    fp8|int4_awq)
        ;;
    *)
        echo "Unsupported quant argument: Expected one of: [fp8, int4_awq]" >&2
        exit 1
    esac
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

if [ -n "$AUTO_QUANTIZE_COMPRESSION" ]; then
    PTQ_ARGS+=" --auto_quantize_compression $AUTO_QUANTIZE_COMPRESSION"
fi

if [[ $TASKS =~ "build" ]] || [[ ! -d "$ENGINE_DIR" ]] || [[ ! $(ls -A $ENGINE_DIR) ]]; then
    if ! [ -f $MODEL_CONFIG ]; then
        echo "Quantizing original model..."
        python hf_ptq.py \
            --pyt_ckpt_path=$MODEL_PATH \
            --export_path=$SAVE_PATH \
            --sparsity_fmt=$SPARSITY_FMT \
            --qformat=$QFORMAT \
            --calib_size=$CALIB_SIZE \
            --batch_size=$CALIB_BATCH_SIZE \
            --inference_tensor_parallel=$TP \
            --inference_pipeline_parallel=$PP \
            --export_fmt=$EXPORT_FORMAT \
            $PTQ_ARGS
    else
        echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
    fi

    if [ $EXPORT_FORMAT != "tensorrt_llm" ]; then
        echo "Please continue to deployment. Checkpoint export_path: $SAVE_PATH"
        exit 0
    fi

    echo "Building tensorrt_llm engine from Model Optimizer-quantized model..."

    python modelopt_to_tensorrt_llm.py \
        --model_config=$MODEL_CONFIG \
        --engine_dir=$ENGINE_DIR \
        --tokenizer=$MODEL_PATH \
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

mpirun -n $GPUS --allow-run-as-root \
    python summarize/summarize.py \
        --engine_dir=$ENGINE_DIR \
        --hf_model_dir=$MODEL_PATH \
        --data_type=fp16 \
        --test_trt_llm \
        --tensorrt_llm_rouge1_threshold=13 \
        --max_ite=$SUMMARIZE_MAX_ITE 2>&1 | tee $SUMMARIZE_RESULT

fi

if [[ -d "${MODEL_PATH}" ]]; then
    MODEL_ABS_PATH=$(realpath ${MODEL_PATH})
else
    # model_path is a HF reference, not a local directory, no need to make the path absolute
    MODEL_ABS_PATH=${MODEL_PATH}
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

python mmlu.py \
    --model_name causal \
    --model_path $MODEL_ABS_PATH \
    --engine_dir $ENGINE_DIR \
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
    --model_path $MODEL_ABS_PATH \
    --engine_dir $ENGINE_DIR \
    --n_sample 1 | tee $HUMANEVAL_RESULT

mv *.jsonl $ENGINE_DIR/

popd

fi

if [[ $TASKS =~ "mtbench" ]]; then

pushd ../llm_eval/

bash run_fastchat.sh -h $MODEL_ABS_PATH -e $ENGINE_DIR
find data/mt_bench/model_answer/ -type f -name '*.jsonl' -exec mv {} $ENGINE_DIR \;

JSONL_PATH=$(readlink -f $(find $ENGINE_DIR -type f -name '*.jsonl'))
echo "FastChat generation complete. The results are saved under $JSONL_PATH . Please run the judge(https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to evaluate the quality of the responses."

popd

fi

if [[ $TASKS =~ "benchmark" ]]; then

if [ "$PP" -ne 1 ]; then
    echo "Benchmark does not work with multi PP. Please run the c++ benchmark in the TensorRT-LLM repo..."
    exit 1
fi

BENCHMARK_RESULT=${ENGINE_DIR}/benchmark.txt
echo "Evaluating performance, result saved to $BENCHMARK_RESULT..."

# Prepare datasets for TRT-LLM benchmark
if [ -z "$TRT_LLM_CODE_PATH" ]; then
    TRT_LLM_CODE_PATH=/workspace/tensorrt-llm
    echo "Setting default TRT_LLM_CODE_PATH to $TRT_LLM_CODE_PATH."
fi

# Synthesize the tokenized benchmarking dataset
TRT_LLM_PREPARE_DATASET=$TRT_LLM_CODE_PATH/benchmarks/cpp/prepare_dataset.py

# Align with the official benchmark
BENCHMARK_INPUT_LEN=2048
BENCHMARK_OUTPUT_LEN=128
BENCHMARK_NUM_REQUESTS=200

DATASET_TXT=${SAVE_PATH}/synthetic_${BENCHMARK_INPUT_LEN}_${BENCHMARK_OUTPUT_LEN}.txt

if [ -z "$TRT_LLM_PREPARE_DATASET" ]; then
    echo "Unable to prepare dataset for benchmarking. Please set TRT_LLM_CODE_PATH to the TRT-LLM code path."
else
    if ! [ -f $DATASET_TXT ]; then
        python $TRT_LLM_PREPARE_DATASET --stdout --tokenizer $MODEL_PATH token-norm-dist \
            --input-mean $BENCHMARK_INPUT_LEN --output-mean $BENCHMARK_OUTPUT_LEN --input-stdev 0 --output-stdev 0 \
            --num-requests $BENCHMARK_NUM_REQUESTS > $DATASET_TXT
    else
        echo "Use exisitng benchmark dataset in $DATASET_TXT."
    fi
fi

python benchmarks/benchmark_suite.py --model $MODEL_PATH throughput --engine_dir $ENGINE_DIR --dataset $DATASET_TXT | tee $BENCHMARK_RESULT

fi

if [ -n "$FREE_SPACE" ]; then
    rm -f $SAVE_PATH/*.json
    rm -f $SAVE_PATH/*.safetensors
    rm -f $ENGINE_DIR/*.json
    rm -f $ENGINE_DIR/*.engine
fi

popd
