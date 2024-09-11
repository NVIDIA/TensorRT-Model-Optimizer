#!/bin/bash

# Define a function to parse command-line options
parse_options() {
  # Default values
    MODEL_TYPE=""
    MODEL_PATH=""
    QFORMAT=""
    TP=1
    CALIB_TP=
    PP=1
    GPUS=1
    SPARSITY_FMT="dense"
    EXPORT_FORMAT="tensorrt_llm"

    TASKS="build,summarize,mmlu,humaneval,mtbench,benchmark"

  # Parse command-line options
  ARGS=$(getopt -o "" -l "type:,model:,quant:,tp:,calib_tp:,pp:,sparsity:,awq_block_size:,calib:,calib_batch_size:,compression:,input:,output:,batch:,summarize_ite:,tasks:,export_fmt:," -n "$0" -- "$@")
  eval set -- "$ARGS"
  while true; do
    case "$1" in
      --type ) MODEL_TYPE="$2"; shift 2;;
      --model ) MODEL_PATH="$2"; shift 2;;
      --quant ) QFORMAT="$2"; shift 2;;
      --tp ) TP="$2"; shift 2;;
      --calib_tp ) CALIB_TP="$2"; shift 2;;
      --pp ) PP="$2"; shift 2;;
      --sparsity ) SPARSITY_FMT="$2"; shift 2;;
      --awq_block_size ) AWQ_BLOCK_SIZE="$2"; shift 2;;
      --calib ) CALIB_SIZE="$2"; shift 2;;
      --calib_batch_size ) CALIB_BATCH_SIZE="$2"; shift 2;;
      --compression ) AUTO_QUANTIZE_COMPRESSION="$2"; shift 2;;
      --input ) BUILD_MAX_INPUT_LEN="$2"; shift 2;;
      --output ) BUILD_MAX_OUTPUT_LEN="$2"; shift 2;;
      --batch ) BUILD_MAX_BATCH_SIZE="$2"; shift 2;;
      --summarize_ite ) SUMMARIZE_MAX_ITE="$2"; shift 2;;
      --tasks ) TASKS="$2"; shift 2;;
      --export_fmt ) EXPORT_FORMAT="$2"; shift 2;;
      -- ) shift; break ;;
      * ) break ;;
    esac
  done

  DEFAULT_CALIB_SIZE=512
  DEFAULT_CALIB_BATCH_SIZE=0
  DEFAULT_BUILD_MAX_INPUT_LEN=2048
  DEFAULT_BUILD_MAX_OUTPUT_LEN=512
  DEFAULT_BUILD_MAX_BATCH_SIZE=4
  DEFAULT_SUMMARIZE_MAX_ITE=20

  DEFAULT_AWQ_BLOCK_SIZE=128
  if [ -z "$AWQ_BLOCK_SIZE" ]; then
    AWQ_BLOCK_SIZE=$DEFAULT_AWQ_BLOCK_SIZE
  fi
  if [ -z "$CALIB_SIZE" ]; then
    CALIB_SIZE=$DEFAULT_CALIB_SIZE
  fi
  if [ -z "$CALIB_BATCH_SIZE" ]; then
    CALIB_BATCH_SIZE=$DEFAULT_CALIB_BATCH_SIZE
  fi
  if [ -z "$BUILD_MAX_INPUT_LEN" ]; then
    BUILD_MAX_INPUT_LEN=$DEFAULT_BUILD_MAX_INPUT_LEN
  fi
  if [ -z "$BUILD_MAX_OUTPUT_LEN" ]; then
    BUILD_MAX_OUTPUT_LEN=$DEFAULT_BUILD_MAX_OUTPUT_LEN
  fi
  if [ -z "$BUILD_MAX_BATCH_SIZE" ]; then
    BUILD_MAX_BATCH_SIZE=$DEFAULT_BUILD_MAX_BATCH_SIZE
  fi


  echo "$TASKS"

  # Verify required options are provided
  if [ -z "$MODEL_TYPE" ] || [ -z "$MODEL_PATH" ] || [ -z "$QFORMAT" ] || [ -z "$TASKS" ]; then
    echo "Usage: $0 --type=<MODEL_TYPE> --model=<MODEL_PATH> --quant=<QFORMAT> --tasks=<TASK,...>"
    echo "Optional args: --tp=<TP> --pp=<PP> --sparsity=<SPARSITY_FMT> --awq_block_size=<AWQ_BLOCK_SIZE> --calib=<CALIB_SIZE>"
    echo "Optional args for NeMo: --calib_tp=<CALIB_TP>"
    exit 1
  fi


  VALID_TASKS=("build" "summarize" "mmlu" "humaneval" "mtbench" "benchmark")

  for task in $(echo $TASKS | tr ',' ' '); do
    if [[ ! " ${VALID_TASKS[@]} " =~ " $task " ]]; then
      echo "task $task is not valid"
      VALID_TASKS_STRING=$(IFS=','; echo "${VALID_TASKS[*]}")
      echo "Allowed tasks are: $VALID_TASKS_STRING"
      exit 1
    fi
  done

  GPUS=$(($TP*$PP))

  # Make sparsity and int4 quantization mutually exclusive as it does not brings speedup
  if [[ "$SPARSITY_FMT" = "sparsegpt" || "$SPARSITY_FMT" = "sparse_magnitude" ]]; then
    if [[ "$QFORMAT" == *"awq"* ]]; then
        echo "Sparsity is not compatible with 'awq' quantization for TRT-LLM deployment."
        exit 1  # Exit script with an error
    fi
  fi

  # Now you can use the variables $GPU, $MODEL, and $TASKS in your script
  echo "================="
  echo "type: $MODEL_TYPE"
  echo "model: $MODEL_PATH"
  echo "quant: $QFORMAT"
  echo "tp: $TP"
  echo "calib_tp: $CALIB_TP"
  echo "pp: $PP"
  echo "gpus: $GPUS"
  echo "sparsity: $SPARSITY_FMT"
  echo "awq_block_size: $AWQ_BLOCK_SIZE"
  echo "calib: $CALIB_SIZE"
  echo "calib_batch_size: $CALIB_BATCH_SIZE"
  echo "compression: $AUTO_QUANTIZE_COMPRESSION"
  echo "input: $BUILD_MAX_INPUT_LEN"
  echo "output: $BUILD_MAX_OUTPUT_LEN"
  echo "batch: $BUILD_MAX_BATCH_SIZE"
  echo "summarize_ite: $SUMMARIZE_MAX_ITE"
  echo "tasks: $TASKS"
  echo "export_fmt: $EXPORT_FORMAT"
  echo "================="
}
