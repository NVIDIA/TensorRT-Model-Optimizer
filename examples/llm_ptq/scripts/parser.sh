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

# Define a function to parse command-line options
parse_options() {
  # Default values
    MODEL_TYPE=""
    MODEL_PATH=""
    QFORMAT=""
    KV_CACHE_QUANT=""
    TP=1
    CALIB_TP=
    PP=1
    GPUS=1
    SPARSITY_FMT="dense"
    EXPORT_FORMAT="tensorrt_llm"
    LM_EVAL_TASKS="mmlu,gsm8k"
    LM_EVAL_LIMIT=
    SIMPLE_EVAL_TASKS="mmlu"

    TASKS="build"

    TRUST_REMOTE_CODE=false
    KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.8
    VERBOSE=true

  # Parse command-line options
  ARGS=$(getopt -o "" -l "type:,model:,quant:,kv_cache_quant:,tp:,calib_tp:,pp:,sparsity:,awq_block_size:,calib:,calib_batch_size:,effective_bits:,input:,output:,batch:,tasks:,export_fmt:,lm_eval_tasks:,lm_eval_limit:,simple_eval_tasks:,trust_remote_code,kv_cache_free_gpu_memory_fraction:,no-verbose" -n "$0" -- "$@")
  eval set -- "$ARGS"
  while true; do
    case "$1" in
      --type ) MODEL_TYPE="$2"; shift 2;;
      --model ) MODEL_PATH="$2"; shift 2;;
      --quant ) QFORMAT="$2"; shift 2;;
      --kv_cache_quant ) KV_CACHE_QUANT="$2"; shift 2;;
      --tp ) TP="$2"; shift 2;;
      --calib_tp ) CALIB_TP="$2"; shift 2;;
      --pp ) PP="$2"; shift 2;;
      --sparsity ) SPARSITY_FMT="$2"; shift 2;;
      --awq_block_size ) AWQ_BLOCK_SIZE="$2"; shift 2;;
      --calib ) CALIB_SIZE="$2"; shift 2;;
      --calib_batch_size ) CALIB_BATCH_SIZE="$2"; shift 2;;
      --effective_bits ) AUTO_QUANTIZE_BITS="$2"; shift 2;;
      --input ) BUILD_MAX_INPUT_LEN="$2"; shift 2;;
      --output ) BUILD_MAX_OUTPUT_LEN="$2"; shift 2;;
      --batch ) BUILD_MAX_BATCH_SIZE="$2"; shift 2;;
      --tasks ) TASKS="$2"; shift 2;;
      --export_fmt ) EXPORT_FORMAT="$2"; shift 2;;
      --lm_eval_tasks ) LM_EVAL_TASKS="$2"; shift 2;;
      --lm_eval_limit ) LM_EVAL_LIMIT="$2"; shift 2;;
      --simple_eval_tasks ) SIMPLE_EVAL_TASKS="$2"; shift 2;;
      --num_samples ) NUM_SAMPLES="$2"; shift 2;;
      --trust_remote_code ) TRUST_REMOTE_CODE=true; shift;;
      --kv_cache_free_gpu_memory_fraction ) KV_CACHE_FREE_GPU_MEMORY_FRACTION="$2"; shift 2;;
      --no-verbose ) VERBOSE=false; shift;;
      -- ) shift; break ;;
      * ) break ;;
    esac
  done

  DEFAULT_CALIB_SIZE=512
  DEFAULT_CALIB_BATCH_SIZE=0
  DEFAULT_BUILD_MAX_INPUT_LEN=3072
  DEFAULT_BUILD_MAX_OUTPUT_LEN=1024
  DEFAULT_BUILD_MAX_BATCH_SIZE=2

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

  # Verify required options are provided
  if [ -z "$MODEL_PATH" ] || [ -z "$QFORMAT" ] || [ -z "$TASKS" ]; then
    echo "Usage: $0 --model=<MODEL_PATH> --quant=<QFORMAT> --tasks=<TASK,...>"
    echo "Optional args: --tp=<TP> --pp=<PP> --sparsity=<SPARSITY_FMT> --awq_block_size=<AWQ_BLOCK_SIZE> --calib=<CALIB_SIZE>"
    echo "Optional args for NeMo: --type=<MODEL_TYPE> --calib_tp=<CALIB_TP>"
    exit 1
  fi

  VALID_TASKS=("build" "mmlu" "mtbench" "benchmark" "lm_eval" "gqa" "livecodebench" "simple_eval")

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
  echo "effective_bits: $AUTO_QUANTIZE_BITS"
  echo "input: $BUILD_MAX_INPUT_LEN"
  echo "output: $BUILD_MAX_OUTPUT_LEN"
  echo "batch: $BUILD_MAX_BATCH_SIZE"
  echo "tasks: $TASKS"
  echo "export_fmt: $EXPORT_FORMAT"
  echo "lm_eval_tasks: $LM_EVAL_TASKS"
  echo "lm_eval_limit: $LM_EVAL_LIMIT"
  echo "simple_eval_tasks: $SIMPLE_EVAL_TASKS"
  echo "num_sample: $NUM_SAMPLES"
  echo "kv_cache_free_gpu_memory_fraction: $KV_CACHE_FREE_GPU_MEMORY_FRACTION"
  echo "================="
}
