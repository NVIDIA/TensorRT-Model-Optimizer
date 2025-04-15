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
set -o pipefail

while [ $# -gt 0 ]; do
  case "$1" in
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --data*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA="${1#*=}"
      ;;
    --mode*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODE="${1#*=}"
      ;;
    --output_dir*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_DIR="${1#*=}"
      ;;
    --num_epochs*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_EPOCHS="${1#*=}"
      ;;
    --save_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      SAVE_STEPS="${1#*=}"
      ;;
    --lr*)
      if [[ "$1" != *=* ]]; then shift; fi
      LR="${1#*=}"
      ;;
    --train_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAIN_BS="${1#*=}"
      ;;
    --medusa_num_heads*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_HEADS="${1#*=}"
      ;;
    --medusa_num_layers*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_LAYERS="${1#*=}"
      ;;
    --eagle_num_layers*)
      if [[ "$1" != *=* ]]; then shift; fi
      EAGLE_NUM_LAYERS="${1#*=}"
      ;;
    --fsdp_transformer_layer_cls_to_wrap*)
      if [[ "$1" != *=* ]]; then shift; fi
      FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${1#*=}"
      ;;
    --num_gpu*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_GPU="${1#*=}"
      ;;
    --do_eval*)
      if [[ "$1" != *=* ]]; then shift; fi
      DO_EVAL="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

set -x

# Get the default value for save_steps based on the available number of GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
# Calculate save_steps
DEFAULT_SAVE_STEPS=$((192 / GPU_COUNT))

MODEL=${MODEL:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
MODE=${MODE:-"medusa"}
OUTPUT_DIR=${OUTPUT_DIR:-"tinyllama-medusa"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
LR=${LR:-"1e-4"}
TRAIN_BS=${TRAIN_BS:-4}
MEDUSA_NUM_HEADS=${MEDUSA_NUM_HEADS:-1}
MEDUSA_NUM_LAYERS=${MEDUSA_NUM_LAYERS:-1}
EAGLE_NUM_LAYERS=${EAGLE_NUM_LAYERS:-1}
REDRAFTER_TOKENS=${REDRAFTER_TOKENS:-1}
REDRAFTER_NUM_LAYERS=${REDRAFTER_NUM_LAYERS:-1}
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-"LlamaDecoderLayer"}
NUM_GPU=${NUM_GPU:-1}
DO_EVAL=${DO_EVAL:-"True"}

if [[ "$MODE" == "medusa" ]]; then
  SPECULATIVE_ARGS="--medusa_num_heads $MEDUSA_NUM_HEADS --medusa_num_layers $MEDUSA_NUM_LAYERS"
elif [[ "$MODE" == "eagle" ]]; then
  SPECULATIVE_ARGS="--eagle_num_layers $EAGLE_NUM_LAYERS"
else
  echo "Only medusa and eagle supported for now!"
  exit 1
fi

if [[ "$NUM_GPU" == 1 ]]; then
  FSDP_ARGS=""
  MULTI_GPU=""
else
  FSDP_ARGS="--fsdp 'full_shard auto_wrap' --fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"
  MULTI_GPU="--multi_gpu"
fi

CMD="accelerate launch $MULTI_GPU --mixed_precision bf16 main.py \
    --mode $MODE \
    --model_name_or_path $MODEL \
    --model_max_length 2048 \
    --dataloader_drop_last True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $TRAIN_BS \
    --gradient_accumulation_steps 1 \
    --do_eval $DO_EVAL \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_steps 100 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --tf32 True \
    --data_path $DATA \
    $FSDP_ARGS \
    $SPECULATIVE_ARGS
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"
