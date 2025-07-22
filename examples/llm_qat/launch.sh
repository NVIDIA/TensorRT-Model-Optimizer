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

set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [ $# -gt 0 ]; do
  case "$1" in
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --output_dir*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_DIR="${1#*=}"
      ;;
    --dataset*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATASET="${1#*=}"
      ;;
    --train_size*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAIN_SIZE="${1#*=}"
      ;;
    --eval_size*)
      if [[ "$1" != *=* ]]; then shift; fi
      EVAL_SIZE="${1#*=}"
      ;;
    --num_epochs*)
      if [[ "$1" != *=* ]]; then shift; fi
      NUM_EPOCHS="${1#*=}"
      ;;
    --max_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      MAX_STEPS="${1#*=}"
      ;;
    --save_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      SAVE_STEPS="${1#*=}"
      ;;
    --accum_steps*)
      if [[ "$1" != *=* ]]; then shift; fi
      ACCUM_STEPS="${1#*=}"
      ;;
    --lr*)
      if [[ "$1" != *=* ]]; then shift; fi
      LR="${1#*=}"
      ;;
    --quant_cfg*)
      if [[ "$1" != *=* ]]; then shift; fi
      QUANT_CFG="${1#*=}"
      ;;
    --compress*)
      if [[ "$1" != *=* ]]; then shift; fi
      COMPRESS="${1#*=}"
      ;;
    --calib_size*)
      if [[ "$1" != *=* ]]; then shift; fi
      CALIB_SIZE="${1#*=}"
      ;;
    --train_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      TRAIN_BS="${1#*=}"
      ;;
    --eval_bs*)
      if [[ "$1" != *=* ]]; then shift; fi
      EVAL_BS="${1#*=}"
      ;;
    --do_train*)
      if [[ "$1" != *=* ]]; then shift; fi
      DO_TRAIN="${1#*=}"
      ;;
    --lora*)
      if [[ "$1" != *=* ]]; then shift; fi
      LORA="${1#*=}"
      ;;
    --teacher_model*)
      if [[ "$1" != *=* ]]; then shift; fi
      TEACHER_MODEL="${1#*=}"
      ;;
    --distill*)
      if [[ "$1" != *=* ]]; then shift; fi
      DISTILL="${1#*=}"
      ;;
    --fsdp_transformer_layer_cls_to_wrap*)
      if [[ "$1" != *=* ]]; then shift; fi
      FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${1#*=}"
      ;;
    --use_fsdp2*)
      if [[ "$1" != *=* ]]; then shift; fi
      USE_FSDP2="${1#*=}"
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

MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
OUTPUT_DIR=${OUTPUT_DIR:-"llama2-finetune"}
DATASET=${DATASET:-"Daring-Anteater"}
TRAIN_SIZE=${TRAIN_SIZE:-0}
EVAL_SIZE=${EVAL_SIZE:-0}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
ACCUM_STEPS=${ACCUM_STEPS:-1}
LR=${LR:-"1e-4"}
CALIB_SIZE=${CALIB_SIZE:-512}
TRAIN_BS=${TRAIN_BS:-4}
EVAL_BS=${EVAL_BS:-4}
DO_TRAIN=${DO_TRAIN:-True}
LORA=${LORA:-"False"}
COMPRESS=${COMPRESS:-"False"}
DISTILL=${DISTILL:-"False"}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL}
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-"LlamaDecoderLayer"}

if [ -z $QUANT_CFG ]; then
  QUANT_ARGS=""
else
  QUANT_ARGS="--quant_cfg $QUANT_CFG --calib_size $CALIB_SIZE"
fi

OPTIONAL_ARGS=""
if [ ! -z $MAX_STEPS ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --max_steps $MAX_STEPS"
fi

CONFIG_FILE="fsdp1.yaml"
FSDP_ARGS="--fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"
GRADIENT_CHECKPOINTING_ARGS="--gradient_checkpointing True"

if [[ "${USE_FSDP2,,}" == "true" ]]; then
  echo "Using FSDP2 instead of FSDP1. FSDP2 is not mature yet! Please use it with latest torch and transformers."
  CONFIG_FILE="fsdp2.yaml"
  GRADIENT_CHECKPOINTING_ARGS=""
fi

DISTILLATION_ARGS=""
if [[ "${DISTILL}" == "True" ]]; then
  DISTILLATION_ARGS="--distill $DISTILL --teacher_model $TEACHER_MODEL"
  # Distillation does not work with memory efficient loading
  FSDP_ARGS="$FSDP_ARGS --fsdp_cpu_ram_efficient_loading False"
fi

# real quantization does not work with FSDP
if [[ "${COMPRESS,,}" == "true" ]]; then
  echo "Compression is not supported with FSDP. Disabling FSDP."
  FSDP_ARGS=""
  CONFIG_FILE="ddp.yaml"
fi

CMD="accelerate launch --config-file accelerate_config/$CONFIG_FILE $FSDP_ARGS \
    main.py \
    --model_name_or_path $MODEL \
    --model_max_length 4096 \
    --dataloader_drop_last True \
    --do_train $DO_TRAIN \
    --do_eval True \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --train_size $TRAIN_SIZE \
    --eval_size $EVAL_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --eval_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $SAVE_STEPS \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --report_to tensorboard \
    --lora $LORA \
    --compress $COMPRESS \
    $QUANT_ARGS $OPTIONAL_ARGS $GRADIENT_CHECKPOINTING_ARGS $DISTILLATION_ARGS
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"
