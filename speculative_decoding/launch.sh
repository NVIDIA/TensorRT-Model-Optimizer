#!/bin/bash
set -e
set -x
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
    --heads*)
      if [[ "$1" != *=* ]]; then shift; fi
      HEADS="${1#*=}"
      ;;
    --layers*)
      if [[ "$1" != *=* ]]; then shift; fi
      LAYERS="${1#*=}"
      ;;
    --only_medusa_heads*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_ONLY_HEADS="${1#*=}"
      ;;
    --num_medusa_heads*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_HEADS="${1#*=}"
      ;;
    --num_medusa_layers*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_NUM_LAYERS="${1#*=}"
      ;;
    --lm_head_medusa*)
      if [[ "$1" != *=* ]]; then shift; fi
      MEDUSA_LM_HEAD="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument\n"
      exit 1
      ;;
  esac
  shift
done

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
HEADS=${HEADS:-2}
LAYERS=${LAYERS:-1}
MEDUSA_ONLY_HEADS=${MEDUSA_ONLY_HEADS:-True}
MEDUSA_NUM_HEADS=${MEDUSA_NUM_HEADS:-1}
MEDUSA_NUM_LAYERS=${MEDUSA_NUM_LAYERS:-1}

MEDUSA_ARGS="--medusa_only_heads $MEDUSA_ONLY_HEADS --medusa_num_heads $MEDUSA_NUM_HEADS \
             --medusa_num_layers $MEDUSA_NUM_LAYERS"
if [ -z $MEDUSA_LM_HEAD ]; then
    MEDUSA_ARGS=$MEDUSA_ARGS
else
    MEDUSA_ARGS="$MEDUSA_ARGS --medusa_lm_head $MEDUSA_LM_HEAD"
fi

if [[ "$MODE" == "medusa" ]]; then
  CMD="accelerate launch --multi_gpu --mixed_precision bf16 main.py \
      --model_name_or_path $MODEL \
      --model_max_length 2048 \
      --dataloader_drop_last True \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --num_train_epochs $NUM_EPOCHS \
      --per_device_train_batch_size $TRAIN_BS \
      --gradient_accumulation_steps 1 \
      --eval_accumulation_steps 1 \
      --gradient_checkpointing True \
      --save_strategy steps \
      --save_steps $SAVE_STEPS \
      --learning_rate $LR \
      --weight_decay 0.0 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type linear \
      --logging_steps 1 \
      --fsdp 'full_shard auto_wrap' \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --data_path $DATA \
      $MEDUSA_ARGS
  "

  start_time=$(date +%s)
  sh -c "$CMD"
  echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"
else
  echo "Only medusa supported for now!"
fi
