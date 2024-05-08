#!/bin/bash

set -x

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
    --quant_cfg*)
      if [[ "$1" != *=* ]]; then shift; fi
      QUANT_CFG="${1#*=}"
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

MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
OUTPUT_DIR=${OUTPUT_DIR:-"llama2-finetune"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}
LR=${LR:-"1e-4"}
CALIB_SIZE=${CALIB_SIZE:-512}
TRAIN_BS=${TRAIN_BS:-4}
EVAL_BS=${EVAL_BS:-1}
DO_TRAIN=${DO_TRAIN:-True}

if [ -z $QUANT_CFG ]; then
    QUANT_ARGS=""
else
    QUANT_ARGS="--quant_cfg $QUANT_CFG --calib_size $CALIB_SIZE"
fi

CMD="accelerate launch --multi_gpu --mixed_precision bf16 main.py \
    --model_name_or_path $MODEL \
    --model_max_length 4096 \
    --dataloader_drop_last True \
    --bf16 True \
    --do_train $DO_TRAIN \
    --do_eval True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --evaluation_strategy steps \
    --eval_steps $SAVE_STEPS \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --report_to tensorboard \
    --fsdp 'full_shard auto_wrap' \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    $QUANT_ARGS
"

start_time=$(date +%s)
sh -c "$CMD"
echo "Total time taken: $(( $(date +%s) - $start_time )) seconds"
