#!/bin/bash
set -e
set -x

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent

if [ ! -f "${BASE_DIR}/pruned_model.pth" ]; then
    modelopt_args="--do_modelopt_prune --modelopt_prune_flops_percent ${FLOPS_PERCENT}"
else
    modelopt_args="--modelopt_restore_path ${BASE_DIR}/pruned_model.pth"
fi
modelopt_args="${modelopt_args} --modelopt_save_file pruned_model.pth"

# Run pruning followed by distributed fine-tuning on the pruned model
accelerate launch --multi_gpu --mixed_precision bf16 bert_prune_distill_quantize.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --output_dir ${BASE_DIR} \
    ${modelopt_args} \
    --do_train \
    --do_modelopt_distill \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 15 \
    --with_tracking \
    --checkpointing_steps epoch \
    --resume_from_last_ckpt
