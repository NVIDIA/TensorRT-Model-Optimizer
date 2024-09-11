#!/bin/bash
set -e
set -x

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent
OUTPUT_DIR=${BASE_DIR}/int8_quantized

if [ ! -f "${OUTPUT_DIR}/quantized_model.pth" ]; then
    modelopt_args="--modelopt_quantize_cfg INT8_DEFAULT_CFG --modelopt_restore_path ${BASE_DIR}/pruned_model.pth"
else
    modelopt_args="--modelopt_restore_path ${OUTPUT_DIR}/quantized_model.pth"
fi

modelopt_args="${modelopt_args} --modelopt_save_file quantized_model.pth"

# Run distributed QAT on the pruned model with 0.1x LR and less epochs
accelerate launch --multi_gpu --mixed_precision bf16 bert_prune_distill_quantize.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --output_dir ${OUTPUT_DIR} \
    ${modelopt_args} \
    --do_train \
    --do_modelopt_distill \
    --lr_scheduler_type cosine \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 2 \
    --with_tracking \
    --checkpointing_steps epoch \
    --resume_from_last_ckpt
