#!/bin/bash
set -e
set -x

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent

# Export to ONNX on a single GPU
python3 bert_prune_distill_quantize.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --output_dir ${BASE_DIR}/onnx \
    --modelopt_restore_path ${BASE_DIR}/int8_quantized/quantized_model.pth \
    --onnx_export_file pruned_model_int8.onnx \
