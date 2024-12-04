#!/bin/bash
set -ex

FLOPS_PERCENT=50
BASE_DIR=results/bert_large_pruned_${FLOPS_PERCENT}_percent

# Export to ONNX on a single GPU
python3 bert_prune_distill_quantize.py \
    --model_name_or_path ${BASE_DIR}/int8_qat/ \
    --onnx_export_path ${BASE_DIR}/pruned_model_int8.onnx \
