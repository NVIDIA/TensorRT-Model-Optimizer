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
set -x

# Default format
format="int8"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --format)
            format="$2"
            shift
            shift
            ;;
        *)  # Unknown option
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Check if the format is valid
if [[ "$format" != "int8" && "$format" != "fp8" ]]; then
    echo "Invalid format. Please choose either 'int8' or 'fp8'."
    exit 1
fi

# Move to the diffusers directory
echo "current folder at $PWD"

# Configurations for quantization
model="sdxl-1.0"
quant_level=3


cleaned_m="${model//\//-}"
curt_exp="${cleaned_m}_${quant_level}_${format}"
echo "=====>Processing $curt_exp"
if [ "$format" == "fp8" ]; then
    python quantize.py --model "$model" --format "$format" --batch-size 2 --calib-size 128 --quant-level 3.0 --n-steps 20 --quantized-torch-ckpt-save-path "$curt_exp".pt --collect-method default --onnx-dir "$curt_exp".onnx
else
    python quantize.py --model "$model" --format "$format" --batch-size 2 --calib-size 32 --collect-method "min-mean" --percentile 1.0 --alpha 0.8 --quant-level "$quant_level" --n-steps 20 --quantized-torch-ckpt-save-path "$curt_exp".pt --onnx-dir "$curt_exp".onnx
fi

echo "=====>Exported to ONNX model."
