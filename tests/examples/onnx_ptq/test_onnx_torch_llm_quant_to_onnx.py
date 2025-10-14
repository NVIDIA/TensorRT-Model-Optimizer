# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
ONNX Export Supported LLM Models

| Model                   | FP16 | INT4 | FP8  | NVFP4 |
|-------------------------|------|------|------|-------|
| Llama-3-8B-Instruct     | ✅   | ✅   | ✅   | ✅    |
| Llama3.1-8B             | ✅   | ✅   | ✅   | ✅    |
| Llama3.2-3B             | ✅   | ✅   | ✅   | ✅    |
| Qwen2-0.5B-Instruct     | ✅   | ✅   | ✅   | ✅    |
| Qwen2-1.5B-Instruct     | ✅   | ✅   | ✅   | ✅    |
| Qwen2-7B-Instruct       | ✅   | ✅   | ✅   | ✅    |
| Qwen2.5-0.5B-Instruct   | ✅   | ✅   | ✅   | ✅    |
| Qwen2.5-1.5B-Instruct   | ✅   | ✅   | ✅   | ✅    |
| Qwen2.5-3B-Instruct     | ✅   | ✅   | ✅   | ✅    |
| Qwen2.5-7B-Instruct     | ✅   | ✅   | ✅   | ✅    |
"""

import pytest
from _test_utils.examples.run_command import run_llm_export_onnx_command
from _test_utils.model import (
    # Llama models
    LLAMA30_8B_INST_PATH,
    LLAMA31_8B_PATH,
    LLAMA32_3B_PATH,
    # Qwen2 models
    QWEN2_7B_INST_PATH,
    # Qwen2.5 models
    QWEN25_0_5B_INST_PATH,
    QWEN25_3B_INST_PATH,
    QWEN25_7B_INST_PATH,
)

"""_summary_

TORCH_DIR="/models/Qwen2.5-7B-Instruct"
QUANT_TYPE=${QUANT_TYPE:-"nvfp4"}
LM_HEAD_TYPE=${LM_HEAD_TYPE:-"fp16"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"./"}

MODEL_NAME=$(basename "$TORCH_DIR")
OUTPUT_DIR="${OUTPUT_BASE_DIR}${MODEL_NAME}_onnx_export_${QUANT_TYPE}"

python llm_export.py \
--torch_dir "$TORCH_DIR" \
--dtype "$QUANT_TYPE" \
--lm_head "$LM_HEAD_TYPE" \
--output_dir "$OUTPUT_DIR"



"""


###################################################################
# 1. test onnx torch llm quant to onnx runner class TODO
###################################################################
class OnnxTorchLLMQuantToOnnxTestRunner:
    """test runner class for onnx torch llm quant to onnx functionality"""

    def __init__(self):
        self.quantize_mode = None
        self.torch_dir = None
        self.lm_head = None
        self.output_dir = None
        self.calibration_data_size = None
        self.run_eval = None

    def run_torch_llm_quant_to_onnx(
        self, quantize_mode, torch_dir, lm_head, output_dir, calibration_data_size
    ):
        """run torch llm quant to onnx workflow"""
        run_llm_export_onnx_command(
            dtype=quantize_mode,
            torch_dir=torch_dir,
            lm_head=lm_head,
            output_dir=output_dir,
            calib_size=calibration_data_size,
        )


###################################################################
# 2. test case parameters
###################################################################
def create_testcase_params(
    quantize_mode,
    torch_dir,
    lm_head,
    output_dir,
    calibration_data_size,
):
    """create parameterized test case with readable test ID

    format: model_name-quantize_mode-calibration_size-eval_flag
    use underscores within same field values
    use dashes between different fields
    all lowercase conversion
    """

    # convert torch dir with underscores for multi-word values
    torch_dir_str = torch_dir.replace(".", "_").lower()

    # convert quantize mode to lowercase
    quantize_mode_str = quantize_mode.lower()

    # convert calibration data size to string
    calibration_size_str = str(calibration_data_size)

    test_id = f"{torch_dir_str}-{quantize_mode_str}-{calibration_size_str}"
    params = [quantize_mode, torch_dir, lm_head, output_dir, calibration_data_size]
    return pytest.param(*params, id=test_id)


# 2.1. test case parameters
test_torch_llm_quant_to_onnx_params = [
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=QWEN25_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-7B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=QWEN25_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-7B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp16",
        torch_dir=QWEN25_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-7B-Instruct_onnx_export_fp16",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="int4_awq",
        torch_dir=QWEN25_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-7B-Instruct_onnx_export_int4_awq",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=QWEN25_3B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-3B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=QWEN25_3B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-3B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=QWEN25_0_5B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-0.5B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=QWEN25_0_5B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2.5-0.5B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=QWEN2_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2-7B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=QWEN2_7B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2-7B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=QWEN25_0_5B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2-0.5B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=QWEN25_0_5B_INST_PATH,
        lm_head="fp16",
        output_dir="Qwen2-0.5B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=LLAMA30_8B_INST_PATH,
        lm_head="fp16",
        output_dir="Llama3-8B-Instruct_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=LLAMA30_8B_INST_PATH,
        lm_head="fp16",
        output_dir="Llama3-8B-Instruct_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=LLAMA31_8B_PATH,
        lm_head="fp16",
        output_dir="Llama3.1-8B_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=LLAMA31_8B_PATH,
        lm_head="fp16",
        output_dir="Llama3.1-8B_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        torch_dir=LLAMA32_3B_PATH,
        lm_head="fp16",
        output_dir="Llama3.2-3B_onnx_export_nvfp4",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="fp8",
        torch_dir=LLAMA32_3B_PATH,
        lm_head="fp16",
        output_dir="Llama3.2-3B_onnx_export_fp8",
        calibration_data_size=512,
    ),
    create_testcase_params(
        quantize_mode="int4_awq",
        torch_dir=LLAMA32_3B_PATH,
        lm_head="fp16",
        output_dir="Llama3.2-3B_onnx_export_int4_awq",
        calibration_data_size=512,
    ),
]

###################################################################
# 3. test cases
###################################################################


@pytest.mark.parametrize(
    (
        "quantize_mode",
        "torch_dir",
        "lm_head",
        "output_dir",
        "calibration_data_size",
    ),
    test_torch_llm_quant_to_onnx_params,
)
def test_onnx_torch_llm_quant_to_onnx(
    quantize_mode, torch_dir, lm_head, output_dir, calibration_data_size
):
    runner = OnnxTorchLLMQuantToOnnxTestRunner()
    runner.run_torch_llm_quant_to_onnx(
        quantize_mode, torch_dir, lm_head, output_dir, calibration_data_size
    )
