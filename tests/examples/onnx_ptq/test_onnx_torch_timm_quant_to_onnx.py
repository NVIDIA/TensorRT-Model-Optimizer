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

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from _test_utils.examples.run_command import run_torch_timm_onnx_command


###################################################################
# 1. test onnx torch quant to onnx runner class TODO
###################################################################
class OnnxTorchTimmQuantToOnnxTestRunner:
    """test runner class for onnx torch timm quant to onnx functionality"""

    def __init__(self):
        self.quantize_mode = None
        self.timm_model_name = None
        self.onnx_save_path = None
        self.calibration_data_size = None
        self.run_eval = None

    def run_torch_quant_to_onnx(
        self,
        quantize_mode: str,
        timm_model_name: str,
        onnx_save_path: str,
        calibration_data_size: int,
    ):
        """run torch quant to onnx workflow"""
        run_torch_timm_onnx_command(
            quantize_mode=quantize_mode,
            timm_model_name=timm_model_name,
            onnx_save_path=onnx_save_path,
            calibration_data_size=calibration_data_size,
        )

    def run_accuracy_evaluation(
        self,
        quantize_mode: str,
        timm_model_name: str,
        onnx_save_path: str,
        calibration_data_size: int,
    ):
        """run accuracy evaluation workflow"""
        # TODO: Add accuracy evaluation after we upgrade TRT version to 10.12
        if self.run_eval:
            pass
        else:
            pass


###################################################################
# 2. test case parameters
###################################################################


def create_testcase_params(
    quantize_mode: str,
    timm_model_name: str,
    onnx_save_path: str,
    calibration_data_size: int,
    run_eval: bool = True,
):
    """create parameterized test case with readable test ID

    format: model_name-quantize_mode-calibration_size-eval_flag
    use underscores within same field values
    use dashes between different fields
    all lowercase conversion
    """
    # convert model name with underscores for multi-word values
    model_name_str = timm_model_name.replace(".", "_").lower()

    # convert quantize mode to lowercase
    quantize_mode_str = quantize_mode.lower()

    # convert calibration data size to string
    calibration_size_str = str(calibration_data_size)

    # convert run_eval boolean to lowercase string
    run_eval_str = str(run_eval).lower()

    test_id = f"{model_name_str}-{quantize_mode_str}-{calibration_size_str}-{run_eval_str}"
    params = [quantize_mode, timm_model_name, onnx_save_path, calibration_data_size, run_eval]
    return pytest.param(*params, id=test_id)


# 2.1. test case parameters
test_torch_quant_to_onnx_params = [
    create_testcase_params(
        quantize_mode="nvfp4",
        timm_model_name="vit_base_patch16_224",
        onnx_save_path="vit_base_patch16_224.nvfp4.onnx",
        calibration_data_size=512,
        run_eval=True,
    ),
    create_testcase_params(
        quantize_mode="nvfp4",
        timm_model_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        onnx_save_path="vit_tiny_patch16_224.augreg_in21k_ft_in1k.nvfp4.onnx",
        calibration_data_size=512,
        run_eval=True,
    ),
    create_testcase_params(
        quantize_mode="mxfp8",
        timm_model_name="vit_base_patch16_224",
        onnx_save_path="vit_base_patch16_224.mxfp8.onnx",
        calibration_data_size=512,
        run_eval=True,
    ),
]

###################################################################
# 3. test cases
###################################################################


@pytest.mark.parametrize(
    (
        "quantize_mode",
        "timm_model_name",
        "onnx_save_path",
        "calibration_data_size",
        "run_eval",
    ),
    test_torch_quant_to_onnx_params,
)
def test_onnx_torch_quant_to_onnx(
    quantize_mode, timm_model_name, onnx_save_path, calibration_data_size, run_eval
):
    runner = OnnxTorchTimmQuantToOnnxTestRunner()
    runner.run_torch_quant_to_onnx(
        quantize_mode, timm_model_name, onnx_save_path, calibration_data_size
    )
    if run_eval:
        runner.run_accuracy_evaluation(
            quantize_mode, timm_model_name, onnx_save_path, calibration_data_size
        )
