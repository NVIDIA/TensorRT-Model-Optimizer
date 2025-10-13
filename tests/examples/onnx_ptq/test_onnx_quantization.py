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


import os

import pytest
from _test_utils.examples.run_command import (
    run_example_command,
    run_onnx_quantization_trtexec_command,
)
from _test_utils.onnx_path import ONNX_VIT_BASE_PATCH16_224_PATH

###################################################################
# 1. test onnx quantization runner class
###################################################################


class OnnxQuantizationTestRunner:
    """test runner class for onnx quantization functionality

    test workflow:
    1. prepare model - copy model file if not exists
    2. run quantization using cli
    3. verify output model generation
    4. validate quantized model with trtexec

    test branches:
    ├─ fp8 quantization (max calibration)
    ├─ int8 quantization (max/entropy calibration)
    └─ int4 quantization (awq_clip/rtn_dq calibration)
    """

    def __init__(self):
        # model parameters for reuse between cli and eval
        self.onnx_path = None
        self.model_name = None
        self.quantize_mode = None
        self.calibration_method = None
        self.calibration_data = None
        self.calibration_eps = None
        self.output_path = None

        # evaluation parameters
        self.imagenet_path = None

    def _prepare_model(self, onnx_path: str, model_name: str):
        """prepare model by copying if not exists"""
        target_path = f"{model_name}.onnx"
        if not os.path.exists(target_path):
            run_example_command(["cp", "-f", onnx_path, target_path], "onnx_ptq")

    def _prepare_calibration_data(self, calibration_data_path: str):
        """prepare calibration data for quantization"""
        if not os.path.exists(calibration_data_path):
            raise FileNotFoundError(f"calibration data not found at {calibration_data_path}")

    def _verify_output_model(self, model_path: str, quantize_mode: str):
        """verify that the quantized model was generated successfully"""
        assert os.path.exists(model_path), (
            f"{quantize_mode} quantized model not generated at {model_path}"
        )

    def _run_trtexec_validation(self, model_path: str, static_plugins: str | None = None):
        """run trtexec validation on the quantized model"""
        run_onnx_quantization_trtexec_command(onnx_path=model_path, static_plugins=static_plugins)

    def run_onnx_quantization_cli(
        self,
        onnx_path: str,
        model_name: str,
        quantize_mode: str,
        calibration_method: str,
        calibration_data: str,
    ):
        """run quantization test using cli command

        args:
            onnx_path: path to the original onnx model
            model_name: name of the model for testing
            quantize_mode: quantization mode (fp8, int8, int4)
            calibration_method: calibration method (max, entropy, awq_clip, rtn_dq)
            calibration_data: path to calibration data
        """
        # store parameters for reuse
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.quantize_mode = quantize_mode
        self.calibration_method = calibration_method
        self.calibration_data = calibration_data
        self.calibration_eps = "trt cpu cuda:0"  # TODO: fix split parsing for trt cuda:0 cpu
        self.output_path = os.path.join(os.getcwd(), f"{model_name}.{quantize_mode}.onnx")

        self._prepare_model(onnx_path, model_name)
        self._prepare_calibration_data(calibration_data)

        cmd_args = [
            "python",
            "-m",
            "modelopt.onnx.quantization",
            "--onnx_path",
            f"{model_name}.onnx",
            "--quantize_mode",
            quantize_mode,
            "--calibration_data",
            calibration_data,
            "--calibration_method",
            calibration_method,
            "--output_path",
            self.output_path,
            "--calibration_eps",
            str(self.calibration_eps),
        ]

        run_example_command(cmd_args, "onnx_ptq")
        self._verify_output_model(self.output_path, quantize_mode)
        # TODO: after fix scaleAllPositive issue
        # self._run_trtexec_validation(model_path=self.output_path, staticPlugins=None)

    def run_onnx_eval(
        self, onnx_path=None, model_name=None, quantize_mode=None, imagenet_path=None
    ):
        """
        Evaluate the quantized onnx model

        the evaluation result will be reported as follows:
            The top1 accuracy of the model is <accuracy score between 0-100%>
            The top5 accuracy of the model is <accuracy score between 0-100%>
            Inference latency of the model is <X> ms
        """
        # use provided parameters or fall back to stored values
        eval_onnx_path = onnx_path or self.output_path
        eval_model_name = model_name or self.model_name
        eval_quantize_mode = quantize_mode or self.quantize_mode
        eval_imagenet_path = imagenet_path or self.imagenet_path

        # validate required parameters
        if not eval_onnx_path:
            raise ValueError("onnx_path is required for evaluation")
        if not eval_model_name:
            raise ValueError("model_name is required for evaluation")
        if not eval_quantize_mode:
            raise ValueError("quantize_mode is required for evaluation")
        if not eval_imagenet_path:
            raise ValueError("imagenet_path is required for evaluation")

        cmd_args = [
            "python",
            "evaluate.py",
            "--onnx_path",
            eval_onnx_path,
            "--imagenet_path",
            eval_imagenet_path,
            "--quantize_mode",
            eval_quantize_mode,
            "--model_name",
            eval_model_name,
            "--results_path",
            f"{eval_model_name}.{eval_quantize_mode}.results.csv",
        ]
        run_example_command(cmd_args, "onnx_ptq")


###################################################################
# 2. test case parameters
###################################################################
def create_testcase_params(
    onnx_path: str,
    model_name: str,
    quantize_mode: str,
    calibration_method: str,
    calibration_data: str,
    run_eval: bool = True,
    imagenet_path: str | None = None,
):
    """create parameterized test case with readable test ID

    format: model_name-quantize_mode-calibration_method-run_eval
    use dash between different parameter categories
    use underscore for multi-word field values
    all lowercase conversion
    """
    # convert model name with underscores for multi-word values
    model_name_str = model_name.lower()

    # convert quantize mode to lowercase
    quantize_mode_str = quantize_mode.lower()

    # convert calibration method to lowercase with underscores
    calibration_method_str = calibration_method.lower()

    # convert run_eval boolean to lowercase string
    run_eval_str = str(run_eval).lower()

    test_id = f"{model_name_str}-{quantize_mode_str}-{calibration_method_str}-{run_eval_str}"
    params = [
        onnx_path,
        model_name,
        quantize_mode,
        calibration_method,
        calibration_data,
        run_eval,
        imagenet_path,
    ]
    return pytest.param(*params, id=test_id)


# 2.1. test case parameters
test_onnx_quantization_params = [
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_PATCH16_224_PATH.model_path,
        model_name="vit_base_patch16_224",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_VIT_BASE_PATCH16_224_PATH.calib_data_path,
        imagenet_path=ONNX_VIT_BASE_PATCH16_224_PATH.dataset_path,
        run_eval=True,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_PATCH16_224_PATH.model_path,
        model_name="vit_base_patch16_224",
        quantize_mode="int8",
        calibration_method="max",
        calibration_data=ONNX_VIT_BASE_PATCH16_224_PATH.calib_data_path,
        imagenet_path=ONNX_VIT_BASE_PATCH16_224_PATH.dataset_path,
        run_eval=True,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_PATCH16_224_PATH.model_path,
        model_name="vit_base_patch16_224",
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_data=ONNX_VIT_BASE_PATCH16_224_PATH.calib_data_path,
        imagenet_path=ONNX_VIT_BASE_PATCH16_224_PATH.dataset_path,
        run_eval=True,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_PATCH16_224_PATH.model_path,
        model_name="vit_base_patch16_224",
        quantize_mode="int4",
        calibration_method="awq_clip",
        calibration_data=ONNX_VIT_BASE_PATCH16_224_PATH.calib_data_path,
        imagenet_path=ONNX_VIT_BASE_PATCH16_224_PATH.dataset_path,
        run_eval=True,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_PATCH16_224_PATH.model_path,
        model_name="vit_base_patch16_224",
        quantize_mode="int4",
        calibration_method="rtn_dq",
        calibration_data=ONNX_VIT_BASE_PATCH16_224_PATH.calib_data_path,
        imagenet_path=ONNX_VIT_BASE_PATCH16_224_PATH.dataset_path,
        run_eval=True,
    ),
]

###################################################################
# 3. test cases
###################################################################


@pytest.mark.parametrize(
    argnames=(
        "onnx_path",
        "model_name",
        "quantize_mode",
        "calibration_method",
        "calibration_data",
        "run_eval",
        "imagenet_path",
    ),
    argvalues=test_onnx_quantization_params,
)
def test_onnx_quantization(
    onnx_path,
    model_name,
    quantize_mode,
    calibration_method,
    calibration_data,
    run_eval,
    imagenet_path,
):
    """test onnx quantization workflow with different quantization modes and calibration methods"""
    runner = OnnxQuantizationTestRunner()

    # run quantization using CLI
    runner.run_onnx_quantization_cli(
        onnx_path, model_name, quantize_mode, calibration_method, calibration_data
    )

    # run evaluation if required
    if run_eval:
        runner.run_onnx_eval(onnx_path, model_name, quantize_mode, imagenet_path)