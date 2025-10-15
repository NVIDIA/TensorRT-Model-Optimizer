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
from _test_utils.examples.run_command import run_example_command
from _test_utils.gpu_arch_utils import skip_if_dtype_unsupported_by_arch
from _test_utils.onnx_path import (
    _ONNX_DEPS_ROOT,
    ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_PATH,
    ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH,
    ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_PATH,
    ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_POST_SIMPLIFIED_PATH,
    ONNX_FAR3D_OPSET17_PATH,
    ONNX_IDENTITY_NEURAL_NETWORK_PATH,
    ONNX_VAD_V1_FORWARD_PATH,
    ONNX_VAD_V1_PREV_FORWARD_PATH,
    ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH,
)

###################################################################
# 1. test onnx quantization with plugin runner class
###################################################################


class OnnxQuantizationWithPluginTestRunner:
    """test runner for ONNX quantization workflows with TensorRT plugin support

    this class provides comprehensive testing for ONNX model quantization
    with TensorRT plugin integration capabilities

    test branches:
    ┌─ quantization modes
    │  ├─ fp8 precision quantization
    │  ├─ int8 precision quantization
    │  └─ int4 precision quantization
    │
    ├─ calibration methods
    │  ├─ max calibration
    │  ├─ entropy calibration
    │  ├─ awq_clip calibration
    │  └─ rtn_dq calibration
    │
    ├─ plugin integration
    │  ├─ bevformer models with custom plugins
    │  ├─ vad models with custom plugins
    │  ├─ vit models with simplified plugins
    │  └─ identity neural network models
    │
    └─ environment configurations
       ├─ TensorRT library path setup
       ├─ cuDNN library path setup
       ├─ CUDA runtime path setup
       └─ plugin library loading

    test workflow:
    1. model preparation: copy and validate model files
    2. calibration data setup: prepare quantization datasets
    3. environment configuration: set TensorRT plugin paths
    4. quantization execution: run quantization with plugin support
    5. output validation: verify generated quantized models
    """

    def __init__(self):
        self.onnx_path = None
        self.model_name = None
        self.quantize_mode = None
        self.calibration_method = None
        self.calibration_data = None
        self.calibration_eps = None
        self.trt_plugins = None
        self.output_path = None

    def _prepare_model(self, onnx_path: str, model_name: str):
        """prepare model by copying if not exists"""
        target_path = f"{model_name}.onnx"
        if not os.path.exists(target_path):
            run_example_command(["cp", "-f", onnx_path, target_path], "onnx_ptq")

    def _prepare_calibration_data(self, calibration_data_path: str):
        """prepare calibration data for quantization"""
        if calibration_data_path is None:
            return
        if not os.path.exists(calibration_data_path):
            raise FileNotFoundError(f"calibration data not found at {calibration_data_path}")

    def _verify_output_model(self, model_path: str, quantize_mode: str):
        """verify that the quantized model was generated successfully"""
        assert os.path.exists(model_path), (
            f"{quantize_mode} quantized model not generated at {model_path}"
        )

    def _get_env_vars(self):
        """get environment variables for onnx quantization with plugin"""
        import os

        # build paths
        trt_root = os.path.join(_ONNX_DEPS_ROOT, "TensorRT-10.11.0.33")
        cudnn_root = os.path.join(_ONNX_DEPS_ROOT, "cudnn-linux-x86_64-9.7.1.26_cuda12-archive")
        cuda_paths = "/usr/local/include:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib:/usr/lib/x86_64-linux-gnu"

        # get current environment variables
        env_vars = os.environ.copy()

        # update environment variables
        env_vars["TRT_ROOT"] = trt_root
        env_vars["CUDNN_ROOT"] = cudnn_root
        env_vars["CUDA_ROOT"] = cuda_paths
        env_vars["PATH"] = f"{trt_root}/bin:{env_vars.get('PATH', '')}"
        env_vars["LD_LIBRARY_PATH"] = (
            f"{trt_root}/lib:{cudnn_root}/lib:{cuda_paths}:{env_vars.get('LD_LIBRARY_PATH', '')}"
        )
        print(f"export TRT_ROOT={trt_root} \\")
        print(f"export CUDNN_ROOT={cudnn_root} \\")
        print(f"export CUDA_ROOT={cuda_paths} \\")
        print(f"export PATH={trt_root}/bin:$PATH \\")
        print(
            f"export LD_LIBRARY_PATH={trt_root}/lib:{cudnn_root}/lib:{cuda_paths}:$LD_LIBRARY_PATH"
        )
        return env_vars

    def run_onnx_quantization_with_plugin(
        self,
        onnx_path: str,
        model_name: str,
        quantize_mode: str,
        calibration_method: str,
        calibration_data: str,
        trt_plugins: str,
    ):
        """run onnx quantization with plugin workflow"""
        # store parameters for reuse
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.quantize_mode = quantize_mode
        self.calibration_method = calibration_method
        self.calibration_data = calibration_data
        self.trt_plugins = trt_plugins
        self.output_path = os.path.join(os.getcwd(), f"{model_name}.{quantize_mode}.onnx")

        # step 1: prepare model and data
        self._prepare_model(onnx_path, model_name)
        self._prepare_calibration_data(calibration_data)

        # step 2: configure environment
        env_vars = self._get_env_vars()

        # step 3: build command arguments for quantization
        cmd_args = [
            "python",
            "-m",
            "modelopt.onnx.quantization",
            "--onnx_path",
            f"{model_name}.onnx",
            "--quantize_mode",
            quantize_mode,
            "--calibration_method",
            calibration_method,
            "--output_path",
            self.output_path,
        ]

        # add optional parameters when available
        if calibration_data is not None:
            cmd_args.extend(["--calibration_data", calibration_data])

        if self.calibration_eps is not None:
            cmd_args.extend(["--calibration_eps", str(self.calibration_eps)])

        if trt_plugins:
            cmd_args.extend(["--trt_plugins", trt_plugins])

        # step 4: execute quantization and validate results
        run_example_command(cmd_args, "onnx_ptq", env_vars=env_vars)
        self._verify_output_model(self.output_path, quantize_mode)

        # step 5: validate with tensorrt
        self.run_onnx_trtexec(
            onnx_path=self.output_path, trt_plugins=trt_plugins, env_vars=env_vars
        )

    def run_onnx_trtexec(self, onnx_path=None, trt_plugins=None, env_vars: dict | None = None):
        """
        run trtexec to validate the quantized onnx model

        args:
            onnx_path: path to the onnx model file
            trt_plugins: path to the tensorrt plugin library
            env_vars: environment variables to set
        """
        # use stored output path if onnx_path not provided
        if onnx_path:
            model_path = onnx_path
        else:
            model_path = self.output_path

        if not model_path:
            raise ValueError("onnx_path is required for trtexec validation")

        # Construct base command
        cmd_args = ["trtexec", "--stronglyTyped", f"--onnx={model_path}"]

        # add plugin library if provided
        if trt_plugins:
            cmd_args.append(f"--staticPlugins={trt_plugins}")

        # Execute the command
        run_example_command(cmd_args, "onnx_ptq", env_vars=env_vars)


###################################################################
# 2. test case parameters
###################################################################
def create_testcase_params(
    onnx_path: str,
    model_name: str,
    quantize_mode: str,
    calibration_method: str,
    calibration_data: str,
    trt_plugins: str,
):
    """create parameterized test case with readable test ID"""
    test_id = f"{model_name.lower()}-{quantize_mode.lower()}-{calibration_method.lower()}-with_trt_plugins"
    params = [
        onnx_path,
        model_name,
        quantize_mode,
        calibration_method,
        calibration_data,
        trt_plugins,
    ]
    return pytest.param(*params, id=test_id)


# 2.1. test case parameters
test_onnx_quantization_with_plugin_params = [
    create_testcase_params(
        onnx_path=ONNX_FAR3D_OPSET17_PATH.model_path,
        model_name="far3d",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_FAR3D_OPSET17_PATH.calib_data_path,
        trt_plugins=ONNX_FAR3D_OPSET17_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_FAR3D_OPSET17_PATH.model_path,
        model_name="far3d",
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_data=ONNX_FAR3D_OPSET17_PATH.calib_data_path,
        trt_plugins=ONNX_FAR3D_OPSET17_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_FAR3D_OPSET17_PATH.model_path,
        model_name="far3d",
        quantize_mode="int4",
        calibration_method="awq_clip",
        calibration_data=ONNX_FAR3D_OPSET17_PATH.calib_data_path,
        trt_plugins=ONNX_FAR3D_OPSET17_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_FAR3D_OPSET17_PATH.model_path,
        model_name="far3d",
        quantize_mode="int4",
        calibration_method="rtn_dq",
        calibration_data=ONNX_FAR3D_OPSET17_PATH.calib_data_path,
        trt_plugins=ONNX_FAR3D_OPSET17_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.model_path,
        model_name="vit_base_opset13_simplified_Conv_LN_withPlugin",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.calib_data_path,
        trt_plugins=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.model_path,
        model_name="vit_base_opset13_simplified_Conv_LN_withPlugin",
        quantize_mode="int8",
        calibration_method="max",
        calibration_data=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.calib_data_path,
        trt_plugins=ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_IDENTITY_NEURAL_NETWORK_PATH.model_path,
        model_name="identity_neural_network",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_IDENTITY_NEURAL_NETWORK_PATH.calib_data_path,
        trt_plugins=ONNX_IDENTITY_NEURAL_NETWORK_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_IDENTITY_NEURAL_NETWORK_PATH.model_path,
        model_name="identity_neural_network",
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_data=ONNX_IDENTITY_NEURAL_NETWORK_PATH.calib_data_path,
        trt_plugins=ONNX_IDENTITY_NEURAL_NETWORK_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_PATH.model_path,
        model_name="bevformer_base_epoch_24_cp2_op13",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_PATH.calib_data_path,
        trt_plugins=ONNX_IDENTITY_NEURAL_NETWORK_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH.model_path,
        model_name="bevformer_base_epoch_24_cp2_op13_post_simp",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH.calib_data_path,
        trt_plugins=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_PATH.model_path,
        model_name="bevformer_base_epoch_24_cp2_op17",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH.calib_data_path,
        trt_plugins=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_POST_SIMPLIFIED_PATH.model_path,
        model_name="bevformer_base_epoch_24_cp2_op17_post_simp",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_POST_SIMPLIFIED_PATH.calib_data_path,
        trt_plugins=ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_POST_SIMPLIFIED_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_VAD_V1_FORWARD_PATH.model_path,
        model_name="vad_v1_forward",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_VAD_V1_FORWARD_PATH.calib_data_path,
        trt_plugins=ONNX_VAD_V1_FORWARD_PATH.trt_plugin_path,
    ),
    create_testcase_params(
        onnx_path=ONNX_VAD_V1_PREV_FORWARD_PATH.model_path,
        model_name="vad_v1_prev_forward",
        quantize_mode="fp8",
        calibration_method="max",
        calibration_data=ONNX_VAD_V1_PREV_FORWARD_PATH.calib_data_path,
        trt_plugins=ONNX_VAD_V1_PREV_FORWARD_PATH.trt_plugin_path,
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
        "trt_plugins",
    ),
    argvalues=test_onnx_quantization_with_plugin_params,
)
def test_onnx_quantization_with_plugin(
    onnx_path, model_name, quantize_mode, calibration_method, calibration_data, trt_plugins
):
    """test onnx quantization with plugin"""

    # skip test if dtype is not supported by arch
    skip_if_dtype_unsupported_by_arch(need_dtype=quantize_mode, need_cpu_arch="x86")

    runner = OnnxQuantizationWithPluginTestRunner()

    # run quantization with plugin
    runner.run_onnx_quantization_with_plugin(
        onnx_path, model_name, quantize_mode, calibration_method, calibration_data, trt_plugins
    )
