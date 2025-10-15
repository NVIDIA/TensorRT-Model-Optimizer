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


"""Integration tests for ONNX AutoCast functionality.

These tests verify end-to-end workflows for the AutoCast tool using both
mock models (simple test models) and release models (realistic models).
Tests cover CLI commands, Python API, and complete conversion workflows.
"""

import os

import onnx
import pytest
from _test_utils.examples.run_command import (
    run_example_command,
    run_onnx_autocast_cli_command,
    run_onnx_quantization_trtexec_command,
)
from _test_utils.gpu_arch_utils import skip_if_dtype_unsupported_by_arch
from _test_utils.onnx_path import (
    _ONNX_DEPS_ROOT,
    ONNX_CONVIT_SMALL_OPSET17_PATH,
    ONNX_DISTILBERT_OPSET17_PATH,
    OnnxPath,
)

###################################################################
# 1. test onnx autocast runner class
###################################################################


class OnnxAutocastTestRunner:
    """test runner class for onnx autocast functionality

    test workflow:
    1. prepare model - copy model file if not exists
    2. run autocast conversion (cli or api)
    3. verify output model generation
    4. validate converted model with trtexec

    test branches:
    ┌─ cli method
    │  ├─ fp16 precision
    │  ├─ bf16 precision
    │  └─ keep_io_types variations
    │
    └─ api method
       ├─ fp16 precision
       ├─ bf16 precision
       ├─ keep_io_types variations
       └─ provider selection (cpu, cuda)
    """

    def __init__(self):
        pass

    def _prepare_model(self, onnx_path, model_name: str):
        """prepare model by copying if not exists

        args:
            onnx_path: OnnxPath object or string path
            model_name: name of the model
        """
        # convert OnnxPath object to string path if needed
        if isinstance(onnx_path, OnnxPath):
            onnx_path_str = onnx_path.model_path
        else:
            onnx_path_str = onnx_path

        target_path = f"{model_name}.onnx"
        if not os.path.exists(target_path):
            run_example_command(["cp", "-f", onnx_path_str, target_path], "onnx_ptq")

    def _verify_output_model(self, model_path: str, precision_type: str):
        """verify that the output model was generated successfully"""
        assert os.path.exists(model_path), f"{precision_type} model not generated at {model_path}"

    def _run_trtexec_validation(self, model_path: str, env_vars: dict | None = None):
        """run trtexec validation on the generated model"""
        run_onnx_quantization_trtexec_command(
            onnx_path=model_path,
            staticPlugins=None,
            env_vars=env_vars,
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

    def run_cli_test(
        self,
        onnx_path,
        model_name: str,
        low_precision_type: str,
        keep_io_types: bool,
        providers: list[str],
    ):
        """run complete CLI-based autocast test workflow

        args:
            onnx_path: OnnxPath object or string path
            model_name: name of the model
            low_precision_type: target precision (fp16, bf16, etc.)
            keep_io_types: whether to keep input/output types
            providers: list of execution providers
        """
        # convert OnnxPath object to string path if needed
        if isinstance(onnx_path, OnnxPath):
            onnx_path_str = onnx_path.model_path
        else:
            onnx_path_str = onnx_path

        # step 1: prepare model
        self._prepare_model(onnx_path, model_name)

        # step 2: run autocast conversion using CLI
        output_path = f"{model_name}.{low_precision_type}.onnx"
        env_vars = self._get_env_vars()
        run_onnx_autocast_cli_command(
            onnx_path=onnx_path_str,
            output_path=output_path,
            low_precision_type=low_precision_type,
            keep_io_types=keep_io_types,
            providers=providers,
            env_vars=env_vars,
        )

        # step 3: verify output model generation
        self._verify_output_model(output_path, low_precision_type)

        # step 4: validate converted model with trtexec
        self._run_trtexec_validation(output_path, env_vars)

    def run_api_test(
        self,
        onnx_path,
        model_name: str,
        low_precision_type: str,
        keep_io_types: bool,
        providers: list[str],
    ):
        """run complete API-based autocast test workflow

        args:
            onnx_path: OnnxPath object or string path
            model_name: name of the model
            low_precision_type: target precision (fp16, bf16, etc.)
            keep_io_types: whether to keep input/output types
            providers: list of execution providers (not used in API test)
        """
        # convert OnnxPath object to string path if needed
        if isinstance(onnx_path, OnnxPath):
            onnx_path_str = onnx_path.model_path
        else:
            onnx_path_str = onnx_path

        # step 1: prepare model
        self._prepare_model(onnx_path, model_name)

        from modelopt.onnx.autocast import convert_to_mixed_precision

        # step 2: run autocast conversion using API
        converted_model = convert_to_mixed_precision(
            onnx_path=onnx_path_str,
            low_precision_type=low_precision_type,
            nodes_to_exclude=None,
            op_types_to_exclude=None,
            data_max=512,
            init_max=65504,
            keep_io_types=keep_io_types,
            calibration_data=None,
        )

        output_path = f"{model_name}.{low_precision_type}.onnx"
        env_vars = self._get_env_vars()
        onnx.save(converted_model, output_path)

        # step 3: verify output model generation
        self._verify_output_model(output_path, low_precision_type)

        # step 4: validate converted model with trtexec
        self._run_trtexec_validation(output_path, env_vars)


###################################################################
# 2. test case parameters
###################################################################


def create_testcase_params(
    onnx_path,
    model_name: str,
    low_precision_type: str,
    keep_io_types: bool,
    providers: list[str],
):
    """create parameterized test case with readable test ID

    args:
        onnx_path: OnnxPath object or string path
        model_name: name of the model
        low_precision_type: target precision (fp16, bf16, etc.)
        keep_io_types: whether to keep input/output types
        providers: list of execution providers

    format: model_name-precision_type-keep_io_flag-providers_list
    use underscores within same field values
    use dashes between different fields
    all lowercase conversion
    """
    # convert providers list to string with underscores
    providers_str = "_".join(providers).lower()

    # convert boolean to readable string
    if keep_io_types:
        keep_io_str = "keep_io_true"
    else:
        keep_io_str = "keep_io_false"

    # build test ID with dashes between different fields and lowercase
    test_id = f"{model_name.lower()}-{low_precision_type.lower()}-{keep_io_str}-{providers_str}"

    params = [onnx_path, model_name, low_precision_type, keep_io_types, providers]
    return pytest.param(*params, id=test_id)


# 2.1. test case parameters for CLI test
test_onnx_quantization_autocast_cli_params = [
    create_testcase_params(
        onnx_path=ONNX_DISTILBERT_OPSET17_PATH,
        model_name="distilbert_Opset17",
        low_precision_type="fp16",
        keep_io_types=True,
        providers=["trt", "cuda", "cpu"],
    ),
    create_testcase_params(
        onnx_path=ONNX_DISTILBERT_OPSET17_PATH,
        model_name="distilbert_Opset17",
        low_precision_type="bf16",
        keep_io_types=False,
        providers=["cuda"],
    ),
    create_testcase_params(
        onnx_path=ONNX_CONVIT_SMALL_OPSET17_PATH,
        model_name="convit_small_Opset17",
        low_precision_type="fp16",
        keep_io_types=True,
        providers=["trt", "cuda", "cpu"],
    ),
]


# 2.2. test case parameters for API test
test_onnx_quantization_autocast_api_params = [
    create_testcase_params(
        onnx_path=ONNX_DISTILBERT_OPSET17_PATH,
        model_name="distilbert_Opset17",
        low_precision_type="fp16",
        keep_io_types=True,
        providers=["trt", "cuda", "cpu"],
    ),
    create_testcase_params(
        onnx_path=ONNX_DISTILBERT_OPSET17_PATH,
        model_name="distilbert_Opset17",
        low_precision_type="bf16",
        keep_io_types=False,
        providers=["cuda"],
    ),
    create_testcase_params(
        onnx_path=ONNX_CONVIT_SMALL_OPSET17_PATH,
        model_name="convit_small_Opset17",
        low_precision_type="fp16",
        keep_io_types=True,
        providers=["trt", "cuda", "cpu"],
    ),
]

###################################################################
# 3. test cases
###################################################################


@pytest.mark.parametrize(
    argnames=("onnx_path", "model_name", "low_precision_type", "keep_io_types", "providers"),
    argvalues=test_onnx_quantization_autocast_cli_params,
)
def test_onnx_quantization_autocast_cli(
    onnx_path, model_name, low_precision_type, keep_io_types, providers
):
    # skip test if dtype is not supported by arch#
    skip_if_dtype_unsupported_by_arch(need_dtype=low_precision_type, need_cpu_arch="x86")

    runner = OnnxAutocastTestRunner()
    runner.run_cli_test(onnx_path, model_name, low_precision_type, keep_io_types, providers)


@pytest.mark.parametrize(
    argnames=("onnx_path", "model_name", "low_precision_type", "keep_io_types", "providers"),
    argvalues=test_onnx_quantization_autocast_api_params,
)
def test_onnx_quantization_autocast_api(
    onnx_path, model_name, low_precision_type, keep_io_types, providers
):
    # skip test if dtype is not supported by arch
    skip_if_dtype_unsupported_by_arch(need_dtype=low_precision_type, need_cpu_arch="x86")

    runner = OnnxAutocastTestRunner()
    runner.run_api_test(onnx_path, model_name, low_precision_type, keep_io_types, providers)
