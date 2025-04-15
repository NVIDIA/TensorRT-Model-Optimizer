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

"""Provides basic ORT inference utils, shoule be replaced by modelopt.torch.ort_client."""

import glob
import logging
import os
import platform
from typing import Union

import onnxruntime as ort
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
from packaging.version import Version

from modelopt.onnx.quantization.operators import QDQConvTranspose, QDQNormalization
from modelopt.onnx.quantization.ort_patching import patch_ort_modules


def _check_for_tensorrt(min_version: str = "10.0"):
    """Check if the `tensorrt` python package is installed and that it's >= min_version."""
    try:
        import tensorrt

        assert Version(tensorrt.__version__) >= Version(min_version)
        logging.info(
            f"Successfully imported the `tensorrt` python package with version {tensorrt.__version__}."
        )
    except (AssertionError, ImportError):
        raise ImportError(
            f"Could not import the `tensorrt` python package. Please install `tensorrt>={min_version}`"
            " to use ORT's TensorRT Execution Provider. For more information on version compatibility,"
            " please check https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements."
        )


def _check_for_libcudnn():
    lib_pattern = "*cudnn*.dll" if platform.system() == "Windows" else "libcudnn_adv*.so*"
    env_variable = "PATH" if platform.system() == "Windows" else "LD_LIBRARY_PATH"
    ld_library_path = os.environ.get(env_variable, "").split(os.pathsep)

    def _check_lib_in_ld_library_path(lib_pattern):
        for directory in ld_library_path:
            matches = glob.glob(os.path.join(directory, lib_pattern))
            if matches:
                return True, matches[0]
        return False, None

    found, lib_path = _check_lib_in_ld_library_path(lib_pattern)
    if found:
        logging.info(
            f"{lib_pattern} is accessible in {lib_path}! Please check that this is the correct version needed"
            f" for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements."
        )
    else:
        raise FileNotFoundError(
            f"{lib_pattern} is not accessible in {env_variable}! Please make sure that the path to that library"
            f" is in the env var to use the CUDA or TensorRT EP and ensure that the correct version is available."
            f" Versioning compatibility can be checked at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements."
        )
    return found


def _prepare_ep_list(calibration_eps: list[str]):
    """Prepares the EP list for ORT from the given user input."""
    providers: list[Union[str, tuple[str, dict]]] = []
    for ep in calibration_eps:
        if "cuda" in ep:
            try:
                _check_for_libcudnn()
                device_id = int(ep.split(":")[1]) if ":" in ep else 0
                providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
            except Exception as e:
                logging.warning(f"Failed to enable ORT with CUDA EP: '{e}'")
        elif "dml" in ep:
            device_id = int(ep.split(":")[1]) if ":" in ep else 0
            providers.append(("DmlExecutionProvider", {"device_id": device_id}))
        elif "cpu" in ep:
            providers.append("CPUExecutionProvider")
        elif "trt" in ep:
            try:
                _check_for_tensorrt()
                _check_for_libcudnn()
                providers.append("TensorrtExecutionProvider")
            except Exception as e:
                logging.warning(f"Failed to enable ORT with TensorRT EP: '{e}'")
        else:
            raise NotImplementedError(f"Execution Provider {ep} not recognized!")

    logging.info(f"Successfully enabled {len(providers)} EPs for ORT: {providers}")
    return providers


def update_trt_ep_support(
    calibration_eps: list[str], has_dds_op: bool, has_custom_op: bool, trt_plugins: str
):
    """Checks whether TRT should be enabled or disabled and updates the list of calibration EPs accordingly."""

    def _make_trt_ep_first_choice(calibration_eps, trt_plugins):
        # Ensure that TRT EP is enabled for models with custom ops.
        # If it's already enabled, ensure that it's the first EP in the list of EPs to mitigate fallback issues.
        if "trt" in calibration_eps:
            calibration_eps.remove("trt")
        calibration_eps.insert(0, "trt")

        # If the model has a custom op and no plugin path was given, assume that this custom op is being implemented
        # by a TRT native plugin. In order to enable the TRT EP, 'trt_extra_plugin_lib_paths' needs to be != None.
        if not trt_plugins:
            trt_plugins = ""
        return trt_plugins

    if has_dds_op:
        if "trt" in calibration_eps:
            try:
                _check_for_tensorrt(min_version="10.6")
            except AssertionError:
                logging.warning(
                    "This model contains DDS ops, which are only supported in ORT with TensorRT EP backend "
                    "for TRT versions 10.6+. Please update TRT to a supported version, disabling it for now. "
                    "See https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops"
                )
                calibration_eps.remove("trt")
            try:
                assert Version(ort.__version__) >= Version("1.21.0")
            except AssertionError:
                logging.warning(
                    "This model contains DDS ops, please upgrade your ORT version to 1.21.0+ for full support with "
                    "TRT EP backend. The reason for that is because DDS ops require TRT 10.6+, and earlier versions "
                    "of ORT were compiled with TRT 10.4 or lower. Disabling it for now. "
                    "See https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops"
                )
                calibration_eps.remove("trt")
        if has_custom_op:
            if "trt" in calibration_eps:
                logging.warning(
                    "This model contains DDS and custom ops, which should be supported in ORT with TensorRT EP "
                    "backend from TRT 10.6+. If you still encounter errors, please try using the '--simplify' "
                    "flag to simplify your model, as it may be able to remove some problematic ops."
                )
                trt_plugins = _make_trt_ep_first_choice(calibration_eps, trt_plugins)
            else:
                raise Exception(
                    "This model contains DDS and custom ops. Custom ops are only supported with the TensorRT EP, but "
                    "that has been disabled. Please update your TRT and/or ORT version."
                )
    elif has_custom_op:
        logging.info("Custom op detected. Enabling TensorRT EP.")
        trt_plugins = _make_trt_ep_first_choice(calibration_eps, trt_plugins)

    return trt_plugins


def create_inference_session(onnx_path_or_model: Union[str, bytes], calibration_eps: list[str]):
    """Create an ORT InferenceSession."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        onnx_path_or_model,
        sess_options=sess_options,
        providers=_prepare_ep_list(calibration_eps),
    )


def get_quantizable_op_types(op_types_to_quantize: list[str]) -> list[str]:
    """Returns a set of quantizable op types.

    Note. This function should be called after quantize._configure_ort() is called once.
    This returns quantizable op types either from the user supplied parameter
    or from modelopt.onnx's default quantizable ops setting.
    """
    op_types_to_quantize = op_types_to_quantize or []

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    return op_types_to_quantize


def configure_ort(
    op_types: list[str],
    op_types_to_quantize: list[str],
    trt_extra_plugin_lib_paths: str = None,
    calibration_eps: list[str] = None,
):
    """Configure and patches ORT to support ModelOpt ONNX quantization."""
    # Register some new QDQ operators on top of ORT
    QDQRegistry["BatchNormalization"] = QDQNormalization
    QDQRegistry["ConvTranspose"] = QDQConvTranspose
    QDQRegistry["LRN"] = QDQNormalization  # Example: caffenet-12.onnx
    QDQRegistry["HardSwish"] = (
        QDQOperatorBase  # Example: mobilenet_v3_opset17, efficientvit_b3_opset17
    )

    # Patch ORT modules to fix bugs and support some edge cases
    patch_ort_modules()

    # Remove copy, reduction and activation ops from ORT QDQ registry
    for op_type in [
        "ArgMax",
        "Concat",
        "EmbedLayerNormalization",
        "Gather",
        "InstanceNormalization",
        "LeakyRelu",
        "Pad",
        "Relu",
        "Reshape",
        "Slice",
        "Sigmoid",
        "Softmax",
        "Split",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]:
        if op_type in QLinearOpsRegistry:
            del QLinearOpsRegistry[op_type]
        if op_type in QDQRegistry:
            del QDQRegistry[op_type]

    # Prepare TensorRT friendly quantization settings
    trt_guided_options = {
        "QuantizeBias": False,
        "ActivationSymmetric": True,
        "OpTypesToExcludeOutputQuantization": op_types,  # No output quantization
        "AddQDQPairToWeight": True,  # Instead of quantizing the weights, add QDQ node
        "QDQOpTypePerChannelSupportToAxis": {
            "Conv": 0,
            "ConvTranspose": 1,
        },  # per_channel should be True
        "DedicatedQDQPair": True,
        "ForceQuantizeNoInputCheck": (
            # By default, for some latent operators like MaxPool, Transpose, etc.,
            # ORT does not quantize if their input is not quantized already.
            True
        ),
        "TrtExtraPluginLibraryPaths": trt_extra_plugin_lib_paths,
        "ExecutionProviders": _prepare_ep_list(calibration_eps),
    }

    quantizable_op_types = get_quantizable_op_types(op_types_to_quantize)
    return trt_guided_options, quantizable_op_types
