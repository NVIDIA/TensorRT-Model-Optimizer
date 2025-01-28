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

from typing import Union

import onnxruntime as ort
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

from modelopt.onnx.quantization.operators import QDQConvTranspose, QDQNormalization
from modelopt.onnx.quantization.ort_patching import patch_ort_modules


def _prepare_ep_list(calibration_eps: list[str]):
    """Prepares the EP list for ORT from the given user input."""
    providers: list[Union[str, tuple[str, dict]]] = []
    for ep in calibration_eps:
        if "cuda" in ep:
            device_id = int(ep.split(":")[1]) if ":" in ep else 0
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        if "dml" in ep:
            device_id = int(ep.split(":")[1]) if ":" in ep else 0
            providers.append(("DmlExecutionProvider", {"device_id": device_id}))
        if "cpu" in ep:
            providers.append("CPUExecutionProvider")
        if "trt" in ep:
            providers.append("TensorrtExecutionProvider")

    return providers


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
