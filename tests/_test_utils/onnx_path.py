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
import warnings
from dataclasses import dataclass

_LOCAL_ROOT = os.getenv("MODELOPT_LOCAL_MODEL_ROOT", "/models")


@dataclass
class OnnxPath:
    model_path: str
    trt_plugin_path: str = None
    calib_data_path: str = None
    dataset_path: str = None

    def __post_init__(self):
        """automatically check paths after initialization"""
        self.check_path_exists()

    def check_path_exists(self):
        """check if the model path exists"""

        if not os.path.exists(self.model_path):
            warnings.warn(f"model path does not exist: {self.model_path}")

        if self.trt_plugin_path and not os.path.exists(self.trt_plugin_path):
            warnings.warn(f"trt plugin path does not exist: {self.trt_plugin_path}")

        return True


# common paths for onnx models

_ONNX_ROOT = f"{_LOCAL_ROOT}/modelopt_onnx_ptq"
_ONNX_DEPS_ROOT = f"{_ONNX_ROOT}/deps"
_ONNX_AUTOCAST_ROOT = f"{_ONNX_ROOT}/onnx_autocast"
_ONNX_QUANTIZATION_ROOT = f"{_ONNX_ROOT}/onnx_quantization"
_ONNX_SAMPLE_ROOT = f"{_ONNX_QUANTIZATION_ROOT}/onnx_sample"
_ONNX_NATIVE_TRT_ROOT = f"{_ONNX_QUANTIZATION_ROOT}/onnx_native_trt_plugin"
_ONNX_CUSTOM_TRT_ROOT = f"{_ONNX_QUANTIZATION_ROOT}/onnx_custom_trt_plugins"

# 1. onnx_autocast
ONNX_DISTILBERT_OPSET17_PATH = OnnxPath(
    model_path=f"{_ONNX_AUTOCAST_ROOT}/distilbert_Opset17.onnx",
    trt_plugin_path=None,
)

ONNX_CONVIT_SMALL_OPSET17_PATH = OnnxPath(
    model_path=f"{_ONNX_AUTOCAST_ROOT}/convit_small_Opset17.onnx",
    trt_plugin_path=None,
)

# 2. onnx_sample
ONNX_VIT_BASE_PATCH16_224_PATH = OnnxPath(
    model_path=f"{_ONNX_SAMPLE_ROOT}/vit_base_patch16_224.onnx",
    trt_plugin_path=None,
    calib_data_path=f"{_ONNX_SAMPLE_ROOT}/calib_tiny_imagenet.npy",
    dataset_path=f"{_ONNX_SAMPLE_ROOT}/imagenet",
)

# 3. onnx_native_trt_plugin
ONNX_FAR3D_OPSET17_PATH = OnnxPath(
    model_path=f"{_ONNX_NATIVE_TRT_ROOT}/far3d_opset17.onnx",
    trt_plugin_path=None,
)

ONNX_VIT_BASE_OPSET13_SIMPLIFIED_CONV_LN_WITH_PLUGIN_PATH = OnnxPath(
    model_path=f"{_ONNX_NATIVE_TRT_ROOT}/vit_base_opset13_simplified_Conv_LN_withPlugin.onnx",
    trt_plugin_path=None,
)

# 4. onnx_custom_trt_plugins
# bevformer models

_BEVFORMER_ROOT = f"{_ONNX_CUSTOM_TRT_ROOT}/bevformer_derryhub"
_BEVFORMER_TRT_PLUGIN = (
    f"{_BEVFORMER_ROOT}/libs/libtensorrt_ops_trt10.9.0.33_cuda12.8_cu12_docker24.08.so"
)

# identity neural network model
_IDENTITY_NN_ROOT = f"{_ONNX_CUSTOM_TRT_ROOT}/identity_neural_network"
_IDENTITY_NN_MODEL = (
    f"{_IDENTITY_NN_ROOT}/TensorRT-Custom-Plugin-Example/data/identity_neural_network.onnx"
)
_IDENTITY_NN_PLUGIN = f"{_IDENTITY_NN_ROOT}/10_11_plugin_so/libidentity_conv_iplugin_v2_io_ext.so"

ONNX_IDENTITY_NEURAL_NETWORK_PATH = OnnxPath(
    model_path=_IDENTITY_NN_MODEL,
    trt_plugin_path=_IDENTITY_NN_PLUGIN,
)

ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_PATH = OnnxPath(
    model_path=f"{_BEVFORMER_ROOT}/bevformer_base_epoch_24_cp2_op13.onnx",
    trt_plugin_path=_BEVFORMER_TRT_PLUGIN,
)

ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP13_POST_SIMPLIFIED_PATH = OnnxPath(
    model_path=f"{_BEVFORMER_ROOT}/bevformer_base_epoch_24_cp2_op13_post_simp.onnx",
    trt_plugin_path=_BEVFORMER_TRT_PLUGIN,
)

ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_PATH = OnnxPath(
    model_path=f"{_BEVFORMER_ROOT}/bevformer_base_epoch_24_cp2_op17.onnx",
    trt_plugin_path=_BEVFORMER_TRT_PLUGIN,
)

ONNX_BEVFORMER_BASE_EPOCH_24_CP2_OP17_POST_SIMPLIFIED_PATH = OnnxPath(
    model_path=f"{_BEVFORMER_ROOT}/bevformer_base_epoch_24_cp2_op17_post_simp.onnx",
    trt_plugin_path=_BEVFORMER_TRT_PLUGIN,
)

# vad models
_VAD_ROOT = f"{_ONNX_CUSTOM_TRT_ROOT}/vad"
_VAD_TRT_PLUGIN = f"{_VAD_ROOT}/libplugins.so_x86_trt10.9.0.33_cuda12.8"

ONNX_VAD_V1_FORWARD_PATH = OnnxPath(
    model_path=f"{_VAD_ROOT}/vadv1.pts_bbox_head.forward.onnx",
    trt_plugin_path=_VAD_TRT_PLUGIN,
)

ONNX_VAD_V1_PREV_FORWARD_PATH = OnnxPath(
    model_path=f"{_VAD_ROOT}/vadv1_prev.pts_bbox_head.forward.onnx",
    trt_plugin_path=_VAD_TRT_PLUGIN,
)