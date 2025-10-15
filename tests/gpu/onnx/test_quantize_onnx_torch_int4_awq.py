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

# NOTE: This test requires modelopt.torch.quantization to be installed as well.

import copy
import os
from functools import partial

import torch
from _test_utils.import_helper import skip_if_no_libcudnn
from _test_utils.onnx_quantization.lib_test_models import SimpleMLP, export_as_onnx, find_init
from _test_utils.torch_quantization.quantize_common import get_awq_config

import modelopt.onnx.quantization.int4 as int4
import modelopt.torch.quantization as mtq
from modelopt.onnx.quantization.int4 import dq_tensor
from modelopt.onnx.quantization.int4 import quantize as quantize_int4

if int4.has_cupy:
    import cupy as np
else:
    import numpy as np

# TODO: Rename this script to test_onnx_torch_int4_awq.py
#       For that, we need to investigate failure in 'pytest tests/gpu/onnx'.
#       test_qdq_utils_fp8.py::test_fused_q[bf16,fp16] fails if this script runs after the int4 test, but not before.


def test_safe_cupy_array_all_paths(monkeypatch):
    """Test safe_cupy_array covering all code paths including ml_dtypes handling"""
    # Test 1: When ml_dtypes import fails (covers ImportError path)
    # Temporarily remove ml_dtypes from sys.modules
    import sys

    if "ml_dtypes" in sys.modules:
        ml_dtypes_backup = sys.modules["ml_dtypes"]
        monkeypatch.delitem(sys.modules, "ml_dtypes")
    else:
        ml_dtypes_backup = None

    tensor = np.array([1, 2, 3, 4], dtype=np.int8)
    result = int4.safe_cupy_array(tensor)
    assert isinstance(result, np.ndarray)  # Should return numpy array

    # Restore ml_dtypes if it existed
    if ml_dtypes_backup:
        sys.modules["ml_dtypes"] = ml_dtypes_backup

    # Test 2: When ml_dtypes exists and tensor has ml_dtypes.int4 dtype
    try:
        import ml_dtypes

        # Create a mock tensor with int4 dtype
        class MockInt4Tensor:
            def __init__(self, data):
                self.data = data
                self.dtype = ml_dtypes.int4
                self.shape = data.shape

            def astype(self, dtype):
                return self.data.astype(dtype)

            def __array__(self):
                return self.data

        mock_tensor = MockInt4Tensor(np.array([1, 2, 3, 4], dtype=np.int8))
        result = int4.safe_cupy_array(mock_tensor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int8
    except ImportError:
        # ml_dtypes not available, skip this part
        pass

    # Test 3: Normal case with regular numpy array
    tensor = np.array([1, 2, 3, 4], dtype=np.int8)
    result = int4.safe_cupy_array(tensor)
    # Should work normally
    assert isinstance(result, (np.ndarray, type(tensor)))


def test_int4_awq(tmp_path):
    def _forward_loop(model, dataloader):
        """Forward loop for calibration."""
        for data in dataloader:
            model(data)

    block_size = 128

    model_torch = SimpleMLP().cuda()
    input_tensor = torch.randn(2, 16, 16).cuda()
    dataloader = [torch.randn(2, 16, 16).cuda()]

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    onnx_dataloader = [{"input": dataloader[0].cpu().numpy()}]
    onnx_model_awq_lite = quantize_int4(
        onnx_path,
        "awq_lite",
        onnx_dataloader,
        block_size=block_size,
        use_external_data_format=False,
        awqlite_fuse_nodes=False,
    )
    onnx_model_awq_clip = quantize_int4(
        onnx_path,
        "awq_clip",
        onnx_dataloader,
        block_size=block_size,
        use_external_data_format=False,
    )

    wq_names = ["onnx::MatMul_12_i4", "onnx::MatMul_13_i4", "onnx::MatMul_14_i4"]
    scale_names = ["onnx::MatMul_12_scale", "onnx::MatMul_13_scale", "onnx::MatMul_14_scale"]

    # Test scale factor computations.
    model_torch_copy = copy.deepcopy(model_torch)
    mtq.quantize(
        model_torch,
        get_awq_config(algorithm="awq_lite", block_size=block_size),
        partial(_forward_loop, model_torch, dataloader),
    )
    mtq.quantize(
        model_torch_copy,
        get_awq_config(algorithm="awq_clip", block_size=block_size),
        partial(_forward_loop, model_torch_copy, dataloader),
    )
    for i in [0, 1, 2]:
        wq_torch_awq_lite = model_torch.net[i * 2].weight_quantizer(model_torch.net[i * 2].weight)
        wq_onnx_awq_lite = find_init(onnx_model_awq_lite, wq_names[i])
        scale_awq_lite = find_init(onnx_model_awq_lite, scale_names[i])

        if int4.has_cupy:
            wq_onnx_awq_lite = int4.safe_cupy_array(wq_onnx_awq_lite)
            scale_awq_lite = int4.safe_cupy_array(scale_awq_lite)

        wq_onnx_awq_lite = dq_tensor(wq_onnx_awq_lite, scale_awq_lite, block_size)
        wq_torch_awq_clip = model_torch_copy.net[i * 2].weight_quantizer(
            model_torch_copy.net[i * 2].weight
        )
        wq_onnx_awq_clip = find_init(onnx_model_awq_clip, wq_names[i])
        scale_awq_clip = find_init(onnx_model_awq_clip, scale_names[i])

        if int4.has_cupy:
            wq_onnx_awq_clip = int4.safe_cupy_array(wq_onnx_awq_clip)
            scale_awq_clip = int4.safe_cupy_array(scale_awq_clip)

        wq_onnx_awq_clip = dq_tensor(wq_onnx_awq_clip, scale_awq_clip, block_size)

        assert np.allclose(wq_torch_awq_lite.detach(), wq_onnx_awq_lite.T, atol=1e-3)
        assert np.allclose(wq_torch_awq_clip.detach(), wq_onnx_awq_clip.T, atol=1e-3)


def test_int4_awq_cuda(tmp_path):
    skip_if_no_libcudnn()
    block_size = 128

    model_torch = SimpleMLP().cuda()
    input_tensor = torch.randn(2, 16, 16).cuda()

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    onnx_model = quantize_int4(
        onnx_path,
        "awq_lite",
        calibration_data_reader=None,
        block_size=block_size,
        use_external_data_format=False,
        awqlite_fuse_nodes=False,
        calibration_eps=["cuda:0", "cpu"],
    )

    assert onnx_model is not None
