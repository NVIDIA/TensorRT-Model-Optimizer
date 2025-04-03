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

import os
from typing import Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from _test_utils.onnx_quantization.lib_test_models import find_init

import modelopt.onnx.quantization as moq
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.utils import save_onnx


def _matmul_model(w: np.ndarray, in_shape: Sequence[int], out_shape: Sequence[int], tmp_path):
    # Assumes
    w = gs.Constant("w", w)
    x = gs.Variable("x", dtype=np.float32, shape=in_shape)
    y = gs.Variable("y", dtype=np.float32, shape=out_shape)
    mm = gs.Node("MatMul", "mm", inputs=[x, w], outputs=[y])
    g = gs.Graph([mm], inputs=[x, w], outputs=[y])

    onnx_model = gs.export_onnx(g)
    onnx_path = os.path.join(tmp_path, "model.onnx")
    save_onnx(onnx_model, onnx_path)

    return onnx_path


def test_int4_rtn(tmp_path):
    # Test scale factor computation.
    # Use moq.quantize once to check that path doesnt have any bugs
    onnx_path = _matmul_model(
        w=np.asarray([[0.5, 1.5], [0.875, 1.75]]),
        in_shape=(1, 2),
        out_shape=(1, 2),
        tmp_path=tmp_path,
    )
    output_path = os.path.join(tmp_path, "model_int4.onnx")
    moq.quantize(onnx_path, "int4", calibration_method="rtn", output_path=output_path, block_size=8)
    onnx_model = onnx.load(output_path)

    node_names = [node.name for node in onnx_model.graph.node]
    assert "w_QuantizeLinear" in node_names
    assert "w_DequantizeLinear" in node_names

    s = find_init(onnx_model, "w_scale")
    assert np.array_equal(s, np.asarray([[0.125, 0.25]]))

    # Test multiple blocks.
    onnx_path = _matmul_model(
        w=np.asarray(
            [
                # Block 0.
                [0.5],
                [-0.5],
                [0.75],
                [-0.75],
                [0.875],
                [-0.875],
                [0.5],
                [-0.5],
                # Block 1.
                [0.25],
                [-0.25],
                [0.4375],
                [-0.4375],
                [0.0],
                [0.0],
                [0.25],
                [-0.25],
            ]
        ),
        in_shape=(1, 16),
        out_shape=(1, 1),
        tmp_path=tmp_path,
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)

    s = find_init(onnx_model, "w_scale")
    assert np.array_equal(s, np.asarray([[0.125], [0.0625]]))

    # Test shape compatibility
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16), in_shape=(96, 288), out_shape=(96, 16), tmp_path=tmp_path
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)  # Ensure it passes.

    onnx_path = _matmul_model(
        w=np.random.rand(577, 3), in_shape=(8, 557), out_shape=(8, 3), tmp_path=tmp_path
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)  # Ensure it passes.


def test_shape_rtn(tmp_path):
    # Test shape compatibility
    onnx_dataloader = [{"x": np.random.rand(96, 288)}]
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16).astype(np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
    )
    quantize_int4(
        onnx_path,
        "rtn",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.


def test_shape_awq(tmp_path):
    # Test shape compatibility
    onnx_dataloader = [{"x": np.random.rand(96, 288).astype(np.float32)}]
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16).astype(np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
    )
    quantize_int4(
        onnx_path,
        "awq_clip",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.
    quantize_int4(
        onnx_path,
        "awq_lite",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.
