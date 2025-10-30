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

import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from _test_utils.onnx.quantization.lib_test_models import (
    SimpleMLP,
    build_convtranspose_conv_residual_model,
    export_as_onnx,
)

import modelopt.onnx.quantization as moq
from modelopt.onnx.utils import save_onnx


def assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable):
                assert node.i(inp_idx).op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True


@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16", "bf16"])
def test_int8(tmp_path, high_precision_dtype):
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)
    moq.quantize(onnx_path, quantize_mode="int8", high_precision_dtype=high_precision_dtype)

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that all MatMul nodes are quantized
    mm_nodes = [n for n in graph.nodes if n.op == "MatMul"]
    assert assert_nodes_are_quantized(mm_nodes)


def test_convtranspose_conv_residual_int8(tmp_path):
    onnx_model = build_convtranspose_conv_residual_model()
    onnx_path = os.path.join(tmp_path, "convtranspose_conv_residual_model.onnx")
    save_onnx(onnx_model, onnx_path)

    moq.quantize(onnx_path, quantize_mode="int8", high_precision_dtype="fp16")

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that Conv and ConvTransposed are quantized
    conv_nodes = [n for n in graph.nodes if "Conv" in n.op]
    assert assert_nodes_are_quantized(conv_nodes)

    # Check that only 1 input of Add is quantized
    add_nodes = [n for n in graph.nodes if n.op == "Add"]
    for node in add_nodes:
        quantized_inputs = [inp for inp in node.inputs if inp.inputs[0].op == "DequantizeLinear"]
        assert len(quantized_inputs) == 1, (
            f"More than one input of {node.name} is being quantized, but only one should be quantized!"
        )
