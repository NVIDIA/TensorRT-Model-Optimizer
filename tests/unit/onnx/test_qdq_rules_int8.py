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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from _test_utils.onnx_quantization.lib_test_models import (
    build_conv_act_pool_model,
    build_r1a_model,
    build_resnet_block,
    build_resnet_block_with_downsample,
    export_as_onnx,
)

from modelopt.onnx.quantization.quantize import quantize
from modelopt.onnx.utils import save_onnx


def _assert_nodes_are_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable) and node.i(inp_idx).op != "Identity":
                assert node.i(inp_idx).op == "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                )
    return True


def _assert_nodes_are_not_quantized(nodes):
    for node in nodes:
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Variable) and inp.inputs:
                assert node.i(inp_idx).op != "DequantizeLinear", (
                    f"Input '{inp.name}' of node '{node.name}' is quantized but shouldn't be!"
                )
    return True


def test_bias_add_rule(tmp_path):
    # Copy the test model to the tmp_path
    model = build_r1a_model()
    onnx_path = os.path.join(tmp_path, "model.onnx")
    onnx.save(model, onnx_path)

    # Quantize the input model
    quantize(onnx_path)

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    #   Check that all Conv nodes are quantized
    conv_nodes = [n for n in graph.nodes if n.op == "Conv"]
    assert _assert_nodes_are_quantized(conv_nodes)

    #   Check that all other nodes are not quantized
    other_nodes = [
        n for n in graph.nodes if n.op not in ["Conv", "QuantizeLinear", "DequantizeLinear"]
    ]
    assert _assert_nodes_are_not_quantized(other_nodes)


def _check_resnet_residual_connection(onnx_path):
    # Quantize the input model
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    quantize(onnx_path)

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    #   Check that all Conv nodes are quantized
    conv_nodes = [n for n in graph.nodes if n.op == "Conv"]
    assert _assert_nodes_are_quantized(conv_nodes)

    #   Check that the left-side branch of Add contains a QDQ node
    #   In this case, this means that the inputs of Add should be DequantizeLinear and Conv.
    add_node = next(n for n in graph.nodes if n.op == "Add")
    add_input_ops = [inp.inputs[0].op for inp in add_node.inputs]
    assert np.isin(add_input_ops, ["Conv", "DequantizeLinear"]).all(), (
        f"Add node {add_node.name} was not quantized correctly!"
    )

    #   Check that all other nodes are not quantized
    other_nodes = [
        n for n in graph.nodes if n.op not in ["Conv", "Add", "QuantizeLinear", "DequantizeLinear"]
    ]
    assert _assert_nodes_are_not_quantized(other_nodes)


def test_resnet_residual_connections(tmp_path):
    model_torch, input_tensor = build_resnet_block()
    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)
    _check_resnet_residual_connection(onnx_path)


def test_resnet_residual_connection_with_downsample(tmp_path):
    model_torch, input_tensor = build_resnet_block_with_downsample()
    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)
    _check_resnet_residual_connection(onnx_path)


@pytest.mark.parametrize("include_reshape_node", [False, True])
def test_conv_act_pool_int8(tmp_path, include_reshape_node):
    onnx_model = build_conv_act_pool_model(include_reshape_node)
    onnx_path = os.path.join(tmp_path, f"conv_act_pool_model_{include_reshape_node}.onnx")
    save_onnx(onnx_model, onnx_path)

    quantize.quantize(onnx_path, quantize_mode="int8", high_precision_dtype="fp16")

    # Output model should be produced in the same tmp_path
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that Conv is quantized
    conv_nodes = [n for n in graph.nodes if n.op == "Conv"]
    assert _assert_nodes_are_quantized(conv_nodes)

    # Check that MaxPool is not quantized
    pool_nodes = [n for n in graph.nodes if n.op == "MaxPool"]
    assert _assert_nodes_are_not_quantized(pool_nodes)
