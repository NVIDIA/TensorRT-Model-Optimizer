# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.autocast.graphsanitizer import GraphSanitizer


def create_layernorm_model(input_shape, epsilon=1e-5, axis=-1, add_scale=True, add_bias=True):
    """Helper function to create an ONNX model with a decomposed LayerNorm pattern"""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    # Create nodes for LayerNorm pattern
    mean = helper.make_node("ReduceMean", ["X"], ["mean"], axes=[axis])
    sub1 = helper.make_node("Sub", ["X", "mean"], ["sub1"])
    pow_node = helper.make_node("Pow", ["sub1", "pow_const"], ["pow_out"])
    var_mean = helper.make_node("ReduceMean", ["pow_out"], ["var_mean"], axes=[axis])
    add_eps = helper.make_node("Add", ["var_mean", "epsilon"], ["add_eps"])
    sqrt = helper.make_node("Sqrt", ["add_eps"], ["sqrt_out"])
    div = helper.make_node("Div", ["sub1", "sqrt_out"], ["div_out"])

    nodes = [mean, sub1, pow_node, var_mean, add_eps, sqrt, div]

    # Add scale and bias if requested
    if add_scale:
        scale_mul = helper.make_node("Mul", ["div_out", "scale"], ["scale_out"])
        nodes.append(scale_mul)
    if add_bias:
        bias_input = "scale_out" if add_scale else "div_out"
        bias_add = helper.make_node("Add", [bias_input, "bias"], ["Y"])
        nodes.append(bias_add)
    else:
        nodes[-1].output[0] = "Y"

    # Create initializers
    initializers = [
        helper.make_tensor("pow_const", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("epsilon", TensorProto.FLOAT, [], [epsilon]),
    ]

    if add_scale:
        scale = np.random.random(input_shape[axis:]).astype(np.float32)
        initializers.append(numpy_helper.from_array(scale, name="scale"))
    if add_bias:
        bias = np.random.random(input_shape[axis:]).astype(np.float32)
        initializers.append(numpy_helper.from_array(bias, name="bias"))

    # Create graph and model
    graph = helper.make_graph(
        nodes=nodes, name="layernorm_test", inputs=[x], outputs=[y], initializer=initializers
    )

    model = helper.make_model(graph, producer_name="layernorm_test")
    model.opset_import[0].version = 13
    return model


@pytest.mark.parametrize("axis", [-1, -2, 1, 2])
def test_layernorm_with_axis(axis):
    """Test LayerNorm pattern replacement with different axes and scale/bias combinations."""
    input_shape = [2, 3, 4, 5]  # Multi-dimensional input
    model = create_layernorm_model(input_shape, axis=axis)

    sanitizer = GraphSanitizer(model)
    sanitizer.replace_layernorm_pattern()

    # Check that LayerNormalization node was created
    ln_nodes = [n for n in sanitizer.model.graph.node if n.op_type == "LayerNormalization"]
    assert len(ln_nodes) == 1

    ln_node = ln_nodes[0]
    assert ln_node.attribute[0].name == "axis"
    assert ln_node.attribute[1].name == "epsilon"


@pytest.mark.parametrize("add_bias", [True, False])
def test_layernorm_with_bias(add_bias):
    """Test LayerNorm pattern replacement with scale and bias"""
    model = create_layernorm_model([1, 32, 128], add_scale=True, add_bias=add_bias)
    sanitizer = GraphSanitizer(model)
    sanitizer.sanitize()

    # Verify the transformation
    ln_nodes = [n for n in sanitizer.model.graph.node if n.op_type == "LayerNormalization"]
    assert len(ln_nodes) == 1
    ln_node = ln_nodes[0]

    # Should have 2-3 inputs: input, scale [, bias]
    # If scale does not exist, it is created by the sanitizer
    expected_inputs = 3 if add_bias else 2
    assert len(ln_node.input) == expected_inputs
    assert ln_node.input[0] == "X"
    assert "scale" in ln_node.input[1]
    if add_bias:
        assert "bias" in ln_node.input[2]


def test_layernorm_different_epsilon():
    """Test LayerNorm pattern replacement with different epsilon values"""
    epsilon = 1e-8
    model = create_layernorm_model([1, 32, 128], epsilon=epsilon)
    sanitizer = GraphSanitizer(model)
    sanitizer.sanitize()

    ln_nodes = [n for n in sanitizer.model.graph.node if n.op_type == "LayerNormalization"]
    assert len(ln_nodes) == 1
    ln_node = ln_nodes[0]

    # Verify epsilon attribute
    epsilon_attr = next(attr for attr in ln_node.attribute if attr.name == "epsilon")
    assert np.isclose(epsilon_attr.f, epsilon)


def test_no_layernorm_pattern():
    """Test that non-LayerNorm patterns are not modified even with similar ops"""
    # Create a model with similar ops to LayerNorm but in different order/pattern
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32, 128])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 32, 128])

    # Constants
    epsilon = helper.make_tensor("epsilon", TensorProto.FLOAT, [1], [1e-5])
    pow_const = helper.make_tensor("pow_const", TensorProto.FLOAT, [1], [2.0])

    # Create nodes in a different order than LayerNorm
    pow_node = helper.make_node("Pow", ["X", "pow_const"], ["pow_out"])
    mean_node = helper.make_node("ReduceMean", ["pow_out"], ["mean"], axes=[-1], keepdims=1)
    add_node = helper.make_node("Add", ["mean", "epsilon"], ["add_out"])
    sqrt_node = helper.make_node("Sqrt", ["add_out"], ["sqrt_out"])
    div_node = helper.make_node("Div", ["X", "sqrt_out"], ["Y"])

    graph = helper.make_graph(
        nodes=[pow_node, mean_node, add_node, sqrt_node, div_node],
        name="not_layernorm",
        inputs=[x],
        outputs=[y],
        initializer=[epsilon, pow_const],
    )

    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    sanitizer = GraphSanitizer(model)
    sanitizer.sanitize()

    # Verify no LayerNorm transformation occurred despite similar ops
    assert len(sanitizer.model.graph.node) == 5
    assert not any(node.op_type == "LayerNormalization" for node in sanitizer.model.graph.node)
    assert [node.op_type for node in sanitizer.model.graph.node] == [
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
    ]


def test_invalid_layernorm_pattern():
    """Test that invalid LayerNorm-like patterns are not transformed"""
    # Create a pattern that's similar to LayerNorm but with wrong power value
    model = create_layernorm_model([1, 32, 128])
    # Modify the power constant to 3 instead of 2
    for initializer in model.graph.initializer:
        if initializer.name == "pow_const":
            initializer.float_data[0] = 3.0

    sanitizer = GraphSanitizer(model)
    sanitizer.sanitize()

    # Verify no LayerNorm transformation occurred
    assert not any(node.op_type == "LayerNormalization" for node in sanitizer.model.graph.node)
