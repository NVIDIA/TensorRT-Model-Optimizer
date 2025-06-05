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

from pathlib import Path

import numpy as np
import onnx
import pytest

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast import convert
from modelopt.onnx.autocast.logging_config import configure_logging

configure_logging("DEBUG")


@pytest.fixture
def simple_model():
    # Create a model with multiple nodes and initializers
    input_shape = [1, 5]
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_shape)
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])

    # Create initializers
    gemm_weight = np.random.randn(5, 3).astype(np.float32)
    gemm_bias = np.random.randn(3).astype(np.float32)
    add_const = np.random.randn(1, 3).astype(np.float32)

    gemm_w_init = onnx.numpy_helper.from_array(gemm_weight, name="gemm_weight")
    gemm_b_init = onnx.numpy_helper.from_array(gemm_bias, name="gemm_bias")
    add_init = onnx.numpy_helper.from_array(add_const, name="add_const")

    # Create nodes
    gemm_node = onnx.helper.make_node(
        "Gemm", inputs=["X", "gemm_weight", "gemm_bias"], outputs=["gemm_out"], name="gemm_node"
    )

    add_node = onnx.helper.make_node(
        "Add", inputs=["gemm_out", "add_const"], outputs=["add_out"], name="add_node"
    )

    relu_node = onnx.helper.make_node("Relu", inputs=["add_out"], outputs=["Y"], name="relu_node")

    graph = onnx.helper.make_graph(
        [gemm_node, add_node, relu_node],
        "test_complex_graph",
        [x],
        [y],
        [gemm_w_init, gemm_b_init, add_init],
    )

    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 20
    model.ir_version = 10
    return model


@pytest.fixture
def temp_model_path(tmp_path, simple_model):
    model_path = tmp_path / "model.onnx"
    onnx.save(simple_model, str(model_path))
    return str(model_path)


@pytest.fixture
def temp_output_path(tmp_path):
    return str(tmp_path / "output_model.onnx")


def test_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        convert(onnx_path="nonexistent.onnx")


def test_setup_mappings(simple_model):
    simple_model = onnx_utils.infer_shapes(simple_model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(simple_model)

    # Test value_info_map
    assert len(value_info_map) == 4  # X, Y, gemm_out, add_out
    assert all(name in value_info_map for name in ["X", "Y", "gemm_out", "add_out"])
    assert value_info_map["X"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert value_info_map["Y"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    # Test initializer_map
    assert len(initializer_map) == 3  # gemm_weight, gemm_bias, add_const
    assert all(name in initializer_map for name in ["gemm_weight", "gemm_bias", "add_const"])
    assert initializer_map["gemm_weight"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["gemm_bias"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["add_const"].data_type == onnx.TensorProto.FLOAT

    # Test node_to_init_map
    assert len(node_to_init_map) == 3  # gemm_node, add_node, relu_node
    assert "gemm_node" in node_to_init_map
    assert "add_node" in node_to_init_map
    assert "relu_node" in node_to_init_map
    assert len(node_to_init_map["gemm_node"]) == 2  # gemm_weight, gemm_bias
    assert len(node_to_init_map["add_node"]) == 1  # add_const
    assert len(node_to_init_map["relu_node"]) == 0  # no initializers


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_convert_simple_model(temp_model_path, temp_output_path, keep_io_types):
    # Convert the model
    converted_model = convert(onnx_path=temp_model_path, keep_io_types=keep_io_types)

    # Save the converted model
    onnx.save(converted_model, temp_output_path)

    assert isinstance(converted_model, onnx.ModelProto)
    assert Path(temp_output_path).exists()

    # Load and verify the saved model
    loaded_model = onnx.load(temp_output_path)
    if keep_io_types:
        assert len(loaded_model.graph.node) == 5
        assert loaded_model.graph.node[0].op_type == "Cast"
        assert loaded_model.graph.node[1].op_type == "Gemm"
        assert loaded_model.graph.node[2].op_type == "Add"
        assert loaded_model.graph.node[3].op_type == "Relu"
        assert loaded_model.graph.node[4].op_type == "Cast"
    else:
        assert len(loaded_model.graph.node) == 3
        assert loaded_model.graph.node[0].op_type == "Gemm"
        assert loaded_model.graph.node[1].op_type == "Add"
        assert loaded_model.graph.node[2].op_type == "Relu"

    # Verify input/output types
    expected_io_type = onnx.TensorProto.FLOAT if keep_io_types else onnx.TensorProto.FLOAT16
    assert loaded_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert loaded_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    onnx.checker.check_model(loaded_model)
