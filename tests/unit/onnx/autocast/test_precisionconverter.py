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
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import configure_logging
from modelopt.onnx.autocast.precisionconverter import PrecisionConverter

configure_logging("DEBUG")


def low_precision_onnx_type(low_precision_type_str):
    return TensorProto.FLOAT16 if low_precision_type_str == "fp16" else TensorProto.BFLOAT16


####################################################################################################
# Testing with a basic GEMM->Add->Relu graph
####################################################################################################
@pytest.fixture
def simple_model():
    # Create a simple model with a GEMM->Add->Relu chain
    input_shape = [1, 5]
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    gemm_init_numpy = np.random.randn(5, 3).astype(np.float32)
    add_init_numpy = np.random.randn(1, 3).astype(np.float32)
    gemm_init = numpy_helper.from_array(gemm_init_numpy, name="gemm_init")
    add_init = numpy_helper.from_array(add_init_numpy, name="add_init")
    gemm_node = helper.make_node("MatMul", ["X", "gemm_init"], ["gemm_output"], name="gemm")
    add_node = helper.make_node("Add", ["gemm_output", "add_init"], ["add_output"], name="add")
    relu_node = helper.make_node("Relu", ["add_output"], ["Y"], name="relu")
    graph = helper.make_graph(
        [gemm_node, add_node, relu_node], "model_base", [x], [y], [gemm_init, add_init]
    )
    model = helper.make_model(graph, producer_name="model_base")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


def test_graph_converter_init(simple_model):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model, value_info_map, initializer_map, node_to_init_map, keep_io_types=True
    )
    assert converter.model == model
    assert converter.value_info_map == value_info_map
    assert converter.initializer_map == initializer_map
    assert converter.keep_io_types


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_simple_convert(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert add node to fp16, keep mul in fp32
    converted_model = converter.convert(
        high_precision_nodes=["gemm", "add"], low_precision_nodes=["relu"]
    )

    # Verify input type changed to fp16
    expected_io_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert converted_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    # Verify cast nodes were added in the expected locations
    assert len(converted_model.graph.node) == 5
    if keep_io_types:
        assert converted_model.graph.node[0].op_type == "MatMul"
        assert converted_model.graph.node[1].op_type == "Add"
        assert converted_model.graph.node[2].op_type == "Cast"
        assert converted_model.graph.node[3].op_type == "Relu"
        assert converted_model.graph.node[4].op_type == "Cast"
    else:
        assert converted_model.graph.node[0].op_type == "Cast"
        assert converted_model.graph.node[1].op_type == "MatMul"
        assert converted_model.graph.node[2].op_type == "Add"
        assert converted_model.graph.node[3].op_type == "Cast"
        assert converted_model.graph.node[4].op_type == "Relu"

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("low_precision_type", ["int8", "fp8", "fp4", "fp32", "typo"])
def test_unsupported_precision_type(simple_model, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    with pytest.raises(ValueError, match=f"Unsupported precision type: {low_precision_type}"):
        PrecisionConverter(
            model,
            value_info_map,
            initializer_map,
            node_to_init_map,
            keep_io_types=False,
            low_precision_type=low_precision_type,
        )


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_no_disabled_nodes(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert all nodes to fp16
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=["gemm", "add", "relu"]
    )

    # Verify input type changed to fp16
    expected_io_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_io_type
    assert converted_model.graph.output[0].type.tensor_type.elem_type == expected_io_type

    # Verify cast nodes were added only for keep_io_types=True
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    expected_cast_nodes = 2 if keep_io_types else 0
    assert len(cast_nodes) == expected_cast_nodes

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_get_tensors_to_cast(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Test when relu node is in low precision
    cast_down, cast_up = converter._get_tensors_to_cast(["relu"])
    assert "add_output" in cast_down  # Input to relu should be cast down
    assert "Y" in cast_up  # Output of relu should be cast up
    if not keep_io_types:
        assert (
            "X" in cast_up
        )  # Input to gemm should be cast up, because network input are converted to FP16

    # Test when add node is in low precision
    cast_down, cast_up = converter._get_tensors_to_cast(["add"])
    assert "gemm_output" in cast_down  # Input to add should be cast down
    assert "add_init" not in cast_down  # Initializer should not be in cast list
    assert "add_output" in cast_up  # Output of add should be cast up


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_keep_io_names(simple_model, keep_io_types, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = simple_model
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Convert all nodes to low precision
    converted_model = converter.convert(
        high_precision_nodes=["gemm", "add"], low_precision_nodes=["relu"]
    )

    # Verify that the input and output names are the same as in the original model
    for i in range(len(model.graph.input)):
        assert converted_model.graph.input[i].name == model.graph.input[i].name
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


####################################################################################################
# Graph with multiple consumers for a single input and multiple consumers for a single initializer
####################################################################################################
@pytest.fixture
def model_with_multiple_consumers():
    # Create a model where a single input and a single initializer are consumed by multiple nodes
    input_shape = [1, 5]
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 3])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 3])
    y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT, [1, 3])

    # Create initializers
    gemm1_init_numpy = np.random.randn(5, 3).astype(np.float32)
    add_init_numpy = np.random.randn(1, 3).astype(np.float32)
    gemm2_init_numpy = np.random.randn(5, 3).astype(np.float32)
    gemm1_init = numpy_helper.from_array(gemm1_init_numpy, name="gemm1_init")
    add_init = numpy_helper.from_array(add_init_numpy, name="add_init")
    gemm2_init = numpy_helper.from_array(gemm2_init_numpy, name="gemm2_init")

    # Create nodes where X is used by both gemm1 and gemm2
    gemm1_node = helper.make_node("MatMul", ["X", "gemm1_init"], ["gemm1_output"], name="gemm1")
    gemm2_node = helper.make_node("MatMul", ["X", "gemm2_init"], ["gemm2_output"], name="gemm2")
    add1_node = helper.make_node("Add", ["gemm1_output", "add_init"], ["Y1"], name="add1")
    add2_node = helper.make_node("Add", ["gemm2_output", "add_init"], ["Y2"], name="add2")
    add3_node = helper.make_node("Add", ["gemm1_output", "add_init"], ["Y3"], name="add3")

    graph = helper.make_graph(
        [gemm1_node, gemm2_node, add1_node, add2_node, add3_node],
        "model_multi_consumer",
        [x],
        [y1, y2, y3],
        [gemm1_init, add_init, gemm2_init],
    )
    model = helper.make_model(graph, producer_name="model_multi_consumer")
    model.opset_import[0].version = 20
    model.ir_version = 10
    onnx.checker.check_model(model)

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_with_multiple_consumers(
    model_with_multiple_consumers, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Only gemm1 and add1 are converted to fp32, gemm2 and add2 are fp16
    converted_model = converter.convert(
        high_precision_nodes=["gemm1", "add1"], low_precision_nodes=["gemm2", "add2"]
    )

    # Verify input type changed to low precision according to keep_io_types
    expected_input_type = (
        TensorProto.FLOAT if keep_io_types else low_precision_onnx_type(low_precision_type)
    )
    assert converted_model.graph.input[0].type.tensor_type.elem_type == expected_input_type

    # Verify cast nodes were added
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) > 0

    # Verify that the model is valid
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_get_tensors_to_cast_multiple_consumers(
    model_with_multiple_consumers, keep_io_types, low_precision_type
):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
    )

    # Test when gemm2 and add1 nodes are in low precision
    cast_down, cast_up = converter._get_tensors_to_cast(["gemm2", "add1"])
    assert "X" in cast_down  # Input to gemm2 should be cast down
    assert "gemm2_output" in cast_up  # Output of gemm2 should be cast up
    assert "Y1" in cast_up  # Output of add1 should be cast up

    # Test when all nodes except gemm1 are in low precision
    cast_down, cast_up = converter._get_tensors_to_cast(["gemm2", "add1", "add2"])
    assert "gemm1_output" in cast_down  # Input to gemm2 should be cast down
    assert "Y1" in cast_up  # Output of add1 should be cast up
    assert "Y2" in cast_up  # Output of add2 should be cast up


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_convert_initializers(model_with_multiple_consumers, low_precision_type):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )

    # Test successful cast, add1 and add2 share add_init and operate in different precisions
    add1_node = next(n for n in converter.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter._convert_initializers(
        low_precision_nodes=["add1"], high_precision_nodes=["add2", "add3"]
    )

    assert add1_node.input[1] != add2_node.input[1]  # add1 and add2 have different initializers
    assert add2_node.input[1] == add3_node.input[1]  # add2 and add3 share the same initializer
    init_names = [init.name for init in converter.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" in init_names

    # Test successful cast, add1, add2, add3 share add_init.and all operate in FP16
    converter2 = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )
    add1_node = next(n for n in converter2.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter2.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter2.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter2._convert_initializers(
        low_precision_nodes=["add1", "add2", "add3"], high_precision_nodes=[]
    )

    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]
    init_names = [init.name for init in converter2.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" not in init_names

    # Test successful cast, add1 and add2 share add_init and both operate in FP16, 'add3' is kept in FP32
    converter3 = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type=low_precision_type,
    )
    add1_node = next(n for n in converter3.model.graph.node if n.name == "add1")
    add2_node = next(n for n in converter3.model.graph.node if n.name == "add2")
    add3_node = next(n for n in converter3.model.graph.node if n.name == "add3")
    assert add1_node.input[1] == add2_node.input[1] == add3_node.input[1]

    converter3._convert_initializers(
        low_precision_nodes=["add1", "add2"], high_precision_nodes=["add3"]
    )

    assert (
        add1_node.input[1] == add2_node.input[1]
    )  # after cast, add1 and add2 have the same initializer
    assert add1_node.input[1] != add3_node.input[1]  # add1 and add3 have different initializers
    init_names = [init.name for init in converter3.model.graph.initializer]
    assert len(init_names) == len(set(init_names))  # no duplicate initializers
    assert "add_init" in init_names
    assert f"add_init_{low_precision_type}" in init_names


def test_clamping_fp16_initializers_out_of_range(model_with_multiple_consumers):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers

    # Initializer is out of FP16 range, node is converted to FP16
    add_init_out_of_range = np.array([[-70000.0, 70000.0]], dtype=np.float32)
    add_init = numpy_helper.from_array(add_init_out_of_range, name="add_init")
    model.graph.initializer[1].CopyFrom(add_init)

    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter._convert_initializers(low_precision_nodes=["add1", "add2"], high_precision_nodes=[])

    # Verify initializer is clamped
    add_init_converted = [
        init for init in converter.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = numpy_helper.to_array(add_init_converted[0])
    assert add_init_converted_array.dtype == np.float16
    assert add_init_converted_array.shape == (1, 2)
    assert add_init_converted_array[0, 0] == np.finfo(np.float16).min
    assert add_init_converted_array[0, 1] == np.finfo(np.float16).max

    # Initializer is out of FP16 range, node is kept in FP32
    converter2 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter2._convert_initializers(low_precision_nodes=[], high_precision_nodes=["add1", "add2"])

    # Verify initializer is not clamped
    add_init_converted = [
        init for init in converter2.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = numpy_helper.to_array(add_init_converted[0])
    assert add_init_converted_array.dtype == np.float32
    assert add_init_converted_array.shape == (1, 2)
    assert np.all(add_init_converted_array == add_init_out_of_range)

    # Initializer is out of FP16 range, one consumer is converted to FP16, the other is kept in FP32
    converter3 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter3._convert_initializers(low_precision_nodes=["add1"], high_precision_nodes=["add2"])

    # Verify initializer is duplicated, and the FP16 copy is clamped
    add_init_fp16 = [
        init for init in converter3.model.graph.initializer if init.name == "add_init_fp16"
    ]
    add_init_fp32 = [init for init in converter3.model.graph.initializer if init.name == "add_init"]
    assert len(add_init_fp16) == 1
    assert len(add_init_fp32) == 1
    add_init_fp16_array = numpy_helper.to_array(add_init_fp16[0])
    add_init_fp32_array = numpy_helper.to_array(add_init_fp32[0])
    assert add_init_fp16_array.dtype == np.float16
    assert add_init_fp32_array.dtype == np.float32
    assert np.all(
        add_init_fp16_array
        == np.asarray([np.finfo(np.float16).min, np.finfo(np.float16).max], dtype=np.float16)
    )
    assert np.all(add_init_fp32_array == add_init_out_of_range)


def test_bf16_no_clamping_initializers_out_of_range(model_with_multiple_consumers):
    model, value_info_map, initializer_map, node_to_init_map = model_with_multiple_consumers

    # Initializer is out of FP16 range, but that does not affect BF16 conversion
    add_init_out_of_range = np.array([[2 << 16, 2 << 17]], dtype=np.float32)
    add_init = numpy_helper.from_array(add_init_out_of_range, name="add_init")
    model.graph.initializer[1].CopyFrom(add_init)

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        low_precision_type="bf16",
    )
    converter._convert_initializers(low_precision_nodes=["add1", "add2"], high_precision_nodes=[])

    # Verify initializer is not clamped
    add_init_converted = [
        init for init in converter.model.graph.initializer if init.name == "add_init"
    ]
    assert len(add_init_converted) == 1
    add_init_converted_array = onnx_utils.read_f16_tensor_as_fp32(add_init_converted[0])
    print(add_init_out_of_range)
    print(add_init_converted_array)
    assert add_init_converted_array.shape == (1, 2)
    assert np.all(add_init_converted_array == add_init_out_of_range)


####################################################################################################
# Testing with dynamic shapes, since shape_inference invoked in PrecisionConverter
####################################################################################################
@pytest.fixture
def model_with_dynamic_shapes():
    # Create inputs with different dynamic shapes
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])  # Dynamic batch size
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, None])  # Dynamic second dimension
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [8, 4])  # Output shape

    # Create initializers
    weight = np.random.randn(4, 8).astype(np.float32)
    bias = np.random.randn(8).astype(np.float32)
    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")

    # Create nodes with shape-changing operations
    matmul_node = helper.make_node("MatMul", ["X", "weight"], ["matmul_out"], name="matmul")
    transpose_node = helper.make_node("Transpose", ["Y"], ["transpose_out"], name="transpose")
    concat_node = helper.make_node(
        "Concat", ["matmul_out", "transpose_out"], ["concat_out"], name="concat", axis=0
    )
    size_y = helper.make_node("Size", ["concat_out"], ["total_size"], name="size")
    const_4 = numpy_helper.from_array(np.array([4], dtype=np.int64), name="const_4")
    first_dim = helper.make_node("Div", ["total_size", "const_4"], ["first_dim"], name="div")
    concat_dims_node = helper.make_node(
        "Concat", ["first_dim", "const_4"], ["final_shape"], name="concat", axis=0
    )
    reshape_node = helper.make_node("Reshape", ["concat_out", "final_shape"], ["Z"], name="reshape")

    # Create graph and model
    graph = helper.make_graph(
        [
            matmul_node,
            transpose_node,
            concat_node,
            size_y,
            first_dim,
            concat_dims_node,
            reshape_node,
        ],
        "model_dynamic",
        [x, y],
        [z],
        [weight_init, bias_init, const_4],
    )
    model = helper.make_model(graph, producer_name="model_dynamic")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    return model, value_info_map, initializer_map, node_to_init_map


def test_dynamic_model_conversion(model_with_dynamic_shapes):
    model, value_info_map, initializer_map, node_to_init_map = model_with_dynamic_shapes

    # Test mixed precision conversion
    converter2 = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    high_precision_nodes = ["matmul"]
    low_precision_nodes = ["transpose", "concat", "size", "div", "concat_dims", "reshape"]

    converted_model = converter2.convert(high_precision_nodes, low_precision_nodes)
    # Verify model is valid
    onnx.checker.check_model(converted_model)


####################################################################################################
# Cast cleanup logic
####################################################################################################
def test_cast_output_pattern():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    node1 = helper.make_node("Add", ["X", "weight"], ["Y1"], name="node1")
    node2 = helper.make_node("Reciprocal", ["Y1"], ["Y2"], name="node2")

    graph = helper.make_graph(
        [node1, node2],
        "model_output_cast",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_double_cast")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)

    # Setting all nodes to FP16 means that the final graph should have no cast nodes
    converted_model = converter.convert(
        high_precision_nodes=[], low_precision_nodes=["node1", "node2"]
    )

    # Verify all cast nodes were removed
    cast_nodes = [n for n in converted_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 0
    assert len(converted_model.graph.node) == 2
    # Verify that the output names are the same as in the original model
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


def test_cast_output_pattern_mixed_precision():
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [3, 4])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [3, 4])
    y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    node0 = helper.make_node("Add", ["X1", "weight"], ["Y0"], name="node0")
    node1 = helper.make_node("Mul", ["X2", "Y0"], ["Y1"], name="node1")
    node2 = helper.make_node("Div", ["X2", "Y0"], ["Y2"], name="node2")

    graph = helper.make_graph(
        [node0, node1, node2],
        "model_output_cast",
        [x1, x2],
        [y0, y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_double_cast")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)

    # Network output Y0 has two consumers, one is FP16 and the other is FP32
    converted_model = converter.convert(
        high_precision_nodes=["node0", "node2"], low_precision_nodes=["node1"]
    )

    # Verify that the output names are the same as in the original model
    for i in range(len(model.graph.output)):
        assert converted_model.graph.output[i].name == model.graph.output[i].name


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_chain_of_casts_pattern(keep_io_types):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")
    cast1_node = helper.make_node("Cast", ["X"], ["cast1_out"], name="cast1", to=TensorProto.FLOAT)
    cast2_node = helper.make_node(
        "Cast", ["cast1_out"], ["cast2_out"], name="cast2", to=TensorProto.FLOAT
    )
    cast3_node = helper.make_node(
        "Cast", ["cast2_out"], ["cast3_out"], name="cast3", to=TensorProto.FLOAT
    )
    cast4_node = helper.make_node(
        "Cast", ["cast3_out"], ["cast4_out"], name="cast4", to=TensorProto.FLOAT
    )
    add_node = helper.make_node("Add", ["cast4_out", "weight"], ["add_out"], name="add")
    cast5_node = helper.make_node(
        "Cast", ["add_out"], ["cast5_out"], name="cast5", to=TensorProto.FLOAT16
    )
    cast6_node = helper.make_node(
        "Cast", ["cast5_out"], ["cast6_out"], name="cast6", to=TensorProto.FLOAT16
    )
    cast7_node = helper.make_node(
        "Cast", ["cast6_out"], ["cast7_out"], name="cast7", to=TensorProto.FLOAT16
    )
    cast8_node = helper.make_node(
        "Cast", ["cast7_out"], ["Y"], name="cast8", to=TensorProto.FLOAT16
    )

    graph = helper.make_graph(
        [
            cast1_node,
            cast2_node,
            cast3_node,
            cast4_node,
            add_node,
            cast5_node,
            cast6_node,
            cast7_node,
            cast8_node,
        ],
        "model_cast_chain",
        [x],
        [y],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_cast_chain")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model, value_info_map, initializer_map, node_to_init_map, keep_io_types=keep_io_types
    )
    converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    # Verify cast chain was removed
    cast_nodes = [n for n in converter.model.graph.node if n.op_type == "Cast"]
    expected_cast_nodes = 0 if keep_io_types else 2
    assert len(cast_nodes) == expected_cast_nodes


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_existing_low_precision_output(low_precision_type):
    # Create a simple model with FP16 output
    x = helper.make_tensor_value_info("X", low_precision_onnx_type(low_precision_type), [3, 4])
    y = helper.make_tensor_value_info("Y", low_precision_onnx_type(low_precision_type), [3, 4])
    add_node = helper.make_node("Add", ["X", "X"], ["Y"], name="add")
    graph = helper.make_graph([add_node], "model_add", [x], [y], [])
    model = helper.make_model(graph, producer_name="model_add")
    model.opset_import[0].version = 20
    model.ir_version = 10

    model = onnx_utils.infer_shapes(model)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )
    converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    assert len(converter.model.graph.node) == 3
    assert converter.model.graph.node[0].op_type == "Cast"
    assert converter.model.graph.node[1].op_type == "Add"
    assert converter.model.graph.node[2].op_type == "Cast"

    # check that the I/O remains in low precision
    assert converter.model.graph.input[0].type.tensor_type.elem_type == low_precision_onnx_type(
        low_precision_type
    )
    assert converter.model.graph.output[0].type.tensor_type.elem_type == low_precision_onnx_type(
        low_precision_type
    )


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_output_cast_output_pattern(low_precision_type):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", low_precision_onnx_type(low_precision_type), [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    cast_to = low_precision_onnx_type(low_precision_type)
    add_node = helper.make_node("Add", ["X", "weight"], ["Y1"], name="add")
    cast_node = helper.make_node("Cast", ["Y1"], ["Y2"], name="cast", to=cast_to)

    graph = helper.make_graph(
        [add_node, cast_node],
        "model_output_cast_output",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_output_cast_output")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )

    # Setting nodes precision to match I/O type means that the final graph should have no cast nodes
    converted_model = converter.convert(high_precision_nodes=["add"], low_precision_nodes=[])

    assert len(converted_model.graph.node) == 2
    assert converted_model.graph.node[0].op_type == "Add"
    assert converted_model.graph.node[1].op_type == "Cast"

    y1_out = next(y for y in converted_model.graph.output if y.name == "Y1")
    y2_out = next(y for y in converted_model.graph.output if y.name == "Y2")
    assert y1_out.type.tensor_type.elem_type == TensorProto.FLOAT
    assert y2_out.type.tensor_type.elem_type == low_precision_onnx_type(low_precision_type)


@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_cast_output_keep_io_types_pattern(low_precision_type):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [3, 4])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [3, 4])

    init_weight = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="weight")

    add1_node = helper.make_node("Add", ["X", "weight"], ["Y1"], name="add1")
    add2_node = helper.make_node("Add", ["Y1", "weight"], ["Y2"], name="add2")
    graph = helper.make_graph(
        [add1_node, add2_node],
        "model_cast_output_keep_io_types",
        [x],
        [y1, y2],
        [init_weight],
    )
    model = helper.make_model(graph, producer_name="model_cast_output_keep_io_types")
    model.opset_import[0].version = 20
    model.ir_version = 10
    model = onnx_utils.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=True,
        low_precision_type=low_precision_type,
    )
    converter.convert(high_precision_nodes=[], low_precision_nodes=["add1", "add2"])

    # Outputs should be FP32
    assert converter.model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converter.model.graph.output[1].type.tensor_type.elem_type == TensorProto.FLOAT


def test_unsupported_op_types_model():
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    roi = helper.make_tensor_value_info("roi", TensorProto.FLOAT, [3, 4])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [3, 4, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [3, 4])
    celu_out = helper.make_tensor_value_info("celu_out", TensorProto.FLOAT, [3, 4])
    resize_out = helper.make_tensor_value_info("resize_out", TensorProto.FLOAT, [3, 4])
    nms_out = helper.make_tensor_value_info("nms_out", TensorProto.INT64, [-1, 3])

    node1 = helper.make_node("Celu", ["X"], ["celu_out"], name="celu")
    node2 = helper.make_node("Resize", ["X", "roi", "scales"], ["resize_out"], name="resize")
    node3 = helper.make_node("NonMaxSuppression", ["boxes", "scores"], ["nms_out"], name="nms")
    graph = helper.make_graph(
        [node1, node2, node3],
        "model_celu",
        [x, roi, scales, boxes, scores],
        [celu_out, resize_out, nms_out],
        [],
    )
    model = helper.make_model(graph, producer_name="model_celu")
    model = onnx.shape_inference.infer_shapes(model)

    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)
    converter = PrecisionConverter(model, value_info_map, initializer_map, node_to_init_map)
    converter.convert(high_precision_nodes=[], low_precision_nodes=["celu", "resize", "nms"])
    onnx.checker.check_model(converter.model)
