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

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.quantization.qdq_utils import _cast_fp4, _cast_fp8, quantize_weights_to_int4


def create_test_model_with_dq_reshape_transpose_matmul(constant_scale: bool = False):
    """Create a test ONNX model with DequantizeLinear -> Reshape -> Transpose -> MatMul pattern.
    If constant_scale is True, the scale is a Constant node instead of an initializer."""
    # Create weight tensor (4x8 matrix scaled by 32 blocks)
    weight_data = np.random.randint(-8, 8, size=(32, 8), dtype=np.int8)
    weight_tensor = numpy_helper.from_array(weight_data, "weight")

    # Create scale tensor for block quantization (block_size = 32)
    scale_data = np.random.uniform(0.1, 1.0, size=(32, 1)).astype(np.float32)
    scale_tensor = numpy_helper.from_array(scale_data, "scale")

    # Create reshape shape tensor
    reshape_shape = np.array([16, 16], dtype=np.int64)
    reshape_shape_tensor = numpy_helper.from_array(reshape_shape, "reshape_shape")

    # Create input tensor for MatMul
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])

    if constant_scale:
        scale_constant = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Constant_output_0"],
            value=numpy_helper.from_array(scale_data),
            name="scale_constant",
        )

    # Create nodes
    dq_inputs = ["weight", "Constant_output_0"] if constant_scale else ["weight", "scale"]
    dq_node = helper.make_node(
        "DequantizeLinear", inputs=dq_inputs, outputs=["dq_output"], name="weight_dq"
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["dq_output", "reshape_shape"],
        outputs=["reshape_output"],
        name="weight_reshape",
    )

    transpose_node = helper.make_node(
        "Transpose",
        inputs=["reshape_output"],
        outputs=["transpose_output"],
        perm=[1, 0],
        name="weight_transpose",
    )

    matmul_node = helper.make_node(
        "MatMul", inputs=["input", "transpose_output"], outputs=["output"], name="matmul"
    )

    # Create value info for intermediate tensors
    reshape_output_info = helper.make_tensor_value_info(
        "reshape_output", TensorProto.FLOAT, [16, 16]
    )

    # Create graph
    nodes = [dq_node, reshape_node, transpose_node, matmul_node]
    if constant_scale:
        nodes.append(scale_constant)
    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 16])],
        initializer=[weight_tensor, scale_tensor, reshape_shape_tensor],
        value_info=[reshape_output_info],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_cast_nodes():
    """Create a test model with various Cast nodes to test float32->float16 conversion."""
    # Create a simple model with Cast nodes
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])

    # Cast to float32 (should be converted to float16)
    cast_fp32 = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_fp32_output"],
        to=TensorProto.FLOAT,
        name="regular_cast",
    )

    # Cast in normalization (should be preserved)
    cast_norm = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["cast_norm_output"],
        to=TensorProto.FLOAT,
        name="layer_norm/Cast",
    )

    # Output cast (should be preserved)
    cast_output = helper.make_node(
        "Cast", inputs=["input"], outputs=["final_output"], to=TensorProto.FLOAT, name="/Cast"
    )

    graph = helper.make_graph(
        nodes=[cast_fp32, cast_norm, cast_output],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [4])],
        initializer=[],
    )

    model = helper.make_model(graph)
    return model


def create_test_model_with_proj_nodes():
    """Create a test model with projection nodes to test bias and scale casting."""
    # Create bias tensor
    bias_data = np.random.uniform(-1.0, 1.0, size=(16,)).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias_data, "proj_bias")

    # Create scale tensor for quantization
    scale_data = np.random.uniform(0.1, 1.0, size=(1,)).astype(np.float32)
    scale_tensor = numpy_helper.from_array(scale_data, "quant_scale")

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 16])

    # Add node (projection bias)
    add_node = helper.make_node(
        "Add", inputs=["input", "proj_bias"], outputs=["add_output"], name="o_proj/Add"
    )

    # Mul node (quantization scale)
    mul_node = helper.make_node(
        "Mul",
        inputs=["add_output", "quant_scale"],
        outputs=["output"],
        name="o_proj/input_quantizer/Mul",
    )

    graph = helper.make_graph(
        nodes=[add_node, mul_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 16])],
        initializer=[bias_tensor, scale_tensor],
    )

    model = helper.make_model(graph)
    return model


class TestQuantizeWeightsToInt4:
    """Test suite for quantize_weights_to_int4 function."""

    def test_basic_quantization_with_reshape_transpose(self):
        """Test basic INT4 quantization with Reshape and Transpose removal."""
        model = create_test_model_with_dq_reshape_transpose_matmul()

        # Run quantization
        quantized_model = quantize_weights_to_int4(model)

        # Verify weight is converted to INT4
        weight_tensor = next(
            init for init in quantized_model.graph.initializer if init.name == "weight"
        )
        assert weight_tensor.data_type == TensorProto.INT4

        # Verify Reshape and Transpose nodes are removed
        node_types = [node.op_type for node in quantized_model.graph.node]
        assert "Reshape" not in node_types
        assert "Transpose" not in node_types

        # Verify MatMul input is connected directly to DequantizeLinear output
        matmul_node = next(node for node in quantized_model.graph.node if node.op_type == "MatMul")
        dq_node = next(
            node for node in quantized_model.graph.node if node.op_type == "DequantizeLinear"
        )
        assert matmul_node.input[1] == dq_node.output[0]

    def test_quantization_with_constant_scale(self):
        """Test quantization when scale comes from a Constant node."""
        model = create_test_model_with_dq_reshape_transpose_matmul(constant_scale=True)

        # Run quantization
        quantized_model = quantize_weights_to_int4(model)

        # Verify Constant node is removed
        constant_nodes = [node for node in quantized_model.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 0

        # Verify new scale initializer is created
        scale_initializers = [
            init for init in quantized_model.graph.initializer if "scale" in init.name
        ]
        assert len(scale_initializers) > 0

        # Verify DequantizeLinear references the new scale
        dq_node = next(
            node for node in quantized_model.graph.node if node.op_type == "DequantizeLinear"
        )
        assert any("scale" in input_name for input_name in dq_node.input)

    def test_cast_node_conversion(self):
        """Test that Cast nodes are properly converted from float32 to float16."""
        model = create_test_model_with_cast_nodes()

        # Run quantization
        quantized_model = quantize_weights_to_int4(model)

        # Check Cast node conversions
        for node in quantized_model.graph.node:
            if node.op_type == "Cast":
                to_attr = next(attr for attr in node.attribute if attr.name == "to")

                if "norm/Cast" in node.name or node.name == "/Cast":
                    # These should remain as float32
                    assert to_attr.i == TensorProto.FLOAT
                else:
                    # Regular Cast nodes should be converted to float16
                    assert to_attr.i == TensorProto.FLOAT16

    def test_projection_bias_and_scale_casting(self):
        """Test that projection biases and quantization scales are cast to float16."""
        model = create_test_model_with_proj_nodes()

        # Run quantization
        quantized_model = quantize_weights_to_int4(model)

        # Verify bias tensor is cast to float16
        bias_tensor = next(
            init for init in quantized_model.graph.initializer if "proj_bias" in init.name
        )
        assert bias_tensor.data_type == TensorProto.FLOAT16

        # Verify quantization scale is cast to float16
        scale_tensor = next(
            init for init in quantized_model.graph.initializer if "quant_scale" in init.name
        )
        assert scale_tensor.data_type == TensorProto.FLOAT16


class TestCastFunctions:
    """Test suite for _cast_fp8 and _cast_fp4 functions."""

    @pytest.mark.parametrize(
        ("input_array", "expected_array"),
        [
            (
                np.array([1.0, 0.5, 2.0], dtype=np.float32),
                np.array([56, 48, 64], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([-1.0, -0.5, -2.0], dtype=np.float32),
                np.array([184, 176, 192], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([0.0, -0.0], dtype=np.float32),
                np.array([0, 128], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
            (
                np.array([1e10, -1e10, 1e-10, -1e-10], dtype=np.float32),
                np.array([126, 254, 0, 128], dtype=(np.uint8, [("e4m3fn", "u1")])),
            ),
        ],
    )
    def test_cast_fp8(self, input_array, expected_array):
        """Test FP8 casting functionality."""
        result = _cast_fp8(input_array)
        assert result.dtype == np.dtype((np.uint8, [("e4m3fn", "u1")]))
        assert result.shape == expected_array.shape
        assert np.all(result == expected_array)

    @pytest.mark.parametrize(
        ("input_array", "expected_array"),
        [
            # Basic positive values
            (
                np.array([0.0, 0.5, 1.0], dtype=np.float32),
                np.array([0, 1, 2], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
            # Basic negative values
            (
                np.array([-0.5, -1.0, -1.5], dtype=np.float32),
                np.array([9, 10, 11], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
            # Boundary values with rounding
            (
                np.array([0.75, 1.75, 3.5], dtype=np.float32),
                np.array([2, 4, 6], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
            # Large values (saturate to max)
            (
                np.array([10.0, -10.0], dtype=np.float32),
                np.array([7, 15], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
            # Very small values (map to zero)
            (
                np.array([0.1, -0.1], dtype=np.float32),
                np.array([0, 8], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
            # Zero and negative zero
            (
                np.array([0.0, -0.0], dtype=np.float32),
                np.array([0, 0], dtype=(np.uint8, [("float4e2m1", "u1")])),
            ),
        ],
    )
    def test_cast_fp4(self, input_array, expected_array):
        """Test FP4 casting functionality."""
        result = _cast_fp4(input_array)
        assert result.dtype == np.dtype((np.uint8, [("float4e2m1", "u1")]))
        assert result.shape == expected_array.shape
        assert np.all(result == expected_array)
