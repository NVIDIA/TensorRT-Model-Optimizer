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

"""Various utils to support inserting Q/DQ nodes."""

import logging
from typing import Any, Sequence, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx.reference.custom_element_types import float8e4m3fn
from onnx.reference.ops.op_cast import Cast_19 as Cast

from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)
from modelopt.onnx.quantization.quant_utils import (
    get_weights_scaling_factor,
    get_weights_scaling_factor_2,
    quantize,
)

QUANTIZE_NODE_NAME = "QuantizeLinear"
DEQUANTIZE_NODE_NAME = "DequantizeLinear"

onnx_dtype_map = {
    "BFloat16": onnx.TensorProto.BFLOAT16,
    "Float": onnx.TensorProto.FLOAT,
    "Float8": onnx.TensorProto.FLOAT8E4M3FN,
    "Half": onnx.TensorProto.FLOAT16,
    "INT8": onnx.TensorProto.INT8,
    "UINT8": onnx.TensorProto.UINT8,
}


def use_trt_qdq_ops():
    """Globally set node names to TRT custom names."""
    global QUANTIZE_NODE_NAME
    QUANTIZE_NODE_NAME = "TRT_INT4QuantizeLinear"
    global DEQUANTIZE_NODE_NAME
    DEQUANTIZE_NODE_NAME = "TRT_INT4DequantizeLinear"


def _wq_name(name: str):
    return name + "_i4"


def _scale_name(name: str):
    return name + "_scale"


def _awq_scale_name(name: str):
    return name + "_awq_scale"


def _zp_name(name: str):
    return name + "_zp"


def _q_name(name: str):
    return name + "_QuantizeLinear"


def _q_out_name(name: str):
    return name + "_QuantizeLinear_Output"


def _dq_name(name: str):
    return name + "_DequantizeLinear"


def _pqs_name(name: str):
    return name + "_PQS"


def _dq_out_name(name: str):
    return name + "_DequantizeLinear_Output"


def _pqs_out_name(name: str):
    return name + "_PQS_Tensor"


def make_gs_quantized_weight(name: str, wq: np.ndarray, dtype) -> gs.Constant:
    """Create a GraphSurgeon tensor from a quantized weight tensor.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(_wq_name(name), np.asarray(wq), dtype)


def make_gs_zp(name: str, shape: Sequence[int], dtype) -> gs.Constant:
    """Create a GraphSurgeon zero-point tensor of all zeroes with the given shape.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(
        _zp_name(name),
        np.zeros(shape, dtype=onnx.mapping.TENSOR_TYPE_MAP[int(dtype)].np_dtype),
        dtype,
    )


def make_gs_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_scale_name(name), np.asarray(scale))


def make_gs_awq_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_awq_scale_name(name), np.asarray(scale))


def make_gs_quantize_output(
    name: str, shape: Sequence[int], dtype: onnx.TensorProto.DataType
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.make_variable(_q_out_name(name), dtype=dtype, shape=shape)


def make_gs_quantize_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        QUANTIZE_NODE_NAME,
        name=_q_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_pre_quant_scale_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_pqs_out_name(name), dtype=dtype, shape=shape)


def make_gs_dequantize_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_dq_out_name(name), dtype=dtype, shape=shape)


def make_gs_pre_quant_scale_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        "Mul",
        name=_pqs_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_dequantize_node(
    name: str,
    inputs: Sequence[gs.Tensor],
    outputs: Sequence[gs.Tensor],
    attributes: dict[str, Any] = None,
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        DEQUANTIZE_NODE_NAME,
        name=_dq_name(name),
        inputs=inputs,
        outputs=outputs,
        attrs=attributes,
    )


def _postprocess_qdq(
    graph: gs.Graph,
    orig_weight_names: set[str],
    q_nodes: dict[str, gs.Node] = {},
    dq_nodes: dict[str, gs.Node] = {},
):
    # Inserts all newly created nodes to graph.
    # Update all consumers of original initializers to point to the DQ nodes.
    for node in graph.nodes:
        for i in range(len(node.inputs)):
            key = node.inputs[i].name
            if key not in orig_weight_names:
                continue
            node.inputs[i] = dq_nodes[key].outputs[0]

    # Insert new nodes.
    graph.nodes.extend(q_nodes.values())
    graph.nodes.extend(dq_nodes.values())

    graph.cleanup()
    graph.toposort()


def insert_pre_quant_scale_nodes(
    graph: gs.Graph, input_tensors: dict[str, str], pre_quant_scale: dict[str, np.ndarray]
):
    """Insert new mul nodes into graph.

    Args:
        graph: The graph to modify.
        input_tensors: A dictionary of weight tensor names mapped to corresponding input tensor names
        pre_quant_scale: A map from ONNX input tensor name to corresponding pre-quant scale.
    """

    def _insert_helper(
        weight_tensor_name: str,
        input_tensor_name: str,
        scale: np.ndarray,
        mul_nodes: dict[str, gs.Node],
    ):
        pre_quant_scale_tensor = make_gs_awq_scale(weight_tensor_name, scale)
        # TODO: Study effects of caching Gemm/Matmul nodes on perf and mem usage.
        gemm_nodes = [node for node in graph.nodes if node.op in ["Gemm", "MatMul"]]
        for node in gemm_nodes:
            input_set = set([input.name for input in node.inputs])
            input_idxs = {input.name: idx for idx, input in enumerate(node.inputs)}
            if _dq_out_name(weight_tensor_name) in input_set and input_tensor_name in input_set:
                pqs_in = node.inputs[input_idxs[input_tensor_name]]
                pqs_out = make_gs_pre_quant_scale_output(
                    weight_tensor_name, shape=pqs_in.shape, dtype=scale.dtype
                )
                mul_node = make_gs_pre_quant_scale_node(
                    weight_tensor_name, inputs=[pqs_in, pre_quant_scale_tensor], outputs=[pqs_out]
                )
                node.inputs[input_idxs[input_tensor_name]] = mul_node.outputs[0]
                mul_nodes[weight_tensor_name] = mul_node

    mul_nodes = {}
    for w_name, scale in pre_quant_scale.items():
        inv_scale = 1.0 / scale
        _insert_helper(w_name, input_tensors[w_name], inv_scale, mul_nodes)

    graph.nodes.extend(mul_nodes.values())

    graph.cleanup()
    graph.toposort()


def insert_dq_nodes(
    graph: gs.Graph,
    scales: dict[str, np.ndarray],
    quantized_weights: dict[str, np.ndarray],
    attributes: dict[str, Any] = None,
    zero_points: Union[dict[str, np.ndarray], None] = None,
):
    """Insert new initializers and DQ nodes into graph.

    Args:
        graph: The graph to modify.
        weights: A map from ONNX initializer name to tensor.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        dq_only: Whether to only insert dq nodes.
    """

    def _insert_helper(
        name: str,
        wq: np.ndarray,
        scale: np.ndarray,
        dq_nodes: dict[str, gs.Node],
        attrs: dict[str, Any],
        zp: np.ndarray,
    ):
        tensor_dtype = onnx.TensorProto.INT4 if zp is None else onnx.TensorProto.UINT4
        wq_tensor = make_gs_quantized_weight(name, wq, tensor_dtype)
        scale_tensor = make_gs_scale(name, scale)
        dq_out = make_gs_dequantize_output(name, shape=wq.shape, dtype=scale.dtype)
        inputs = [wq_tensor, scale_tensor]
        if zp is not None:
            zp_tensor = gs.make_constant(_zp_name(name), zp, tensor_dtype)
            inputs.append(zp_tensor)
        dq_node = make_gs_dequantize_node(
            name,
            inputs=inputs,
            outputs=[dq_out],
            attributes=attrs,
        )
        dq_nodes[name] = dq_node

    dq_nodes = {}
    for name, scale in scales.items():
        zp = None
        if zero_points is not None:
            zp = zero_points.get(name)
            assert zp is not None, "zero-point is enabled but zero-point values not found"
        _insert_helper(name, quantized_weights[name], scale, dq_nodes, attributes, zp)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        dq_nodes=dq_nodes,
    )


def insert_qdq_nodes(
    graph: gs.Graph,
    scales: dict[str, np.ndarray],
    weight_map: dict[str, gs.Tensor],
):
    """Insert scales and QDQ nodes into graph.

    Args:
        graph: The graph to modify.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        weight_map: A map from ONNX initializer name to graphsurgeon tensor.
    """

    def _insert_helper(
        name: str,
        weight_to_quantize: gs.Tensor,
        scale: np.ndarray,
        q_nodes: dict[str, gs.Node],
        dq_nodes: dict[str, gs.Node],
    ):
        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, onnx.TensorProto.INT4)
        q_out = make_gs_quantize_output(name, weight_to_quantize.shape, onnx.TensorProto.INT4)
        q_node = make_gs_quantize_node(
            name, inputs=[weight_to_quantize, scale_tensor, zp_tensor], outputs=[q_out]
        )
        dq_out = make_gs_dequantize_output(name, shape=weight_to_quantize.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name, inputs=[q_out, scale_tensor, zp_tensor], outputs=[dq_out]
        )
        q_nodes[name] = q_node
        dq_nodes[name] = dq_node

    q_nodes, dq_nodes = {}, {}
    for name, scale in scales.items():
        _insert_helper(name, weight_map[name], scale, q_nodes, dq_nodes)

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        q_nodes=q_nodes,
        dq_nodes=dq_nodes,
    )


def replace_scale_values(graph: onnx.onnx_ml_pb2.GraphProto, act_scales_dict: dict[str, float]):
    """Replaces the scales values from calibration cache."""
    initializers = graph.initializer
    initializer_indices = {
        initializer.name: idx for idx, initializer in enumerate(graph.initializer)
    }

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            scale_input_name = node.input[1]
            if scale_input_name in act_scales_dict:
                idx = initializer_indices.get(scale_input_name, None)
                assert idx is not None, (
                    f"Expected '{scale_input_name}' to be found in 'graph.initializer', but it was not present."
                )
                scale = onnx.numpy_helper.from_array(
                    np.float32(act_scales_dict[scale_input_name]), scale_input_name
                )
                initializers[idx].CopyFrom(scale)
            else:
                # If the scale is not present in the act_scales_dict
                # then the current node must be an weight quantizer and
                # the weight should be available in the graph initializer
                assert initializer_indices.get(node.input[0], None) is not None, (
                    f"Tensor {node.input[0]} not found in initializers."
                )


def qdq_to_dq(
    onnx_model: onnx.onnx_pb.ModelProto, verbose: bool = False
) -> onnx.onnx_pb.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to INT8/FP8 weights.

    Q nodes will get removed from the weights and have only DQ nodes with those converted INT8/FP8
    weights in the output model. Also dangling Q nodes get fused and update its consumer's weight.

    Args:
        onnx_model: ONNX model protobuf.

    Returns:
        ONNX model protobuf with only DQ nodes for weights and QDQ nodes for activations.
    """
    graph = onnx_model.graph
    initializers = graph.initializer
    initializer_indices = {
        initializer.name: idx for idx, initializer in enumerate(graph.initializer)
    }

    def _get_tensor_type(tensor_name):
        for value_info in graph.value_info:
            if value_info.name == tensor_name:
                return value_info.type.tensor_type.elem_type
        return None

    def _remove_unnecessary_cast():
        # Remove two pattern of unnecessary Cast node
        cast_indices = []

        tensor_consumers = get_tensor_consumer_nodes(graph)
        output_names = [output.name for output in graph.output]

        # find all Cast node with same input and output type
        for node_idx, node in enumerate(graph.node):
            if node.op_type != "Cast":
                continue

            if any(out_name in output_names for out_name in node.output):
                continue

            # if input type matches attribute "to", this is a useless Cast node
            assert len(node.input) == 1
            input_name = node.input[0]
            idx = initializer_indices.get(input_name, None)
            if idx is not None:
                data_type = initializers[idx].data_type
            else:
                data_type = _get_tensor_type(input_name)

            attr = node.attribute[0]
            assert attr.name == "to"

            # Pattern 1: Input and Output Type are the same.
            if data_type == attr.i:
                cast_indices.append(node_idx)
            else:
                # Pattern 2: Input and Output Type differ but Cast node doesn't have a producer
                # We do the conversion and fuse Cast node.
                if idx is not None:
                    cast_indices.append(node_idx)
                    # Replace Q node input with new input
                    cast_input = onnx.numpy_helper.to_array(initializers[idx])

                    dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
                    converted_tensor = onnx.numpy_helper.from_array(
                        cast_input.astype(dtype), input_name
                    )

                    initializers[idx].CopyFrom(converted_tensor)
                else:
                    continue

            # Renew input of consumer nodes
            output_name = node.output[0]
            consumers = tensor_consumers[output_name]
            for q_node in consumers:
                for i in range(len(q_node.input)):
                    if q_node.input[i] == output_name:
                        q_node.input[i] = input_name
                        break

        # Delete Cast node
        for node_idx in sorted(cast_indices, reverse=True):
            del graph.node[node_idx]

    def _convert(node: onnx.onnx_ml_pb2.NodeProto):
        if verbose:
            logging.info(f"Processing {node.name}")

        idx1 = initializer_indices.get(node.input[0], None)
        assert idx1 is not None, (
            f"Expected '{node.input[0]}' to be found in 'graph.initializer', but it was not present."
        )
        w = initializers[idx1]

        w32 = onnx.numpy_helper.to_array(w)

        idx2 = initializer_indices.get(node.input[1], None)
        if idx2 is not None:
            y_scale = initializers[idx2]
        else:
            producer_node = tensor_producers[node.input[1]]
            attr = producer_node.attribute[0]
            assert attr.name == "value"
            y_scale = attr.t

        np_y_scale = onnx.numpy_helper.to_array(y_scale)

        idx3 = initializer_indices.get(node.input[2], None)
        if idx3 is not None:
            zero_point = initializers[idx3]
        else:
            producer_node = tensor_producers[node.input[2]]
            attr = producer_node.attribute[0]
            assert attr.name == "value"
            zero_point = attr.t

        np_zero_point = onnx.numpy_helper.to_array(zero_point)

        dq_node = tensor_consumers[node.output[0]][0]
        next_node = tensor_consumers[dq_node.output[0]][0]

        # No transpose is needed for 2D "MatMul", only for 3D (fails with PETR otherwise)
        transpose_nodes = ["Conv", "Transpose", "Gemm"]
        is_3d_matmul = next_node.op_type in "MatMul" and len(np.shape(w32)) == 3
        do_transpose = next_node.op_type in transpose_nodes or is_3d_matmul

        if do_transpose:
            w32 = np.transpose(w32, axes=[0, 2, 1]) if is_3d_matmul else np.transpose(w32)

        # Scale should be a scaler or vector with the same length as the last dimension of the weight
        assert not np_y_scale.shape or w32.shape[-1] == np_y_scale.shape[0]

        fp8 = np_zero_point.dtype == float8e4m3fn

        if fp8:
            scaled = np.asarray(w32 / np_y_scale) + np_zero_point
        else:
            scaled = np.asarray((w32 / np_y_scale).round())
            np.clip(scaled + np_zero_point, -128, 127, out=scaled)

        if do_transpose:
            scaled = np.transpose(scaled, axes=[0, 2, 1]) if is_3d_matmul else np.transpose(scaled)

        if fp8:
            w8 = onnx.numpy_helper.from_array(
                Cast.eval(scaled, to=onnx.TensorProto.FLOAT8E4M3FN), w.name
            )
        else:
            w8 = onnx.numpy_helper.from_array(scaled.astype("int8"), w.name)

        initializers[idx1].CopyFrom(w8)

        return idx2, idx3

    _remove_unnecessary_cast()

    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)

    dangling_q_indices = []
    dangling_init_indices = []

    for node_idx, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            weight_name = node.input[0]

            # Const input to quantize linear means weighted layer
            if weight_name not in tensor_producers:
                scale_init_idx, zp_init_idx = _convert(node)
                dangling_q_indices.append(node_idx)
                dangling_init_indices.extend([scale_init_idx, zp_init_idx])

                # Update following DQ nodes input name, each q should only have one dq consumer
                consumers = tensor_consumers[node.output[0]]
                assert len(consumers) == 1
                dq_node = consumers[0]
                assert dq_node.op_type == "DequantizeLinear"
                dq_node.input[0] = weight_name

    # Remove Q nodes
    for node_idx in sorted(dangling_q_indices, reverse=True):
        del graph.node[node_idx]

    return onnx_model


def replace_fp4qdq_with_2dq(
    graph: onnx.onnx_ml_pb2.GraphProto,
    node: onnx.onnx_ml_pb2.NodeProto,
    initializer_indices: dict[str, int],
    value_info_map: dict[str, onnx.onnx_ml_pb2.ValueInfoProto],
    graph_inputs: set[str],
    w_f4: np.ndarray,
    sw_f32_per_tensor: np.ndarray,
    sw_f8_per_block: np.ndarray,
    precision_dtype: str,
    block_size: int,
):
    """Replaces the given node in the ONNX graph with a subgraph consisting of two DequantizeLinear nodes.

    Args:
        graph: The ONNX graph containing the node to replace.
        node: The node to be replaced.
        initializer_indices: A dictionary mapping initializer names to their indices in the graph.
        value_info_map: A dictionary mapping value info names to their ValueInfoProto objects.
        graph_inputs: A set of graph input names.
        w_f4: NumPy array for w_f4.
        sw_f32_per_tensor: NumPy array for sw_f32_per_tensor.
        sw_f8_per_block: NumPy array for sw_f8_per_block.
        precision_dtype: The precision of the weights.
        block_size: Block size used in block quantization.
    """

    def _add_initializer(initializer):
        if initializer.name not in initializer_indices:
            graph.initializer.append(initializer)

    def _add_input_value_info(graph, tensor_proto):
        assert tensor_proto.name not in graph_inputs, (
            f"{tensor_proto.name} already in graph inputs."
        )
        assert tensor_proto.name not in value_info_map, (
            f"{tensor_proto.name} already in value info."
        )

        value_info = onnx.helper.make_tensor_value_info(
            tensor_proto.name, tensor_proto.data_type, tensor_proto.dims
        )
        graph.input.append(value_info)

    # Remove the original node from the graph
    graph.node.remove(node)
    weight_name = node.input[0]

    # Generate unique names for the initializers
    w_f4_name = weight_name + "_f4"
    sw_f8_per_block_name = weight_name + "_f8_scale"
    sw_f32_per_tensor_name = sw_f8_per_block_name + "_f32_scale"

    # Create TensorProto for initializers
    w_f4_proto = onnx.numpy_helper.from_array(w_f4, w_f4_name)
    sw_f32_per_tensor_proto = onnx.numpy_helper.from_array(
        sw_f32_per_tensor, sw_f32_per_tensor_name
    )
    sw_f8_per_block_proto = onnx.numpy_helper.from_array(sw_f8_per_block, sw_f8_per_block_name)

    # Add ValueInfo for the initializers if not present
    _add_input_value_info(graph, w_f4_proto)
    _add_input_value_info(graph, sw_f32_per_tensor_proto)
    _add_input_value_info(graph, sw_f8_per_block_proto)

    # Add the initializers to the graph
    _add_initializer(w_f4_proto)
    _add_initializer(sw_f32_per_tensor_proto)
    _add_initializer(sw_f8_per_block_proto)

    # Create DequantizeLinear_1 node: (sw_f8_per_block, sw_f32_per_tensor) -> sw_f16
    sw_f16_name = weight_name + "_f16_scale"
    dequant1 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[sw_f8_per_block_proto.name, sw_f32_per_tensor_proto.name],
        outputs=[sw_f16_name],
        name=weight_name + "_DequantizeLinear",
    )

    # Create DequantizeLinear_2 node: (w_f4, sw_f16) -> w_16
    w16_name = node.output[0]
    dequant2 = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[w_f4_proto.name, sw_f16_name],
        outputs=[w16_name],
        name=weight_name + "_DequantizeLinear_1",
        axis=-1,
        block_size=block_size,
    )

    # Add value_info for sw_f16
    # Assuming sw_f16 has the same shape as sw_f8_per_block
    sw_f16_type_proto = onnx.helper.make_tensor_type_proto(
        elem_type=onnx_dtype_map[precision_dtype], shape=sw_f8_per_block.shape
    )
    sw_f16_value_info = onnx.helper.make_value_info(name=sw_f16_name, type_proto=sw_f16_type_proto)
    graph.value_info.append(sw_f16_value_info)

    # Change the data type of w16 (output of 2nd DQ) to model weight precision type
    if w16_name in value_info_map:
        value_info_map[w16_name].type.tensor_type.elem_type = onnx_dtype_map[precision_dtype]
    else:
        raise ValueError(f"ValueInfo for {w16_name} not found.")

    # Add the new nodes to the graph
    graph.node.extend([dequant1, dequant2])


def fp4qdq_to_2dq(onnx_model: onnx.onnx_pb.ModelProto) -> onnx.onnx_pb.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to FP4 weights and scaling factors.

    TRT_FP4QDQ nodes will get removed from the weights and have two DQ nodes with those converted FP4
    weights and scaling factors in the output model.

    Args:
        onnx_model: ONNX model protobuf.

    Returns:
        ONNX model protobuf with DQ nodes for weights and DynQ + DQ nodes for activations.
    """
    graph = onnx_model.graph
    initializers = graph.initializer
    initializers_to_delete = []
    tensor_consumers = get_tensor_consumer_nodes(graph)
    initializer_indices = {
        initializer.name: idx for idx, initializer in enumerate(graph.initializer)
    }
    value_info_map = {vi.name: vi for vi in graph.value_info}
    graph_inputs = {inp.name for inp in graph.input}

    def _cast_input_dtypes(node: onnx.onnx_ml_pb2.NodeProto, precision_dtype: str):
        # Change the input types to match weight precision (precision_dtype)
        if node.op_type == "Transpose":
            maybe_matmul = tensor_consumers[node.output[0]][0]
            assert maybe_matmul.op_type == "MatMul"
            node = maybe_matmul

        # Create Cast nodes for each input of the target node except bias
        for i, input_name in enumerate(node.input[:2]):
            cast_output_name = input_name + "_f16"  # Unique name for the cast output

            # Create a Cast node to convert the input to FP16/BF16
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[input_name],  # Original input of the target node
                outputs=[cast_output_name],
                to=onnx_dtype_map[precision_dtype],  # Cast to FP16/BF16
            )

            # Insert the Cast node into the graph
            graph.node.extend([cast_node])

            # Update the target node input to use the cast node output
            node.input[i] = cast_output_name

    def _get_precision_dtype() -> str:
        # Check initializers to determine the precision of the weights
        precision_dtype = "Half"
        for initializer in graph.initializer:
            if initializer.data_type == 16:
                precision_dtype = "BFloat16"
                break  # Assuming all weights are of the same precision

        return precision_dtype

    def _bfloat16_to_float32(bf16_array):
        uint32_array = bf16_array.astype(np.uint32) << 16
        return uint32_array.view(np.float32)

    def _read_f16_tensor_as_fp32(tensor):
        if tensor.data_type == onnx.TensorProto.BFLOAT16:
            raw_data = tensor.raw_data
            uint16_array = np.frombuffer(raw_data, dtype=np.uint16)
            float32_array = _bfloat16_to_float32(uint16_array)
            tensor_shape = tuple(dim for dim in tensor.dims)
            return float32_array.reshape(tensor_shape)

        # Read FLOAT16 tensor and return
        return onnx.numpy_helper.to_array(tensor).astype(np.float32)

    print("Post-processing TRT_FP4QDQ nodes for TRT deployment...")
    precision_dtype = _get_precision_dtype()
    fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]

    for node in fp4_qdq_nodes:
        idx1 = initializer_indices.get(node.input[0], None)
        assert idx1 is not None, f"Initializer for weight '{node.input[0]}' not found."
        block_size = node.attribute[0].i
        initializers_to_delete.append(initializers[idx1].name)

        tensor = initializers[idx1]
        w32 = _read_f16_tensor_as_fp32(tensor)
        sw_f32_per_tensor = get_weights_scaling_factor_2(w32)
        sw_f32_per_block = get_weights_scaling_factor(w32, block_size, sw_f32_per_tensor)
        w_f32 = quantize(w32, block_size, sw_f32_per_block, sw_f32_per_tensor)

        # Real quantize the tensors
        w_f4 = Cast.eval(w_f32, to=onnx.TensorProto.FLOAT4E2M1)
        sw_f8_per_block = Cast.eval(sw_f32_per_block, to=onnx.TensorProto.FLOAT8E4M3FN)

        replace_fp4qdq_with_2dq(
            graph,
            node,
            initializer_indices,
            value_info_map,
            graph_inputs,
            w_f4,
            sw_f32_per_tensor,
            sw_f8_per_block,
            precision_dtype,
            block_size,
        )

        # We need to change the bias etc. type
        next_node = tensor_consumers[node.output[0]][0]
        _cast_input_dtypes(next_node, precision_dtype)

        print(f"Replaced {node.name} with 2 DQ nodes.")

    new_initializers = [
        init for init in graph.initializer if init.name not in initializers_to_delete
    ]
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)

    return onnx_model
