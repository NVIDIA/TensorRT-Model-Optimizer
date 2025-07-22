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

from collections.abc import Sequence
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx.reference.custom_element_types import float8e4m3fn
from onnx.reference.ops.op_cast import Cast_19 as Cast

from modelopt.onnx import utils
from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
    remove_redundant_cast_nodes,
)
from modelopt.onnx.quantization.quant_utils import (
    compute_e8m0,
    get_amax,
    get_weights_scaling_factor,
    get_weights_scaling_factor_2,
    quantize,
)
from modelopt.onnx.utils import get_attribute, has_attribute

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
    logger.debug("Using TRT QDQ ops")
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
        np.zeros(shape, dtype=onnx.helper.tensor_dtype_to_np_dtype(dtype)),
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
    attributes: dict[str, Any] | None = None,
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
    logger.debug(f"Postprocessing QDQ nodes for {len(orig_weight_names)} weights")
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
    logger.debug(f"Added {len(q_nodes)} Q nodes and {len(dq_nodes)} DQ nodes")


def insert_pre_quant_scale_nodes(
    graph: gs.Graph, input_tensors: dict[str, str], pre_quant_scale: dict[str, np.ndarray]
):
    """Insert new mul nodes into graph.

    Args:
        graph: The graph to modify.
        input_tensors: A dictionary of weight tensor names mapped to corresponding input tensor names
        pre_quant_scale: A map from ONNX input tensor name to corresponding pre-quant scale.
    """
    logger.debug(f"Inserting pre-quant scale nodes for {len(pre_quant_scale)} tensors")

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
            input_set = {input.name for input in node.inputs}
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
    attributes: dict[str, Any] | None = None,
    zero_points: dict[str, np.ndarray] | None = None,
):
    """Insert new initializers and DQ nodes into graph.

    Args:
        graph: The graph to modify.
        weights: A map from ONNX initializer name to tensor.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        dq_only: Whether to only insert dq nodes.
    """
    logger.debug(f"Inserting DQ nodes for {len(scales)} weights")

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
        _insert_helper(name, quantized_weights[name], scale, dq_nodes, attributes, zp)  # type: ignore[arg-type]

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
    logger.debug(f"Inserting QDQ nodes for {len(scales)} weights")

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


def replace_scale_values(graph: onnx.GraphProto, act_scales_dict: dict[str, float]) -> None:
    """Replace scale values in the graph with values from calibration cache.

    Args:
        graph: ONNX graph to modify
        act_scales_dict: Dictionary mapping scale tensor names to their new values
    """
    logger.debug(f"Replacing scale values for {len(act_scales_dict)} tensors")
    initializer_indices = {init.name: idx for idx, init in enumerate(graph.initializer)}

    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue

        scale_name = node.input[1]
        if scale_name in act_scales_dict:
            if scale_name not in initializer_indices:
                raise ValueError(f"Scale tensor '{scale_name}' not found in graph initializers")

            scale = onnx.numpy_helper.from_array(
                np.float32(act_scales_dict[scale_name]), scale_name
            )
            graph.initializer[initializer_indices[scale_name]].CopyFrom(scale)
            logger.debug(f"Updated scale value for {scale_name}")
        else:
            # For weight quantizers, verify the weight tensor exists
            weight_name = node.input[0]
            if weight_name not in initializer_indices:
                raise ValueError(f"Weight tensor '{weight_name}' not found in graph initializers")


def has_qdq_nodes(onnx_model: onnx.ModelProto):
    """Check if the onnx graph already has QDQ nodes."""
    qdq_ops = {QUANTIZE_NODE_NAME, DEQUANTIZE_NODE_NAME}
    return any(node.op_type in qdq_ops for node in onnx_model.graph.node)


def _get_graph_metadata(
    graph: onnx.GraphProto,
) -> tuple[dict[str, onnx.TensorProto], dict[str, onnx.NodeProto], dict[str, list[onnx.NodeProto]]]:
    """Get helper dictionaries for efficient graph traversal and node analysis.

    Args:
        graph: ONNX graph to analyze

    Returns:
        Tuple containing:
            - initializers: Maps initializer names to their TensorProto objects
            - tensor_producers: Maps tensor names to their producer nodes
            - tensor_consumers: Maps tensor names to their consumer nodes
    """
    initializers = {init.name: init for init in graph.initializer}
    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)
    return initializers, tensor_producers, tensor_consumers


def _get_scale_and_zp(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    tensor_producers: dict[str, onnx.NodeProto],
) -> tuple[np.ndarray, np.ndarray]:
    """Get scale and zero point tensors for a node.

    Args:
        node: ONNX node to get scale and zero point for
        initializers: Dictionary of initializers
        tensor_producers: Dictionary of tensor producers

    Returns:
        Tuple of (scale_array, zero_point_array)

    Raises:
        ValueError: If scale or zero point cannot be found
    """
    # Get scale tensor
    scale_name = node.input[1]
    if scale_name in initializers:
        scale = initializers[scale_name]
    else:
        producer = tensor_producers.get(scale_name)
        if not producer or not producer.attribute:
            raise ValueError(f"Invalid scale producer for {scale_name}")
        scale = producer.attribute[0].t
    scale_array = onnx.numpy_helper.to_array(scale)

    # Get zero point tensor
    zp_name = node.input[2]
    if zp_name in initializers:
        zp = initializers[zp_name]
    else:
        producer = tensor_producers.get(zp_name)
        if not producer or not producer.attribute:
            raise ValueError(f"Invalid zero point producer for {zp_name}")
        zp = producer.attribute[0].t
    zp_array = onnx.numpy_helper.to_array(zp)

    return scale_array, zp_array


def _get_succesive_consumers(
    node: onnx.NodeProto, tensor_consumers: dict[str, list[onnx.NodeProto]]
) -> tuple[onnx.NodeProto, onnx.NodeProto]:
    """Get the DequantizeLinear node and its consumer node for a given QuantizeLinear node.

    This function validates and retrieves the next two nodes in the quantization chain:
    QuantizeLinear -> DequantizeLinear -> Operation

    Args:
        node: The QuantizeLinear node to find consumers for
        tensor_consumers: Dictionary mapping tensor names to their consumer nodes

    Returns:
        Tuple containing:
            - dq_node: The DequantizeLinear node that consumes the QuantizeLinear output
            - quantized_node: The operation node that consumes the DequantizeLinear output
    """
    dq_node = tensor_consumers.get(node.output[0], [None])[0]
    if not dq_node or dq_node.op_type != "DequantizeLinear":
        raise ValueError(f"Invalid consumer for {node.name}")

    quantized_node = tensor_consumers.get(dq_node.output[0], [None])[0]
    if not quantized_node:
        raise ValueError(f"No consumer found for {dq_node.name}")

    return dq_node, quantized_node


def _convert_weight(
    weight_array: np.ndarray,
    scale_array: np.ndarray,
    zp_array: np.ndarray,
    quantized_node: onnx.NodeProto,
) -> np.ndarray:
    """Convert a weight tensor to INT8/FP8 format based on scale and zero point.

    Args:
        weight_array: The weight tensor to convert
        scale_array: The scale tensor for quantization
        zp_array: The zero point tensor for quantization
        quantized_node: The operation node that will use the converted weight

    Returns:
        The converted weight tensor as a numpy array

    Raises:
        ValueError: If scale shape doesn't match weight shape for the operation

    Note:
        - INT8 weights are clipped to [-128, 127]
        - FP8 weights use float8e4m3fn format
    """
    # Per-op quantization axis mapping (must match ORT config)
    weight_shape = weight_array.shape
    op_type = quantized_node.op_type

    # Dynamically determine transB for Gemm
    trans_b = 0
    if op_type == "Gemm":
        for attr in quantized_node.attribute:
            if attr.name == "transB":
                trans_b = attr.i
                break

    axis_map = {
        "Conv": 0,
        "ConvTranspose": 1,
        "Gemm": 0 if trans_b else 1,
        "MatMul": 1,
    }

    if op_type not in axis_map:
        raise ValueError(f"Unsupported op_type for real weight quantization: {op_type}")

    axis = axis_map[op_type]

    if scale_array.shape and scale_array.shape[0] != weight_shape[axis]:
        raise ValueError(
            f"Scale shape {scale_array.shape} does not match weight shape {weight_shape} along axis {axis}"
        )

    reshape_dims = [1] * len(weight_shape)
    reshape_dims[axis] = scale_array.shape[0]
    scale_array = scale_array.reshape(*reshape_dims)
    zp_array = zp_array.reshape(*reshape_dims)

    # Convert to INT8/FP8
    if zp_array.dtype == float8e4m3fn:
        scaled = np.asarray(weight_array / scale_array) + zp_array
    else:
        scaled = np.asarray((weight_array / scale_array).round())
        np.clip(scaled + zp_array, -128, 127, out=scaled)

    return scaled


def _create_fp8_tensor(scaled: np.ndarray, weight_name: str) -> onnx.TensorProto:
    """Create a FLOAT8E4M3FN tensor directly from numpy array.

    This function uses ONNX's reference implementation but it is not efficient.
    TODO: Use a more efficient implementation.
    """
    # Use ONNX's reference implementation for correct float8e4m3fn conversion
    fp8_data = Cast.eval(scaled, to=onnx.TensorProto.FLOAT8E4M3FN)
    return onnx.numpy_helper.from_array(fp8_data, weight_name)


def qdq_to_dq(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to INT8/FP8 weights.

    This function converts a model with QDQ (QuantizeLinear-DequantizeLinear) nodes to a model
    with only DQ nodes for weights. It:
    1. Converts FP32/FP16 weights to INT8/FP8
    2. Updates the graph to maintain proper connections
    3. Removes redundant cast nodes in the quantized model (additional optimization for diffusers)

    Args:
        onnx_model: ONNX model protobuf to convert

    Returns:
        ONNX model protobuf with only DQ nodes for weights

    Raises:
        ValueError: If the model is invalid or conversion fails
        RuntimeError: If graph operations fail
    """
    logger.info("Converting model with QDQ nodes to DQ only model")
    if not isinstance(onnx_model, onnx.ModelProto):
        raise ValueError("Input must be an ONNX model protobuf")

    graph = onnx_model.graph
    if not graph.node:
        raise ValueError("Model graph is empty")

    initializers, tensor_producers, tensor_consumers = _get_graph_metadata(graph)
    q_nodes = [
        (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "QuantizeLinear"
    ]
    q_indices = []

    for node_idx, node in q_nodes:
        weight_name = node.input[0]
        logger.debug(f"Processing QDQ node for weight {weight_name}")

        # Nothing to do for non-const weight inputs
        if weight_name in tensor_producers:
            continue

        try:
            # Get weight tensor
            if weight_name not in initializers:
                raise ValueError(f"Weight {weight_name} not found in initializers")
            weight = initializers[weight_name]
            weight_array = onnx.numpy_helper.to_array(weight)

            # Get scale and zero point
            scale_array, zp_array = _get_scale_and_zp(node, initializers, tensor_producers)

            # Validate Q->DQ->Op pattern and get consumers
            dq_node, quantized_node = _get_succesive_consumers(node, tensor_consumers)

            # Convert weight
            scaled = _convert_weight(weight_array, scale_array, zp_array, quantized_node)

            # Create and update new weight tensor
            if zp_array.dtype == float8e4m3fn:
                new_weight = _create_fp8_tensor(scaled, weight_name)
                logger.debug(f"Converted {weight_name} to FP8")
            else:
                new_weight = onnx.numpy_helper.from_array(scaled.astype("int8"), weight_name)
                logger.debug(f"Converted {weight_name} to INT8")
            weight.CopyFrom(new_weight)

            # Track QuantizeLinear node indices for cleanup
            # Note. Scale and zero point tensors are shared between Q and DQ nodes and should not be deleted
            q_indices.append(node_idx)

            # Update following DQ nodes input name, each q should only have one dq consumer
            consumers = tensor_consumers[node.output[0]]
            assert len(consumers) == 1, f"Expected exactly one consumer for {node.name}"
            dq_node = consumers[0]
            assert dq_node.op_type == "DequantizeLinear", (
                f"Expected DequantizeLinear consumer for {node.name}"
            )
            dq_node.input[0] = weight_name

        except Exception as e:
            raise RuntimeError(f"Failed to convert node {node.name}: {e!s}")

    # Remove processed nodes
    for node_idx in sorted(q_indices, reverse=True):
        del graph.node[node_idx]

    # Remove redundant cast nodes in the quantized model
    # Note. This optimization is used by diffusers through --dq_only option, so keeping it here as well
    remove_redundant_cast_nodes(graph)
    logger.info(f"Removed {len(q_indices)} Q nodes and redundant cast nodes")

    return onnx_model


def quantize_weights_to_mxfp8(
    onnx_model: onnx.ModelProto,
) -> onnx.ModelProto:
    """Converts the weights to FP8 precision using MXFP8 quantization.

    For TRT_MXFP8DynamicQuantize, we update the output type to FP8.
    For TRT_MXFP8DequantizeLinear, we compute the scales in e8m0 format and saves them as a new initializer.
    We then expand the scale to the same shape as the weight and divide the weight by the scale to get the FP8 weights.

    Args:
        graph: ONNX model protobuf.

    Returns:
        ONNX model protobuf with weights quantized to FP8 precision using MXFP8 quantization.
    """
    logger.info("Converting weights to MXFP8 precision")
    graph = onnx_model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    tensor_producers = get_tensor_producer_nodes(graph)
    e8_m0_bias = 127
    weight_dq_nodes = [
        node
        for node in graph.node
        if node.op_type == "TRT_MXFP8DequantizeLinear"
        and any(".weight" in input for input in node.input)
    ]
    gelu_nodes = [node for node in graph.node if node.op_type == "Gelu"]
    logger.debug(f"Found {len(weight_dq_nodes)} weight DQ nodes and {len(gelu_nodes)} GELU nodes")

    for node in weight_dq_nodes:
        # Get weights and node attributes
        weight_name = node.input[0]
        logger.debug(f"Processing MXFP8 conversion for weight {weight_name}")
        weight = numpy_helper.to_array(initializers[weight_name])
        if has_attribute(node, "axis"):
            quant_axis = int(get_attribute(node, "axis"))
        else:
            quant_axis = -1
            logger.warning(
                "axis attribute not found for MXFP8DequantizeLinear node. Setting axis to -1"
            )

        if has_attribute(node, "block_size"):
            block_size = int(get_attribute(node, "block_size"))
        else:
            block_size = 32
            logger.warning(
                "block_size attribute not found for MXFP8DequantizeLinear node. Setting block_size to 32"
            )

        # Compute and save scales as uint8
        amax = get_amax(weight, quant_axis, block_size)
        se8m0_fp32 = compute_e8m0(amax, weight.shape, quant_axis, block_size)
        se8m0 = se8m0_fp32.astype(np.uint8)

        # Remove scale producer if it's a Constant node
        scale_name = node.input[1]
        scale_producer = tensor_producers[scale_name]
        if scale_producer.op_type == "Constant":
            graph.node.remove(scale_producer)

        # Create a new scale tensor
        scale_name = scale_name.replace("Constant_output_0", "scale")
        scale_tensor = onnx.numpy_helper.from_array(se8m0, scale_name)
        graph.initializer.append(scale_tensor)
        node.input[1] = scale_name

        # Convert weights to FP8
        # Expand block array so that it can be broadcasted with weight
        se8m0_fp32 = np.repeat(se8m0_fp32, block_size, axis=quant_axis)
        scaled_weight = weight / np.exp2(se8m0_fp32 - e8_m0_bias)
        # TODO: replace with an efficient kernel
        weights_e4m3 = onnx.numpy_helper.from_array(
            Cast.eval(scaled_weight, to=onnx.TensorProto.FLOAT8E4M3FN), weight_name
        )
        initializers[weight_name].CopyFrom(weights_e4m3)
        logger.debug(f"Converted {weight_name} to MXFP8")

    # Currently only tanh approximation is supported for Gelu
    for node in gelu_nodes:
        for attr in node.attribute:
            if attr.name == "approximate":
                attr.s = b"tanh"
                logger.debug(f"Updated GELU node {node.name} to use tanh approximation")

    return onnx_model


def replace_fp4qdq_with_2dq(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    initializer_indices: dict[str, int],
    value_info_map: dict[str, onnx.ValueInfoProto],
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


def fp4qdq_to_2dq(onnx_model: onnx.ModelProto, verbose: bool = False) -> onnx.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to FP4 weights and scaling factors.

    TRT_FP4QDQ nodes will get removed from the weights and have two DQ nodes with those converted FP4
    weights and scaling factors in the output model.

    Args:
        onnx_model: ONNX model protobuf.

    Returns:
        ONNX model protobuf with DQ nodes for weights and DynQ + DQ nodes for activations.
    """
    logger.info("Converting model with FP4QDQ nodes to 2DQ only model")
    graph = onnx_model.graph
    initializers = graph.initializer
    initializers_to_delete = []
    tensor_consumers = get_tensor_consumer_nodes(graph)
    initializer_indices = {
        initializer.name: idx for idx, initializer in enumerate(graph.initializer)
    }
    value_info_map = {vi.name: vi for vi in graph.value_info}
    graph_inputs = {inp.name for inp in graph.input}

    def _cast_input_dtypes(node: onnx.NodeProto, precision_dtype: str):
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

    if verbose:
        logger.info("Post-processing TRT_FP4QDQ nodes for TRT deployment")
    precision_dtype = _get_precision_dtype()
    logger.debug(f"Using precision dtype: {precision_dtype}")
    fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
    logger.debug(f"Found {len(fp4_qdq_nodes)} FP4QDQ nodes to convert")

    for node in fp4_qdq_nodes:
        idx1 = initializer_indices.get(node.input[0], None)
        assert idx1 is not None, f"Initializer for weight '{node.input[0]}' not found."
        block_size = node.attribute[0].i
        initializers_to_delete.append(initializers[idx1].name)
        logger.debug(
            f"Processing FP4QDQ node for weight {node.input[0]} with block size {block_size}"
        )

        tensor = initializers[idx1]
        w32 = utils.read_f16_tensor_as_fp32(tensor)
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

        if verbose:
            logger.debug(f"Replaced {node.name} with 2 DQ nodes")

    new_initializers = [
        init for init in graph.initializer if init.name not in initializers_to_delete
    ]
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)
    logger.info(f"Removed {len(initializers_to_delete)} initializers")

    return onnx_model
