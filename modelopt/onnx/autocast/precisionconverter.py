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

"""Precision conversion module for ONNX models.

This module provides functionality for converting ONNX models between different floating point
precisions, specifically handling conversions between FP32 and lower precisions like FP16 or BF16.
It handles the insertion of cast operations, conversion of initializers, and ensures model validity
through type checking and cleanup of redundant operations.
"""

from collections import namedtuple
from copy import deepcopy

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import configure_logging, logger

configure_logging()

PrecisionTypes = namedtuple("PrecisionTypes", ["onnx_type", "numpy_type", "str_short", "str_full"])

PRECISION_MAP = {
    "fp32": PrecisionTypes(TensorProto.FLOAT, np.float32, "fp32", "float32"),
    "fp16": PrecisionTypes(TensorProto.FLOAT16, np.float16, "fp16", "float16"),
    "bf16": PrecisionTypes(TensorProto.BFLOAT16, None, "bf16", "bfloat16"),
}

ONNX_TYPES = [t.onnx_type for t in PRECISION_MAP.values()]

OP_TYPES_NOT_SUPPORTED_IN_LOW_PRECISION = ["Resize", "Upsample", "NonMaxSuppression", "Celu"]

# Temporarily block these ops in low precision, as they are not supported yet
OP_TYPES_NOT_SUPPORTED_IN_LOW_PRECISION.extend(["Scan", "If", "Loop", "LSTM"])


class PrecisionConverter:
    """Precision conversion module for ONNX models.

    This module provides functionality for converting ONNX models between different floating point
    precisions, specifically handling conversions between FP32 and lower precisions like FP16 or BF16.
    It handles the insertion of cast operations, conversion of initializers, and ensures model validity.

    Public Methods:
        convert: Convert specified nodes to FP16/BF16 precision while keeping others in FP32.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        value_info_map: dict[str, onnx.ValueInfoProto],
        initializer_map: dict[str, onnx.TensorProto],
        node_to_init_map: dict[str, list[str]],
        keep_io_types: bool = False,
        low_precision_type: str = "fp16",
        init_conversion_max_bytes: int | None = None,
        custom_ops: set[str] | None = None,
    ) -> None:
        """Initialize PrecisionConverter.

        Args:
            model: ONNX model to convert.
            value_info_map: Map of tensor names to value info.
            initializer_map: Map of tensor names to initializers.
            node_to_init_map: Map of node names to lists of initializer names.
            keep_io_types: Keep the input and output types of the model, otherwise they will be converted.
            low_precision_type: Precision to convert to.
            init_conversion_max_bytes: Maximum size in bytes for initializer conversion. Larger initializers will be
                                       cast at runtime.
            custom_ops: List of custom ops.
        """
        self.model = deepcopy(model)
        self.value_info_map = value_info_map
        self.initializer_map = initializer_map
        self.node_to_init_map = node_to_init_map
        self.keep_io_types = keep_io_types
        self.init_conversion_max_bytes = (
            np.inf if init_conversion_max_bytes is None else init_conversion_max_bytes
        )
        self.custom_ops = custom_ops
        if low_precision_type not in ["fp16", "bf16"]:
            raise ValueError(f"Unsupported precision type: {low_precision_type}")

        self.low_precision_type = PRECISION_MAP[low_precision_type]
        self.high_precision_type = PRECISION_MAP["fp32"]

        # Preserve original network inputs and outputs for sanity checks
        self.original_network_io = {
            io.name: io.type.tensor_type.elem_type for io in self.model.graph.input
        }
        self.original_network_io.update(
            {io.name: io.type.tensor_type.elem_type for io in self.model.graph.output}
        )

    def convert(
        self,
        high_precision_nodes: list[str],
        low_precision_nodes: list[str],
    ) -> onnx.ModelProto:
        """Convert model to mixed precision.

        Args:
            high_precision_nodes: List of node names to keep in high precision.
            low_precision_nodes: List of node names to convert to low precision.

        Returns:
            onnx.ModelProto: The converted mixed precision model.
        """
        try:
            self.model = onnx_utils.check_model(self.model)
        except onnx.checker.ValidationError as e:
            logger.error(f"Internal error: onnx.checker failed on input model {e}")
            raise Exception(
                "AutoCast can only operate on valid ONNX models, but the input model is invalid. See log for details."
            )

        # Filter out nodes that are not allowed to be in low precision
        # This is done here and not in NodeClassifier because it is required for the model to be valid
        high_precision_nodes, low_precision_nodes = self._filter_unsupported_op_types(
            high_precision_nodes, low_precision_nodes
        )

        # We remove any existing casts to FP16/BF16/FP32, as we will be adding our own
        self._remove_preexisting_casts()

        # Convert inputs to reduced precision type
        if not self.keep_io_types:
            for input in self.model.graph.input:
                if input.type.tensor_type.elem_type == self.high_precision_type.onnx_type:
                    input.type.tensor_type.elem_type = self.low_precision_type.onnx_type

        cast_down_tensors, cast_up_tensors = self._get_tensors_to_cast(low_precision_nodes)
        logger.debug(f"cast down (to {self.low_precision_type.str_full}): {cast_down_tensors}")
        logger.debug(f"cast up (to {self.high_precision_type.str_full}): {cast_up_tensors}")

        # Add cast nodes for "cast_up" tensors
        for tensor_name in cast_up_tensors:
            self._add_cast(
                tensor_name, self.high_precision_type, exclude_consumers=low_precision_nodes
            )

        # Add cast nodes for "cast_down" tensors
        for tensor_name in cast_down_tensors:
            self._add_cast(
                tensor_name,
                self.low_precision_type,
                exclude_consumers=high_precision_nodes,
            )

        # Convert initializers to correct precision according to the consumer nodes
        self._convert_initializers(
            low_precision_nodes=low_precision_nodes, high_precision_nodes=high_precision_nodes
        )

        # Infer data types (and shapes), propagating the changes we made from graph inputs to outputs
        if self.custom_ops:
            # Populate type information with inferred types
            self.model = self._propagate_types_shapes_custom_ops(self.model)
        else:
            # Clear type/shape information for intermediates and outputs
            for vi in self.model.graph.value_info:
                vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
                for idx, d in enumerate(vi.type.tensor_type.shape.dim):
                    if d.dim_value:
                        vi.type.tensor_type.shape.dim[idx].dim_param = "unk"
            for out in self.model.graph.output:
                out.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
                for idx, d in enumerate(out.type.tensor_type.shape.dim):
                    if d.dim_value:
                        out.type.tensor_type.shape.dim[idx].dim_param = "unk"
            # Populate type information with inferred types
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True, check_type=False)
            self._ensure_types_are_defined()
            # Sanity check: Verify type correctness
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True, check_type=True)

        # Update value_info_map and initializer_map with casts we added
        self.value_info_map, self.initializer_map, self.node_to_init_map = utils.setup_mappings(
            self.model
        )

        # Remove redundant casts
        self._cleanup()

        self._sanity_check()

        return self.model

    def _ensure_types_are_defined(self):
        """Ensure that all tensor types are defined."""
        for vi in self.model.graph.value_info:
            if vi.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
                vi.type.tensor_type.elem_type = self.low_precision_type.onnx_type

    def _propagate_types_shapes_custom_ops(self, model):
        """Propagate types and shapes after insertion of 'Cast' nodes or other graph modifications."""
        logger.info("Propagating tensor shapes and types in model with custom ops.")
        graph = gs.import_onnx(model)
        traversed_tensors = []

        def _get_np_type(node, inp, opset=onnx.defs.onnx_opset_version()):
            if node.op == "Cast":
                return helper.tensor_dtype_to_np_dtype(node.attrs["to"])
            elif node.op == "DequantizeLinear":
                return node.inputs[1].dtype  # scale type
            elif not inp.dtype or inp.dtype == onnx.TensorProto.UNDEFINED:
                return None
            elif node.op not in self.custom_ops:
                op_schema = onnx.defs.get_schema(node.op, opset)
                out_types = list(op_schema.outputs[0].types)
                inp_type = f"tensor({'float' if inp.dtype == 'float32' else inp.dtype})"
                return (
                    inp.dtype
                    if inp_type in out_types
                    else helper.tensor_dtype_to_np_dtype(
                        onnx_utils.onnx_type_str_to_enum(out_types[0])
                    )
                )
            return None

        def _can_propagate_type(from_type, to_type):
            try:
                from_type_onnx = helper.np_dtype_to_tensor_dtype(from_type)
                to_type_onnx = helper.np_dtype_to_tensor_dtype(to_type)
                return (
                    from_type_onnx in [*ONNX_TYPES, onnx.TensorProto.UNDEFINED]
                    and to_type_onnx in ONNX_TYPES
                )
            except Exception as e:
                logger.warning(f"Failed to check if type can be propagated: {e}")
                return False

        def _propagate_cast_type_through_nodes(node, np_type, iter=1):
            # Return if node is of cast type (from iter=2)
            indent = "  " * iter
            if iter > 1 and any(op in node.op.lower() for op in ["cast"]):
                return

            out = node.outputs[0]
            # Return if there's no consumer node
            is_graph_output_tensor = any(out.name == n.name for n in graph.outputs)
            if is_graph_output_tensor or not out.outputs:
                out.dtype = np_type
                logger.debug(f"{indent}Updated type in {out.name} to {np_type}.")
                return

            # Search children nodes
            for child_node in out.outputs:
                for child_out in child_node.outputs:
                    # Continue if the type is already correct
                    if child_out.dtype and child_out.dtype == np_type:
                        logger.debug(
                            f"{indent}Type is already correct in {child_out.name}: {child_out.dtype}. Continue."
                        )
                        continue

                    # Continue if the tensor was already traversed
                    if child_out.name in traversed_tensors and all(
                        inp.name in traversed_tensors for inp in child_node.inputs
                    ):
                        logger.debug(
                            f"{indent}Tensor {child_out.name} of shape {child_out.shape} and type {child_out.dtype} "
                            f"was already traversed. Continue."
                        )
                        return
                    if child_out.dtype and child_out.dtype != onnx.TensorProto.UNDEFINED:
                        traversed_tensors.append(child_out.name)

                    # Update tensor type if the types are supported
                    if child_out.dtype:
                        if _can_propagate_type(child_out.dtype, np_type):
                            child_out.dtype = np_type
                            logger.debug(
                                f"{indent}Updated type in {child_out.name} from {child_out.dtype} to {np_type}."
                            )
                    elif helper.np_dtype_to_tensor_dtype(np_type) in ONNX_TYPES:
                        child_out.dtype = np_type
                        logger.debug(
                            f"{indent}Updated type in {child_out.name} from 'None' to {np_type}."
                        )

                    # Propagate types to the next node
                    if child_out.outputs:
                        _propagate_cast_type_through_nodes(child_node, np_type, iter=iter + 1)
            return

        # Propagate tensor types and shapes for all layers in the graph
        for node in graph.nodes:
            # Get input and type information
            if not (inp := (node.inputs[0] if node.inputs else None)):
                continue
            if not (np_type := _get_np_type(node, inp)):
                continue

            # Propagate tensor types to outputs
            for out in node.outputs:
                # Update the output type if relevant
                if not out.dtype or _can_propagate_type(out.dtype, np_type):
                    out.dtype = np_type

                # Set the output shape
                if not out.shape:
                    if isinstance(inp, gs.Constant):
                        out.shape = inp.values.shape
                    elif inp.inputs and inp.inputs[0].op == "Constant":
                        out.shape = inp.inputs[0].attrs["value"].values.shape
                    elif inp.shape:
                        out.shape = inp.shape

            # Propagate tensor types to the children nodes (until another Cast or Q node is met)
            _propagate_cast_type_through_nodes(node, np_type)

        return gs.export_onnx(graph)

    def _is_bf16(self, type: PrecisionTypes = None) -> bool:
        if type is None:
            type = self.low_precision_type
        return type.onnx_type == onnx.TensorProto.BFLOAT16

    def _is_fp16(self, type: PrecisionTypes = None) -> bool:
        if type is None:
            type = self.low_precision_type
        return type.onnx_type == onnx.TensorProto.FLOAT16

    def _is_fp32(self, type: PrecisionTypes = None) -> bool:
        if type is None:
            type = self.high_precision_type
        return type.onnx_type == onnx.TensorProto.FLOAT

    def _get_node_initializers_map(self) -> dict[str, list[str]]:
        """Creates a mapping from node names to lists of initializer names used as inputs by that node.

        Returns:
            dict[str, list[str]]: Mapping from node names to lists of initializer names.
        """
        node_to_initializers = {}
        for node in self.model.graph.node:
            initializer_inputs = [
                self.initializer_map[input_name]
                for input_name in node.input
                if input_name in self.initializer_map
            ]
            node_to_initializers[node.name] = initializer_inputs
        return node_to_initializers

    def _is_castable_tensor(self, tensor_name: str) -> bool:
        if tensor_name in self.value_info_map:
            return self.value_info_map[tensor_name].type.tensor_type.elem_type in ONNX_TYPES
        elif tensor_name in self.initializer_map:
            return self.initializer_map[tensor_name].data_type in ONNX_TYPES
        else:
            logger.warning(f"Did not find {tensor_name} in value info map! Assuming not castable")
            return False

    def _is_empty_tensor(self, tensor_name: str) -> bool:
        if tensor_name in self.value_info_map:
            tensor_info = self.value_info_map[tensor_name]
            for dim in tensor_info.type.tensor_type.shape.dim:
                if (dim.HasField("dim_value") and dim.dim_value == 0) or (
                    dim.HasField("dim_param") and dim.dim_param == "0"
                ):
                    return True
        return False

    def _filter_unsupported_op_types(
        self, high_precision_nodes: list[str], low_precision_nodes: list[str]
    ) -> tuple[list[str], list[str]]:
        # NonMaxSuppression and Celu require FP32 inputs per ONNX standard
        # Resize and Upsample allow the data input (index 0) to be FP16/BF16 per ONNX standard, but require the scale
        # input (index 1) to be FP32. However, AutoCast requires a binary classification for each node: high/low
        # precision so we need to set Resize and Upsample to high precision
        for node in self.model.graph.node:
            if (
                node.op_type in OP_TYPES_NOT_SUPPORTED_IN_LOW_PRECISION
                and node.name in low_precision_nodes
            ):
                low_precision_nodes.remove(node.name)
                high_precision_nodes.append(node.name)
                logger.debug(
                    f"Node {node.name} (op type: {node.op_type}) is not supported in low precision, moving"
                    " to high precision"
                )
        return high_precision_nodes, low_precision_nodes

    def _get_tensors_to_cast(self, low_precision_nodes: list[str]) -> tuple[list[str], list[str]]:
        cast_to_fp16 = []  # Tensors to cast down to FP16
        cast_to_fp32 = []  # Tensors to cast up to FP32

        # Get tensors for FP16 nodes
        for node in self.model.graph.node:
            if node.name in low_precision_nodes:
                # Cast inputs to FP16 nodes down to FP16
                cast_to_fp16.extend(node.input)
                # Cast outputs from FP16 nodes up to FP32
                cast_to_fp32.extend(node.output)

        # Handle consumers and producers of network inputs and outputs
        high_precision_nodes = [
            node for node in self.model.graph.node if node.name not in low_precision_nodes
        ]
        network_inputs = [input.name for input in self.model.graph.input]
        network_outputs = [output.name for output in self.model.graph.output]
        for node in high_precision_nodes:
            # Add cast up for network inputs
            cast_to_fp32.extend([input for input in node.input if input in network_inputs])
            # Add cast down for network outputs
            cast_to_fp16.extend([output for output in node.output if output in network_outputs])

        # Remove initializers, they are handled separately
        initializers = {init.name for init in self.model.graph.initializer}
        cast_to_fp16 = list(set(cast_to_fp16) - initializers)
        cast_to_fp32 = list(set(cast_to_fp32) - initializers)

        # Filter out non-float tensors
        cast_to_fp16 = [t for t in cast_to_fp16 if self._is_castable_tensor(t)]
        cast_to_fp32 = [t for t in cast_to_fp32 if self._is_castable_tensor(t)]

        logger.debug(f"tensors to cast to FP16: {cast_to_fp16}")
        logger.debug(f"tensors to cast to FP32: {cast_to_fp32}")
        return cast_to_fp16, cast_to_fp32

    def _convert_initializers(
        self, low_precision_nodes: list[str], high_precision_nodes: list[str]
    ) -> onnx.ModelProto:
        def convert_initializer(
            init: onnx.TensorProto,
            node: onnx.NodeProto,
            from_type: PrecisionTypes,
            to_type: PrecisionTypes,
        ):
            if init.data_type != from_type.onnx_type:
                logger.debug(
                    f"Initializer {init.name} has data type {init.data_type}, and size {len(init.raw_data)},"
                    "skipping conversion"
                )
                return False

            # If initializer is too large, skip conversion, perform cast instead
            if init.raw_data and len(init.raw_data) > self.init_conversion_max_bytes:
                logger.debug(
                    f"Initializer {init.name} is too large, skipping initializer conversion, cast in "
                    "runtime instead"
                )
                exclude_consumers = (
                    low_precision_nodes if self._is_fp32(to_type) else high_precision_nodes
                )
                self._add_cast(init.name, to_type, exclude_consumers=exclude_consumers)
                return True
            try:
                np_array = numpy_helper.to_array(init)
                assert from_type.str_short in PRECISION_MAP
                assert to_type.str_short in PRECISION_MAP
                assert from_type.str_short != to_type.str_short

                if np_array.dtype == from_type.numpy_type:
                    consumers = [n.name for n in utils.get_consumer_nodes(self.model, init.name)]
                    should_duplicate = len(consumers) > 1 and set(consumers) & set(
                        high_precision_nodes
                    )

                    if should_duplicate:
                        # Create a new low precision copy with a different name
                        new_name = f"{init.name}_{to_type.str_short}"
                        logger.debug(
                            f"Initializer {init.name} is shared, creating {to_type.str_short} copy as {new_name} due "
                            f"to node {node.name}"
                        )

                        # Update the node to use the new initializer
                        for i, input_name in enumerate(node.input):
                            if input_name == init.name:
                                node.input[i] = new_name
                                break

                        if init.name in initializer_converted_dup:
                            return False
                        initializer_converted_dup.append(init.name)
                    else:
                        if init.name in initializer_converted:
                            return False
                        new_name = init.name
                        logger.debug(
                            f"Converting initializer {new_name} to {to_type.str_short} due to node {node.name}"
                        )
                        initializer_converted.append(init.name)
                        self.model.graph.initializer.remove(init)

                    # Numpy does not support bfloat16, use ml_dtypes to create the raw data instead
                    if self._is_bf16(to_type) and self._is_fp32(from_type):
                        new_init = onnx.TensorProto()
                        new_init.dims.extend(np_array.shape)
                        new_init.name = new_name
                        new_init.data_type = onnx.TensorProto.BFLOAT16
                        bf16_bytes = np_array.astype(ml_dtypes.bfloat16).view(np.uint16)
                        new_init.raw_data = bf16_bytes.tobytes()
                    else:
                        assert to_type.numpy_type is not None
                        data_max, data_lowest = (
                            np.finfo(to_type.numpy_type).max,
                            np.finfo(to_type.numpy_type).smallest_subnormal,
                        )
                        if np.any(np.abs(np_array) > data_max):
                            logger.warning(
                                f"Initializer {init.name} used by node {node.name} contains values larger than "
                                f"largest {to_type.str_short} value, values will be clamped to {data_max}."
                            )
                            np_array = np.clip(np_array, -1 * data_max, data_max)
                        if np.any((np_array != 0.0) & (np.abs(np_array) < data_lowest)):
                            logger.warning(
                                f"Initializer {init.name} used by node {node.name} contains values smaller than "
                                f"smallest {to_type.str_short} value, values will be replaced with {data_lowest:.1e}."
                            )
                            np_array = np.where(
                                (np_array != 0.0) & (np.abs(np_array) < data_lowest),
                                data_lowest,
                                np_array,
                            )
                        new_array = np_array.astype(to_type.numpy_type)
                        new_init = numpy_helper.from_array(new_array, new_name)
                    self.model.graph.initializer.extend([new_init])
                    return True
                return False
            except Exception as e:
                logger.error(f"Error converting initializer {init.name}: {e}")
                return False

        initializer_converted = []
        initializer_converted_dup = []
        modified = False
        for node in self.model.graph.node:
            if node.name in low_precision_nodes:
                for init in self.node_to_init_map[node.name]:
                    modified |= convert_initializer(
                        init,
                        node,
                        from_type=self.high_precision_type,
                        to_type=self.low_precision_type,
                    )
            if modified:
                _, _, self.node_to_init_map = utils.setup_mappings(self.model)

            if node.name in high_precision_nodes:
                for init in self.node_to_init_map[node.name]:
                    convert_initializer(
                        init,
                        node,
                        from_type=self.low_precision_type,
                        to_type=self.high_precision_type,
                    )

    def _replace_tensor_name(
        self, consumers: list[onnx.NodeProto], original_tensor_name: str, new_tensor_name: str
    ) -> None:
        """Replace occurrences of a tensor name in the given consumers' inputs with a new tensor name."""
        for consumer in consumers:
            for idx, inp in enumerate(consumer.input):
                if inp == original_tensor_name:
                    consumer.input[idx] = new_tensor_name

    def _bypass_cast_node(self, node: onnx.NodeProto) -> None:
        # handling only a single input and output, as we only remove cast nodes
        assert len(node.input) == 1
        assert len(node.output) == 1

        input_tensor = node.input[0]
        output_tensor = node.output[0]

        # Check if the cast output is also a graph output
        is_output_producer = any(output.name == output_tensor for output in self.model.graph.output)

        # If the removed cast node is producing a network output, update the producer of the cast input so
        # the network output name is preserved.
        if is_output_producer:
            producers = utils.get_producer_nodes(self.model, input_tensor)
            for producer in producers:
                for i, prod_out in enumerate(producer.output):
                    if prod_out == input_tensor:
                        producer.output[i] = output_tensor
                        consumers = utils.get_consumer_nodes(self.model, prod_out)
                        if len(consumers) > 1:
                            self._replace_tensor_name(consumers, prod_out, output_tensor)
        else:
            # Reconnect consumers of the cast output to use the cast input instead
            consumers = utils.get_consumer_nodes(self.model, output_tensor)
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == output_tensor:
                        consumer.input[i] = input_tensor

    def _remove_preexisting_casts(self) -> None:
        nodes_to_remove = []
        for node in self.model.graph.node:
            if node.op_type == "Cast":
                cast_from_type = self._get_tensor_type(node.input[0])
                cast_to_type = utils.get_cast_to_type(node)
                is_fp_cast = cast_to_type in [
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.FLOAT,
                ] and cast_from_type in [
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.FLOAT,
                    onnx.TensorProto.BFLOAT16,
                ]
                # Check if input comes from an initializer - don't remove cast in that case
                input_from_initializer = node.input[0] in {
                    init.name for init in self.model.graph.initializer
                }
                if is_fp_cast and not input_from_initializer:
                    # Keep cast nodes that are necessary producers of network outputs
                    if any(node.input[0] == out.name for out in self.model.graph.output) and any(
                        node.output[0] == out.name for out in self.model.graph.output
                    ):
                        continue
                    nodes_to_remove.append(node)
                    self._bypass_cast_node(node)
        logger.debug(f"Removing {len(nodes_to_remove)} pre-existing casts")

        for node in nodes_to_remove:
            self.model.graph.node.remove(node)

    def _add_cast(
        self, tensor_name: str, cast_to: PrecisionTypes, exclude_consumers: list[str] = []
    ) -> None:
        """Adds a cast operation on a tensor and reconnects its consumers.

        Args:
            tensor_name: Name of the tensor to cast.
            cast_to: Target precision type to cast to.
            exclude_consumers: List of consumer nodes to exclude from reconnection.
        """
        # Empty tensors may have special handling in ONNX (e.g. for Resize scales) which can break when redundant casts
        # are injected. Since there's no data, it's safe to only update the metadata.
        if self._is_empty_tensor(tensor_name):
            logger.debug(f"Fake-casting empty tensor: {tensor_name}")
            if tensor_name in self.value_info_map:
                tensor_info = self.value_info_map[tensor_name]
                tensor_info.type.tensor_type.elem_type = cast_to.onnx_type
                # Update the corresponding value_info in the model graph
                for vi in self.model.graph.value_info:
                    if vi.name == tensor_name:
                        vi.type.tensor_type.elem_type = cast_to.onnx_type
                        break

                # Also check if tensor is output of a Constant node and update its value attribute
                for node in self.model.graph.node:
                    if node.op_type == "Constant" and tensor_name in node.output:
                        logger.debug(f"Found {tensor_name} as output of Constant node {node.name}")
                        for attr in node.attribute:
                            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                                attr.t.data_type = cast_to.onnx_type
                                break
                        break
            else:
                logger.error(f"Failed to fake-cast empty tensor: {tensor_name} not found.")
            return

        cast_output_name = f"{tensor_name}_cast_to_{cast_to.str_short}"

        cast_node = helper.make_node(
            "Cast",
            inputs=[tensor_name],
            outputs=[cast_output_name],
            to=cast_to.onnx_type,
            name=f"{tensor_name}_cast_to_{cast_to.str_short}",
        )

        consumer_nodes = utils.get_consumer_nodes(self.model, tensor_name)
        consumer_nodes = [n for n in consumer_nodes if n.name not in exclude_consumers]
        for node in consumer_nodes:
            for i, input_name in enumerate(node.input):
                if input_name == tensor_name:
                    node.input[i] = cast_output_name

        # Update network output
        for output in self.model.graph.output:
            if output.name == tensor_name and (
                (self.keep_io_types and cast_to.onnx_type == output.type.tensor_type.elem_type)
                or (
                    not self.keep_io_types
                    and cast_to.onnx_type == self.low_precision_type.onnx_type
                )
            ):
                output.name = cast_output_name
                break

        # Find producer node to insert cast after it
        producer_nodes = utils.get_producer_nodes(self.model, tensor_name)
        if producer_nodes:
            # Insert after the producer node
            # Find index by iterating since RepeatedCompositeContainer doesn't support index()
            producer_idx = -1
            for i, node in enumerate(self.model.graph.node):
                if node == producer_nodes[0]:
                    producer_idx = i
                    break
            self.model.graph.node.insert(producer_idx + 1, cast_node)
        else:
            # If no producer (e.g. network input), insert at beginning
            self.model.graph.node.insert(0, cast_node)

        logger.debug(f"Inject cast to {cast_to.str_full} on {tensor_name}")

    def _cleanup(self):
        # Cleanup dead-end cast nodes
        self._cleanup_no_consumer_nodes()

        # Cleanup double same-type cast nodes that produce network outputs before calling _fix_network_output_names
        # This is necessary because fix_network_output_names only handles one level of cast nodes
        self._cleanup_pre_output_same_type_cast()

        # Restores the original output names, must execute before removing cast nodes, otherwise
        # the nodes generating the outputs might be removed
        self._fix_network_output_names()

        # Remove redundant casts
        self._remove_redundant_casts()

    def _cleanup_no_consumer_nodes(self):
        network_outputs = {o.name for o in self.model.graph.output}
        nodes_to_remove = [
            node
            for node in self.model.graph.node
            if not any(
                out in network_outputs or utils.get_consumer_nodes(self.model, out)
                for out in node.output
            )
        ]
        for node in nodes_to_remove:
            # We only add Cast nodes, other nodes with no consumers originate from the original model
            if node.op_type != "Cast":
                logger.debug(
                    f"Removing non-cast node with no consumers: {node.name} (type: {node.op_type})"
                )
            self.model.graph.node.remove(node)

    def _cleanup_pre_output_same_type_cast(self):
        if not self.keep_io_types:
            return

        for output in self.model.graph.output:
            if "_cast_to_" in output.name:
                out_producer_nodes = utils.get_producer_nodes(self.model, output.name)
                if len(out_producer_nodes) == 1 and out_producer_nodes[0].op_type == "Cast":
                    second_cast_node = out_producer_nodes[0]
                    cast_producer_nodes = utils.get_producer_nodes(
                        self.model, second_cast_node.input[0]
                    )
                    if len(cast_producer_nodes) == 1 and cast_producer_nodes[0].op_type == "Cast":
                        first_cast_node = cast_producer_nodes[0]
                        if (
                            self._is_same_type_cast(first_cast_node)
                            and utils.get_cast_to_type(second_cast_node)
                            == self.high_precision_type.onnx_type
                        ):
                            logger.debug(f"Removing pre-output double cast: {first_cast_node.name}")
                            self._bypass_cast_node(first_cast_node)
                            self.model.graph.node.remove(first_cast_node)

    def _is_same_type_cast(self, node: onnx.NodeProto) -> bool:
        assert node.op_type == "Cast"
        input_types = [self._get_tensor_type(inp) for inp in node.input]
        output_type = utils.get_cast_to_type(node)
        return all(inp_type == output_type for inp_type in input_types) and input_types is not None

    def _is_sequential_cast(self, node: onnx.NodeProto) -> bool:
        assert node.op_type == "Cast"
        output_type = utils.get_cast_to_type(node)

        # Cast to high precision -> cast to low precision, first cast has no impact and can be safely removed
        # Cast to low precision -> cast to high precision affects precision and should not be removed
        precision_order = [
            TensorProto.DOUBLE,
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
        ]
        consumers = [
            n for n in utils.get_consumer_nodes(self.model, node.output[0]) if n.op_type == "Cast"
        ]

        # If the first cast has additional consumers, we should not remove it
        if len(consumers) != 1:
            return False

        next_node = consumers[0]
        first_cast_type = output_type
        second_cast_type = utils.get_cast_to_type(next_node)

        return (
            first_cast_type in precision_order
            and second_cast_type in precision_order
            and precision_order.index(first_cast_type) <= precision_order.index(second_cast_type)
        )

    def _remove_redundant_casts(self):
        """Removes both sequential casts and casts that don't change precision.

        This method optimizes the graph by removing unnecessary cast operations that either:
        1. Don't actually change the data type
        2. Could be replaced by a single cast operation
        """
        if self.custom_ops:
            self.model = self._propagate_types_shapes_custom_ops(self.model)
        else:
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True)
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True, check_type=True)

        nodes_to_remove = []
        for node in self.model.graph.node:
            if node.op_type == "Cast":
                # Find cast nodes that don't change precision
                if self._is_same_type_cast(node):
                    nodes_to_remove.append(node)
                    self._bypass_cast_node(node)
                    logger.debug(f"Found redundant same-type cast: {node.name}")
                    continue

                # Find sequential casts that don't change precision
                if self._is_sequential_cast(node):
                    nodes_to_remove.append(node)
                    self._bypass_cast_node(node)
                    logger.debug(f"Found removable double-cast: {node.name}")

                # Find foldable Constant -> Cast. Initializers are handled by _convert_initializers.
                if self._is_foldable_constant_cast_pattern(node):
                    nodes_to_remove.append(node)
                    cast_producers = utils.get_producer_nodes(self.model, node.input[0])
                    assert len(cast_producers) == 1 and cast_producers[0].op_type == "Constant"
                    constant_producer = cast_producers[0]
                    self._convert_constant_values(constant_producer, node)
                    self._bypass_cast_node(node)
                    logger.debug(f"Found foldable Constant->Cast pattern, removing {node.name}")

        logger.debug(f"Removing redundant casts: {[n.name for n in nodes_to_remove]}")
        for node in nodes_to_remove:
            self.model.graph.node.remove(node)

    def _fix_network_output_names(self):
        modified = False
        for output in self.model.graph.output:
            if "_cast_to_" in output.name:
                post_cast_name = output.name
                producer_nodes = utils.get_producer_nodes(self.model, output.name)
                if (
                    len(producer_nodes) == 1
                    and producer_nodes[0].op_type == "Cast"
                    and producer_nodes[0].output[0] == output.name
                ):
                    cast_node = producer_nodes[0]
                    assert cast_node.op_type == "Cast"
                    original_name = cast_node.input[0]
                    pre_cast_name = original_name + "_pre_cast"
                    output.name = original_name
                    # Update all consumers of the original (pre-cast) output to use the pre-cast name
                    for node in utils.get_consumer_nodes(self.model, original_name):
                        if node == cast_node:
                            continue
                        for i, input_name in enumerate(node.input):
                            if input_name == original_name:
                                node.input[i] = pre_cast_name
                                # do not break, can use the same tensor for multiple node inputs
                    # Update all consumers of the post-cast output to use the original name
                    for node in utils.get_consumer_nodes(self.model, post_cast_name):
                        for i, input_name in enumerate(node.input):
                            if input_name == post_cast_name:
                                node.input[i] = original_name
                                # do not break, can use the same tensor for multiple node inputs
                    # Update all producers of the original output to use the original name
                    cast_producer_nodes = utils.get_producer_nodes(self.model, cast_node.input[0])
                    for node in cast_producer_nodes:
                        for i, node_output in enumerate(node.output):
                            if node_output == original_name:
                                node.output[i] = pre_cast_name
                                break
                    cast_node.input[0] = pre_cast_name
                    cast_node.output[0] = original_name

                    modified = True
                    logger.debug(f"Fixed network output names: {post_cast_name} -> {output.name}")
        if modified:
            if self.custom_ops:
                self.model = self._propagate_types_shapes_custom_ops(self.model)
            else:
                self.model = onnx_utils.infer_shapes(self.model, strict_mode=True, check_type=True)
            self.value_info_map, self.initializer_map, self.node_to_init_map = utils.setup_mappings(
                self.model
            )

    def _sanity_check(self):
        sanity_ok = True
        try:
            onnx_utils.check_model(self.model)
        except onnx.checker.ValidationError as e:
            logger.error(f"Internal error: onnx.checker failed: {e}")
            sanity_ok = False

        network_inputs = list(self.model.graph.input)
        network_outputs = list(self.model.graph.output)
        disconnected_outputs = []

        # Verify that the output tensors are not disconnected
        for output in network_outputs:
            producer_nodes = utils.get_producer_nodes(self.model, output.name)
            if len(producer_nodes) == 0:
                logger.warning(
                    f"Output tensor {output.name} is disconnected. This may be benign if it's part of a cast operation "
                    "chain (e.g., output1 -> cast -> output2)."
                )
                disconnected_outputs.append(output)

        # Verify that the original and current network inputs/outputs match
        current_io = {io.name for io in network_inputs + network_outputs}
        original_io = set(self.original_network_io.keys())
        if current_io != original_io:
            logger.error(
                f"Internal error: Sanity check failed: Network inputs/outputs do not match original inputs/outputs. "
                f"Current: {current_io}, Original: {original_io}"
            )
            sanity_ok = False

        # Verify that the original input and output types are handled according to keep_io_types
        if sanity_ok:
            for tensor in network_inputs + network_outputs:
                if tensor in disconnected_outputs:
                    logger.debug(
                        f"Skipping validating type of disconnected output tensor {tensor.name}"
                    )
                    continue
                original_type = self.original_network_io[tensor.name]
                converted_type = tensor.type.tensor_type.elem_type

                if converted_type != original_type:
                    # There's one allowed exception: FP32 I/O converted to the selected low precision type with
                    # keep_io_types=False
                    if (
                        original_type == onnx.TensorProto.FLOAT
                        and converted_type == self.low_precision_type.onnx_type
                        and not self.keep_io_types
                    ):
                        continue
                    else:
                        logger.error(
                            f"Internal error: Sanity check failed: Unexpected type in I/O tensor {tensor.name}, "
                            f"keep_io_types={self.keep_io_types}, original type: {original_type}, converted type: "
                            f"{converted_type}."
                        )
                        sanity_ok = False
        if not sanity_ok:
            raise Exception("Sanity Check Failed")

    def _get_tensor_type(self, tensor_name):
        if tensor_name in self.value_info_map:
            return self.value_info_map[tensor_name].type.tensor_type.elem_type
        if tensor_name in self.initializer_map:
            return self.initializer_map[tensor_name].data_type
        raise Exception(f"did not find tensor {tensor_name}")

    def _convert_constant_values(self, const_node, cast_node: onnx.NodeProto) -> None:
        original_tensor = const_node.attribute[0].t
        if original_tensor.data_type == onnx.TensorProto.BFLOAT16:
            original_data = onnx_utils.read_f16_tensor_as_fp32(original_tensor)
        else:
            original_data = onnx.numpy_helper.to_array(original_tensor)

        # Precompute casted value
        cast_to_type = utils.get_cast_to_type(cast_node)
        cast_dtype = onnx.helper.tensor_dtype_to_np_dtype(cast_to_type)

        # Handle bfloat16 conversion manually since numpy doesn't support it natively
        if cast_to_type == onnx.TensorProto.BFLOAT16:
            casted_data = original_data.astype(ml_dtypes.bfloat16)
        else:
            casted_data = original_data.astype(cast_dtype)

        # Workaround for 0-dimensional tensors (scalars)
        if casted_data.ndim == 0:
            casted_data = casted_data.reshape(1)

        # Create a new constant node with casted data
        if cast_to_type == onnx.TensorProto.BFLOAT16:
            # Create TensorProto manually for bfloat16
            tensor_proto = onnx.TensorProto()
            tensor_proto.name = const_node.output[0]
            tensor_proto.data_type = onnx.TensorProto.BFLOAT16
            tensor_proto.dims.extend(casted_data.shape)
            # Convert bfloat16 to raw bytes
            bf16_bytes = casted_data.astype(ml_dtypes.bfloat16).view(np.uint16)
            tensor_proto.raw_data = bf16_bytes.tobytes()
        else:
            # Create tensor manually to ensure proper handling
            tensor_proto = onnx.numpy_helper.from_array(casted_data)
            tensor_proto.name = const_node.output[0]

        new_const_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=const_node.output,
            value=tensor_proto,
            name=const_node.name,
        )

        # Replace the original constant node with the new constant node
        # The scope of this function is to convert the constant node data. Removing the cast is done later.
        for node in utils.get_consumer_nodes(self.model, const_node.name):
            for i, input_name in enumerate(node.input):
                if input_name == const_node.name:
                    node.input[i] = new_const_node.output[0]
                    break

        const_idx = -1
        for i, node in enumerate(self.model.graph.node):
            if node == const_node:
                const_idx = i
                break

        self.model.graph.node.remove(const_node)
        self.model.graph.node.insert(const_idx, new_const_node)
        # The Cast node is the sole consumer of the Constant node, guaranteed by _is_foldable_constant_cast_pattern
        cast_node.input[0] = new_const_node.output[0]

    def _is_foldable_constant_cast_pattern(self, node: onnx.NodeProto) -> bool:
        """Constant -> Cast and Cast is the only consumer of the Constant node."""
        assert node.op_type == "Cast"

        producer = utils.get_producer_nodes(self.model, node.input[0])

        const_producer = (
            producer[0] if len(producer) == 1 and producer[0].op_type == "Constant" else None
        )

        if const_producer:
            get_consumer_nodes = utils.get_consumer_nodes(self.model, const_producer.output[0])
            return len(get_consumer_nodes) == 1 and get_consumer_nodes[0] == node
        return False
