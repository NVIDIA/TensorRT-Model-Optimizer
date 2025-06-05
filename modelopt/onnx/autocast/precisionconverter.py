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

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

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
        init_conversion_max_bytes: int = np.inf,
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
        """
        self.model = deepcopy(model)
        self.value_info_map = value_info_map
        self.initializer_map = initializer_map
        self.node_to_init_map = node_to_init_map
        self.keep_io_types = keep_io_types
        self.init_conversion_max_bytes = init_conversion_max_bytes
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
        self, high_precision_nodes: list[str], low_precision_nodes: list[str]
    ) -> onnx.ModelProto:
        """Convert model to mixed precision.

        Args:
            high_precision_nodes: List of node names to keep in high precision.
            low_precision_nodes: List of node names to convert to low precision.

        Returns:
            onnx.ModelProto: The converted mixed precision model.
        """
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
        # First clear type information for intermediates and outputs
        for vi in self.model.graph.value_info:
            vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
        for out in self.model.graph.output:
            out.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED

        # Run shape and type inference
        # Populate type information with inferred types
        self.model = shape_inference.infer_shapes(self.model, strict_mode=True, check_type=False)
        # Sanity check: Verify type correctness
        self.model = shape_inference.infer_shapes(self.model, strict_mode=True, check_type=True)

        # Update value_info_map and initializer_map with casts we added
        self.value_info_map, self.initializer_map, self.node_to_init_map = utils.setup_mappings(
            self.model
        )

        # Remove redundant casts
        self._cleanup()

        self._sanity_check()

        return self.model

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

    def _filter_unsupported_op_types(
        self, high_precision_nodes: list[str], low_precision_nodes: list[str]
    ) -> tuple[list[str], list[str]]:
        # NonMaxSuppression and Celu require FP32 inputs per ONNX standard
        # Resize and Upsample allow the data input (index 0) to be FP16/BF16 per ONNX standard, but require the scale
        # input (index 1) to be FP32. However, AutoCast requires a binary classification for each node: high/low
        # precision so we need to set Resize and Upsample to high precision
        for node in self.model.graph.node:
            if (
                node.op_type in ["Resize", "Upsample", "NonMaxSuppression", "Celu"]
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
            # If initializer is too large, skip conversion, perform cast instead
            if (
                init.raw_data
                and len(init.raw_data) > self.init_conversion_max_bytes
                and init.data_type == from_type.onnx_type
            ):
                logger.debug(
                    f"Initializer {init.name} is too large, skipping initalizer conversion, cast in "
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
                if init.data_type != from_type.onnx_type:
                    return False

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

                    # Numpy does not support bfloat16, use ONNX utils to create the raw data instead
                    if self._is_bf16(to_type) and self._is_fp32(from_type):
                        new_init = onnx.TensorProto()
                        new_init.dims.extend(np_array.shape)
                        new_init.name = new_name
                        new_init.data_type = onnx.TensorProto.BFLOAT16
                        bf16_bytes = np.vectorize(helper.float32_to_bfloat16)(np_array).astype(
                            np.uint16
                        )
                        new_init.raw_data = bf16_bytes.tobytes()
                    else:
                        assert to_type.numpy_type is not None
                        data_min, data_max = (
                            np.finfo(to_type.numpy_type).min,
                            np.finfo(to_type.numpy_type).max,
                        )
                        if np.any(np_array < data_min) or np.any(np_array > data_max):
                            logger.warning(
                                f"Initializer {init.name} used by node {node.name} contains values outside valid range,"
                                "values will be clamped"
                            )
                            np_array = np.clip(np_array, data_min, data_max)
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

    def _bypass_cast_node(self, node: onnx.NodeProto) -> None:
        # handling only a single input and output, as we only remove cast nodes
        assert len(node.input) == 1
        assert len(node.output) == 1

        input_tensor = node.input[0]
        output_tensor = node.output[0]
        is_output_producer = False

        # If removed cast node is producing a network output, we need to update the node producing the cast
        # Network output name should not be changed
        for output in self.model.graph.output:
            if output.name == output_tensor:
                is_output_producer = True
                producers = utils.get_producer_nodes(self.model, input_tensor)
                for producer in producers:
                    for i, prod_out in enumerate(producer.output):
                        if prod_out == input_tensor:
                            producer.output[i] = output_tensor
        if (
            not is_output_producer
        ):  # Reconnect consumers of the cast output to use the cast input instead
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
                    # Keep cast nodes that are nececcary producers of network outputs
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
        input_type = self._get_tensor_type(node.input[0])
        output_type = utils.get_cast_to_type(node)
        return input_type == output_type and input_type is not None

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

                # Find sequantial casts that don't change precision
                if self._is_sequential_cast(node):
                    nodes_to_remove.append(node)
                    self._bypass_cast_node(node)
                    logger.debug(f"Found removable double-cast: {node.name}")

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
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True, check_type=True)
            self.value_info_map, self.initializer_map, self.node_to_init_map = utils.setup_mappings(
                self.model
            )

    def _sanity_check(self):
        sanity_ok = True
        try:
            onnx.checker.check_model(self.model)
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
                expected_type = self.original_network_io[tensor.name]
                actual_type = tensor.type.tensor_type.elem_type

                if self.keep_io_types:
                    if actual_type != expected_type:
                        logger.error(
                            f"Internal error: Sanity check failed: I/O tensor {tensor.name} type is not preserved, "
                            "keep_io_types=True"
                        )
                        sanity_ok = False
                elif actual_type == TensorProto.FLOAT:  # only convert float I/O
                    logger.error(
                        f"Internal error: Sanity check failed: I/O tensor {tensor.name} type is FP32, "
                        "keep_io_types=False"
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
