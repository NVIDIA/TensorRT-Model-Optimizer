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

"""Graph sanitization and optimization for ONNX models."""

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import logger


class GraphSanitizer:
    """A class for sanitizing ONNX model graphs, a part of the AutoCast tool."""

    def __init__(self, model: onnx.ModelProto, min_opset: int = 13) -> None:
        """Initialize GraphSanitizer.

        Args:
            model: ONNX model to sanitize
            min_opset: minimum opset version to use
        """
        self.model = model
        self.min_opset = min_opset
        self.standard_ops = {schema.name for schema in onnx.defs.get_all_schemas()}

    def sanitize(self) -> None:
        """Sanitize the model graph.

        Currently, this finds decomposed LayerNorm patterns and replaces them with a single LayerNormalization operator.
        Additional functionality may be added in the future.
        """
        self.find_custom_nodes()
        self.remove_disconnected_outputs()
        self.convert_opset(self.min_opset)
        self.replace_layernorm_pattern()
        onnx_utils.name_onnx_nodes(self.model.graph)

    def find_custom_nodes(self) -> None:
        """Find custom nodes in the model.

        Scans through all nodes in the graph and logs any nodes that use custom operators
        that are not part of the standard ONNX operator set.
        """
        custom_ops = {
            node.op_type for node in self.model.graph.node if node.op_type not in self.standard_ops
        }
        if custom_ops:
            logger.error(f"Found custom operators: {custom_ops}")
            raise ValueError("AutoCast does not support custom operators yet.")

    def remove_disconnected_outputs(self) -> None:
        """Remove disconnected outputs from the model."""
        tensors_to_remove = []
        for tensor in self.model.graph.output:
            if not utils.get_producer_nodes(self.model, tensor.name):
                tensors_to_remove.append(tensor)
                logger.debug(f"Found disconnected output: {tensor.name}")

        if tensors_to_remove:
            logger.warning(
                f"Found {len(tensors_to_remove)} disconnected outputs. Removing disconnected outputs from the graph."
            )

        for tensor in tensors_to_remove:
            self.model.graph.output.remove(tensor)

    def convert_opset(self, min_opset: int) -> None:
        """Convert the model to the given opset version.

        Args:
            min_opset: minimum opset version to use

        The method checks all opset imports and converts the model if any are below the minimum version.
        """
        # Check all opset imports
        default_opsets = list(self.model.opset_import)

        # Convert if any default domain opset is below minimum
        if any(op.version < min_opset for op in default_opsets):
            invalid_opsets = [op.version for op in default_opsets if op.version < min_opset]
            try:
                self.model = onnx.version_converter.convert_version(self.model, min_opset)
            except Exception as e:
                logger.warning(f"Failed to convert model to opset {min_opset}: {e!s}")
                logger.warning(f"Attempting to continue with the original opsets: {invalid_opsets}")

    def replace_layernorm_pattern(self) -> None:
        """Detects and replaces LayerNorm operation patterns.

        This method scans through the graph looking for sequences of operations that implement LayerNorm functionality
        and replaces them with the more efficient LayerNormalization operator.
        """
        nodes_to_remove = []
        modified = False

        for node in self.model.graph.node:
            if node.op_type != "ReduceMean":
                continue

            try:
                pattern = self._match_layernorm_pattern(node)
                if not pattern:
                    continue

                ln_node = self._create_layernorm_node(pattern)
                insert_idx = self._find_insertion_point(pattern["input_name"])

                # Insert LayerNorm node and update graph
                self.model.graph.node.insert(insert_idx, ln_node)
                nodes_to_remove.extend(pattern["nodes_to_remove"])
                modified = True

                logger.info(f"Replaced LayerNorm pattern starting at {node.name}")

            except Exception as e:
                logger.debug(f"Failed to match LayerNorm pattern at {node.name}: {e!s}")
                continue
            self._remove_nodes(nodes_to_remove)

        if modified:
            self._update_opset_version()
            self.model = shape_inference.infer_shapes(self.model, strict_mode=True)

    def _match_layernorm_pattern(self, mean_node: onnx.NodeProto) -> dict | None:
        """Match the sequence of operations that constitute a LayerNorm.

        Args:
            mean_node: The ReduceMean node to start pattern matching from.

        Returns:
            Dict | None: Pattern information if matched, None otherwise.
        """
        try:
            sub_node = utils.get_unique_consumer_node(self.model, mean_node.output[0])
            if sub_node.op_type != "Sub":
                return None

            # Find variance computation branch
            pow_nodes = [
                n
                for n in utils.get_consumer_nodes(self.model, sub_node.output[0])
                if n.op_type == "Pow"
            ]
            if len(pow_nodes) != 1:
                return None
            pow_node = pow_nodes[0]
            pow_of_value = self._get_initializer_value(pow_node.input[1])
            if pow_of_value != 2:
                return None

            var_mean_node = utils.get_unique_consumer_node(self.model, pow_node.output[0])
            if var_mean_node.op_type != "ReduceMean":
                return None

            add_eps_node = utils.get_unique_consumer_node(self.model, var_mean_node.output[0])
            if add_eps_node.op_type != "Add":
                return None

            sqrt_node = utils.get_unique_consumer_node(self.model, add_eps_node.output[0])
            if sqrt_node.op_type != "Sqrt":
                return None

            # Find Div node
            # Find the Div node that consumes both sqrt and sub outputs
            sqrt_consumers = utils.get_consumer_nodes(self.model, sqrt_node.output[0])
            sub_consumers = utils.get_consumer_nodes(self.model, sub_node.output[0])

            div_nodes = [n for n in sqrt_consumers if n in sub_consumers and n.op_type == "Div"]
            if len(div_nodes) != 1:
                div_node = None
            else:
                div_node = div_nodes[0]
                if (
                    div_node.input[0] != sub_node.output[0]
                    or div_node.input[1] != sqrt_node.output[0]
                ):
                    div_node = None
            if not div_node:
                return None

            # Get epsilon value
            epsilon = self._get_initializer_value(add_eps_node.input[1])
            if epsilon is None:
                logger.warning(
                    f"Could not find epsilon value for LayerNorm pattern starting at {mean_node.name}"
                )
                return None

            # Check for scale/bias
            # Find and extract scale and bias nodes if present
            scale = None
            bias = None
            final_node = div_node
            nodes_to_remove = []

            consumers = utils.get_consumer_nodes(self.model, div_node.output[0])
            if len(consumers) == 1 and consumers[0].op_type == "Mul":
                mul_node = consumers[0]
                scale = self._get_initializer_value(mul_node.input[1], return_array=True)
                final_node = mul_node
                nodes_to_remove.append(mul_node)

                consumers = utils.get_consumer_nodes(self.model, mul_node.output[0])
                if len(consumers) == 1 and consumers[0].op_type == "Add":
                    add_node = consumers[0]
                    bias = self._get_initializer_value(add_node.input[1], return_array=True)
                    final_node = add_node
                    nodes_to_remove.append(add_node)
            elif len(consumers) == 1 and consumers[0].op_type == "Add":
                # just bias, no scale
                add_node = consumers[0]
                bias = self._get_initializer_value(add_node.input[1], return_array=True)
                final_node = add_node
                nodes_to_remove.append(add_node)

            scale_bias = {
                "scale": scale,
                "bias": bias,
                "final_node": final_node,
                "nodes": nodes_to_remove,
            }

            # Get input shape from value_info or graph input
            input_shape = None
            for vi in list(self.model.graph.value_info) + list(self.model.graph.input):
                if vi.name == sub_node.input[0]:
                    input_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                    break

            if input_shape is None:
                logger.warning(
                    f"Could not determine input shape for LayerNorm pattern at {mean_node.name}"
                )
                return None

            return {
                "mean_node": mean_node,
                "input_name": sub_node.input[0],
                "input_shape": input_shape,
                "output_name": scale_bias["final_node"].output[0],
                "epsilon": epsilon,
                "scale": scale_bias["scale"],
                "bias": scale_bias["bias"],
                "nodes_to_remove": [
                    mean_node,
                    sub_node,
                    pow_node,
                    var_mean_node,
                    add_eps_node,
                    sqrt_node,
                    div_node,
                ]
                + scale_bias["nodes"],
            }

        except Exception as e:
            logger.debug(f"Failed to match LayerNorm pattern at {mean_node.name}: {e!s}")
            return None

    def _create_layernorm_node(self, pattern: dict) -> tuple[onnx.NodeProto, dict]:
        """Create a LayerNormalization node with optional bias."""
        ln_name = f"LayerNorm_{pattern['mean_node'].name}"
        scale_name = f"{ln_name}_scale"
        bias_name = f"{ln_name}_bias" if pattern["bias"] is not None else ""
        axis = pattern["mean_node"].attribute[0].ints[0]

        # Always create scale tensor, default to ones if not provided
        scale_tensor = (
            pattern["scale"]
            if pattern["scale"] is not None
            else np.ones(pattern["input_shape"][axis], dtype=np.float32)
        )
        self.model.graph.initializer.append(numpy_helper.from_array(scale_tensor, name=scale_name))

        if pattern["bias"] is not None:
            self.model.graph.initializer.append(
                numpy_helper.from_array(pattern["bias"], name=bias_name)
            )

        inputs = [pattern["input_name"], scale_name]
        if pattern["bias"] is not None:
            inputs.append(bias_name)

        return helper.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=[pattern["output_name"]],
            name=ln_name,
            epsilon=pattern["epsilon"],
            # Axis is determined by the axes parameter of the ReduceMean node in the pattern
            # The pattern matching code ensures this is always [-1] for LayerNorm
            axis=axis,
        )

    def _find_insertion_point(self, input_name: str) -> int:
        """Find the correct insertion point for the new LayerNorm node."""
        producer_nodes = utils.get_producer_nodes(self.model, input_name)
        if not producer_nodes:
            return 0

        producer_indices = [i for i, n in enumerate(self.model.graph.node) if n in producer_nodes]
        return max(producer_indices) + 1

    def _remove_nodes(self, nodes_to_remove: list[onnx.NodeProto]) -> None:
        """Remove replaced nodes and their corresponding value_info entries."""
        tensors_to_remove = set()
        for node in nodes_to_remove:
            tensors_to_remove.update(node.output)
            if node in self.model.graph.node:
                self.model.graph.node.remove(node)

        value_info_to_remove = [
            vi for vi in self.model.graph.value_info if vi.name in tensors_to_remove
        ]
        for vi in value_info_to_remove:
            self.model.graph.value_info.remove(vi)

    def _update_opset_version(self) -> None:
        """Update the model's opset version to support LayerNormalization."""
        current_opset = None
        for opset in self.model.opset_import:
            if opset.domain in {"", "ai.onnx"}:
                current_opset = opset.version
                break

        if current_opset is None or current_opset < 17:
            # Remove existing default domain opsets
            default_opsets = [op for op in self.model.opset_import if op.domain in ("", "ai.onnx")]
            for op in default_opsets:
                self.model.opset_import.remove(op)
            # Add opset 17
            self.model.opset_import.append(helper.make_opsetid("", 17))
            logger.info(
                f"Updated model opset to 17 for LayerNormalization support (was {current_opset})"
            )

    def _get_initializer_value(self, name: str, return_array: bool = False) -> np.ndarray | None:
        """Get value from an initializer by name."""
        for init in self.model.graph.initializer:
            if init.name == name:
                value = numpy_helper.to_array(init)
                return value if return_array else value.item()
        return None
