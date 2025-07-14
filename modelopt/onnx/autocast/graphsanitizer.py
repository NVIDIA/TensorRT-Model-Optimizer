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
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import logger


class GraphSanitizer:
    """A class for sanitizing ONNX model graphs, a part of the AutoCast tool."""

    def __init__(
        self,
        model: onnx.ModelProto,
        min_opset: int = 13,
        max_ir_version: int | None = None,
        trt_plugins: list[str] | None = [],
    ) -> None:
        """Initialize GraphSanitizer.

        Args:
            model: ONNX model to sanitize
            min_opset: minimum opset version to use
            max_ir_version: maximum IR version supported by ORT
            trt_plugins: list of TensorRT plugin library paths in .so format (compiled shared library).
        """
        self.model = model
        self.min_opset = min_opset
        self.max_ir_version = max_ir_version
        self.standard_ops = {schema.name for schema in onnx.defs.get_all_schemas()}
        self.custom_ops = None
        self.trt_plugins = trt_plugins

    def sanitize(self) -> None:
        """Sanitize the model graph.

        Currently, this finds decomposed LayerNorm patterns and replaces them with a single LayerNormalization operator.
        Additional functionality may be added in the future.
        """
        self.find_custom_nodes()
        self.remove_disconnected_outputs()
        self.convert_opset()
        self.replace_layernorm_pattern()
        self.ensure_graph_name_exists()
        onnx_utils.name_onnx_nodes(self.model.graph)
        self.replace_custom_domain_nodes()
        self.cleanup_model()
        self.set_ir_version(self.max_ir_version)

    def find_custom_nodes(self) -> None:
        """Find custom nodes in the model.

        Scans through all nodes in the graph and logs any nodes that use custom operators
        that are not part of the standard ONNX operator set.
        """
        self.custom_ops = {
            node.op_type for node in self.model.graph.node if node.op_type not in self.standard_ops
        }
        if self.custom_ops:
            from modelopt.onnx.trt_utils import infer_types_shapes_tensorrt, set_trt_plugin_domain

            # Set TensorRT plugin domain info in the graph for ORT compatibility
            self.model = set_trt_plugin_domain(self.model, self.custom_ops)

            # Infer types and shapes in the graph for ORT compatibility
            self.model = infer_types_shapes_tensorrt(self.model, self.trt_plugins)

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

    def convert_opset(self) -> None:
        """Convert the model to the given opset version.

        Args:
            min_opset: minimum opset version to use

        The method checks all opset imports and converts the model if any are below the minimum version.
        """
        # Check all opset imports
        default_opsets = list(self.model.opset_import)

        # Check for quantization nodes and update min_opset if needed
        # Before opset 19, QuantizeLinear and DequantizeLinear data and scale must be fp32
        has_quant_nodes = any(
            node.op_type in ["QuantizeLinear", "DequantizeLinear"] for node in self.model.graph.node
        )
        if has_quant_nodes and self.min_opset < 19:
            logger.warning(
                f"Found QuantizeLinear/DequantizeLinear nodes. Updating minimum opset from {self.min_opset} to 19."
            )
            self.min_opset = 19

        # Convert if any default domain opset is below minimum
        if any(op.version < self.min_opset for op in default_opsets):
            invalid_opsets = [op.version for op in default_opsets if op.version < self.min_opset]
            try:
                self.model = onnx.version_converter.convert_version(self.model, self.min_opset)
            except Exception as e:
                logger.warning(f"Failed to convert model to opset {self.min_opset}: {e!s}")
                logger.warning(f"Attempting to continue with the original opsets: {invalid_opsets}")

    def set_ir_version(self, max_ir_version: int | None) -> None:
        """Set the model's IR version to the maximum supported version.

        Args:
            max_ir_version: maximum IR version to use.

        The method checks the IR version and cuts it off at the maximum supported version.
        See https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support
        """
        if self.custom_ops:
            # Set ir_version to 10, remove it once ORT supports ir_version 11
            self.max_ir_version = 10
            max_ir_version = max_ir_version or self.max_ir_version
        if max_ir_version and self.model.ir_version > max_ir_version:
            try:
                self.model.ir_version = max_ir_version
            except Exception as e:
                logger.warning(f"Failed to set IR version to {max_ir_version}: {e!s}")
                logger.warning(
                    f"Attempting to continue with the original IR version: {self.model.ir_version}"
                )

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
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True)

    def ensure_graph_name_exists(self) -> None:
        """Ensures that the model's name exists."""
        if not self.model.graph.name:
            self.model.graph.name = "model"

    def _match_layernorm_pattern(self, mean_node: onnx.NodeProto) -> dict | None:
        """Match the sequence of operations that constitute a LayerNorm.

        Args:
            mean_node: The ReduceMean node to start pattern matching from.

        Returns:
            Dict | None: Pattern information if matched, None otherwise.
        """
        try:
            axis = mean_node.attribute[0].ints[0]
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
            nodes_to_remove = [
                mean_node,
                sub_node,
                pow_node,
                var_mean_node,
                add_eps_node,
                sqrt_node,
                div_node,
            ]

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

            if scale is not None:
                scale_dimension = scale.shape

            # Skip pattern if we can't determine the scale dimension
            if scale_dimension is None:
                logger.debug(
                    f"Could not determine scale dimension for LayerNorm pattern at {mean_node.name}"
                )
                return None

            return {
                "mean_node": mean_node,
                "input_name": sub_node.input[0],
                "scale_dimension": scale_dimension,
                "output_name": final_node.output[0],
                "epsilon": epsilon,
                "scale": scale,
                "bias": bias,
                "axis": axis,
                "nodes_to_remove": nodes_to_remove,
            }

        except Exception as e:
            logger.debug(f"Failed to match LayerNorm pattern at {mean_node.name}: {e!s}")
            return None

    def _create_layernorm_node(self, pattern: dict) -> onnx.NodeProto:
        """Create a LayerNormalization node with optional bias."""
        ln_name = f"LayerNorm_{pattern['mean_node'].name}"
        scale_name = f"{ln_name}_scale"
        bias_name = f"{ln_name}_bias" if pattern["bias"] is not None else ""
        axis = pattern["axis"]

        # Always create scale tensor, default to ones if not provided
        scale_tensor = (
            pattern["scale"]
            if pattern["scale"] is not None
            else np.ones(pattern["scale_dimension"], dtype=np.float32)
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

    def cleanup_model(self) -> None:
        """Use GraphSurgeon to cleanup unused nodes, tensors and initializers."""
        gs_graph = gs.import_onnx(self.model)
        gs_graph.cleanup()
        self.model = gs.export_onnx(gs_graph)

    def replace_custom_domain_nodes(self):
        """Replace custom domain nodes with standard ONNX nodes."""
        for node in self.model.graph.node:
            if node.domain.startswith("com.microsoft") and node.op_type in self.standard_ops:
                logger.warning(
                    f"Replacing custom domain node {node.name}, domain {node.domain} with standard ONNX "
                    f"node {node.op_type}"
                )
                node.domain = ""
