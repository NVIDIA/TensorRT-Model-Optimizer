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

"""Node classification module for AutoCast.

This module provides classes for classifying ONNX nodes based on various rules
to determine which nodes should be converted to lower precision and which should
remain in high precision. It includes rules for handling node names, operation types,
initializer ranges, and I/O value ranges.
"""

import abc
import re

import numpy as np
import onnx

from modelopt.onnx.autocast.logging_config import configure_logging, logger
from modelopt.onnx.autocast.referencerunner import ReferenceRunner

configure_logging()


class NodeRuleBase:
    """Base class for node classification rules.

    This class defines the interface for rules that determine whether a node
    should be kept in high precision or converted to low precision.
    """

    @abc.abstractmethod
    def _check_inner(self, node):
        """Implement this method to check if node conversion should be skipped based on rule criteria."""

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes."""
        logger.info(f"Skipping node {node.name}: {self.__class__.__name__}")

    def check(self, node):
        """Check if a node should be skipped based on the rule.

        Args:
            node: The ONNX node to check.

        Returns:
            bool: True if the node should be kept in high precision, False otherwise.
        """
        result = self._check_inner(node)
        if result:
            self._log_skipped(node)
            return True
        return False


class DisabledNodeNameRegexRule(NodeRuleBase):
    """Rule for keeping nodes with matching names in high precision."""

    def __init__(self, disabled_node_name_regex):
        """Initialize the rule.

        Args:
            disabled_node_name_regex: List of regex patterns for node names to keep in high precision.
        """
        self.disabled_node_name_regex = disabled_node_name_regex

    def _check_inner(self, node):
        return any(re.match(regex, node.name) for regex in self.disabled_node_name_regex)


class DisabledOpTypes(NodeRuleBase):
    """Rule for keeping nodes with specific operation types in high precision."""

    def __init__(self, op_types_to_exclude):
        """Initialize the rule.

        Args:
            op_types_to_exclude: List of operation types to keep in high precision.
        """
        self.op_types_to_exclude = op_types_to_exclude

    def _check_inner(self, node):
        return node.op_type in self.op_types_to_exclude


class InitializerRangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range initializers in high precision."""

    def __init__(self, init_max, node_to_init_map):
        """Initialize the rule.

        Args:
            init_max: Maximum absolute value allowed for initializers.
            node_to_init_map: Mapping from node names to their initializers.
        """
        self.init_max = init_max
        self.node_to_init_map = node_to_init_map
        self.init_data = (None, None)

    def _check_inner(self, node):
        for init in self.node_to_init_map[node.name]:
            np_array = onnx.numpy_helper.to_array(init)
            if np_array.dtype == np.float32 and np.any(np.abs(np_array) > self.init_max):
                self.init_data = init.name, np_array
                return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with initializer range violations."""
        if self.init_data[1] is not None:
            logger.info(
                f"Skipping node {node.name}: initializer {self.init_data[0]} out of range: "
                f"min={self.init_data[1].min()}, max={self.init_data[1].max()}, range=[{-self.init_max},"
                f"{self.init_max}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class IORangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range inputs/outputs in high precision."""

    def __init__(self, data_max, reference_data, node_to_init_map):
        """Initialize the rule.

        Args:
            data_max: Maximum absolute value allowed for node I/O.
            reference_data: Reference data for checking I/O ranges.
            node_to_init_map: Mapping from node names to their initializers.
        """
        self.data_max = data_max
        self.reference_data = reference_data
        self.node_to_init_map = node_to_init_map
        self.output_data = None

    def _check_inner(self, node):
        def is_io_out_of_range(node, tensor_name):
            if tensor_name not in self.reference_data:
                # Issue a warning only if the tensor is not an initializer/network input
                init_names = [init.name for init in self.node_to_init_map.get(node.name, [])]
                init_names.extend(node.input)
                if tensor_name not in init_names:
                    logger.warning(
                        f"Node {node.name}: Tensor {tensor_name} not found in reference data."
                    )
                return False
            ref_data = self.reference_data[tensor_name]
            if ref_data.size == 0:
                logger.debug(
                    f"Node {node.name}: Tensor {tensor_name} has size 0. Skipping I/O range check."
                )
                return False
            logger.debug(
                f"Node {node.name}: reference data: min={np.min(ref_data)}, max={np.max(ref_data)}"
            )
            if np.any(np.abs(ref_data) > self.data_max):
                self.output_data = ref_data
                return True

        if node.op_type == "Constant":
            return False
        if self.reference_data:
            for in_name in node.input:
                if is_io_out_of_range(node, in_name):
                    return True
            for out_name in node.output:
                if is_io_out_of_range(node, out_name):
                    return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with I/O range violations."""
        if self.output_data is not None:
            logger.info(
                f"Skipping node {node.name}: reference IO out of range: min={np.min(self.output_data)}, "
                f"max={np.max(self.output_data)}, range=[{-self.data_max}, {self.data_max}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class NodeClassifier:
    """Main class for classifying nodes into high and low precision groups."""

    def __init__(
        self,
        model,
        node_to_init_map,
        nodes_to_exclude=None,
        op_types_to_exclude=[],
        custom_rule=None,
        data_max=1000,
        init_max=np.finfo(np.float16).max,
    ):
        """Initialize the node classifier.

        Args:
            model: The ONNX model to classify nodes for.
            node_to_init_map: Mapping from node names to their initializers.
            nodes_to_exclude: List of regex patterns for node names to keep in high precision.
            op_types_to_exclude: List of operation types to keep in high precision.
            custom_rule: Optional custom classification rule.
            data_max: Maximum absolute value allowed for node I/O.
            init_max: Maximum absolute value allowed for initializers.
        """
        self.model = model
        self.node_to_init_map = node_to_init_map
        self.nodes_to_exclude = nodes_to_exclude
        self.op_types_to_exclude = op_types_to_exclude
        self.custom_rule = custom_rule
        self.data_max = data_max
        self.init_max = init_max

    def _gen_block_node_rules(self, reference_data):
        """Generate list of rules for blocking nodes from precision conversion.

        Args:
            reference_data: Reference data for checking I/O ranges.

        Returns:
            list[NodeRuleBase]: List of rules to apply.
        """
        block_node_rules: list[NodeRuleBase] = []
        if self.nodes_to_exclude:
            block_node_rules.append(DisabledNodeNameRegexRule(self.nodes_to_exclude))
        if self.op_types_to_exclude:
            block_node_rules.append(DisabledOpTypes(self.op_types_to_exclude))
        if self.init_max is not None:
            block_node_rules.append(InitializerRangeRule(self.init_max, self.node_to_init_map))
        if reference_data:
            block_node_rules.append(
                IORangeRule(self.data_max, reference_data, self.node_to_init_map)
            )
        if self.custom_rule:
            block_node_rules.append(self.custom_rule)
        return block_node_rules

    def run(self, calibration_data=None):
        """Run node classification.

        Args:
            calibration_data: Optional input data for reference execution.

        Returns:
            tuple: Lists of node names (low_precision_nodes, high_precision_nodes).
        """
        ref_outputs_dict = None
        if self.data_max is not None and self.data_max != np.inf:
            ref_runner = ReferenceRunner(self.model)
            ref_outputs_dict = ref_runner.run(calibration_data)
        block_node_rules = self._gen_block_node_rules(ref_outputs_dict)
        low_precision_nodes = []
        high_precision_nodes = []
        for node in self.model.graph.node:
            # If any condition is met - node will be executed in high precision
            if any(rule.check(node) for rule in block_node_rules):
                high_precision_nodes.append(node.name)
            else:
                low_precision_nodes.append(node.name)
        logger.debug(f"Low Precision Nodes: {low_precision_nodes}")
        logger.debug(f"High Precision Nodes: {high_precision_nodes}")
        return low_precision_nodes, high_precision_nodes
