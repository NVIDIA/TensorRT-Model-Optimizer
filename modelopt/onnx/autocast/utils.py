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

"""Utility functions for AutoCast.

This module provides common utility functions used across the AutoCast package.
It includes functions for graph traversal, tensor type inference, model validation,
and mapping setup between nodes, initializers, and value info. These utilities
support the core functionality of model precision conversion.
"""

import logging
from collections import defaultdict

import onnx


def setup_mappings(model: onnx.ModelProto) -> tuple[dict, dict, dict]:
    """Setup and return mappings for model components.

    Args:
        model: ONNX model to create mappings for.

    Returns:
        Tuple containing:
        - value_info_map: Mapping of names to value infos.
        - initializer_map: Mapping of names to initializers.
        - node_to_init_map: Mapping of node names to their initializer inputs.
    """
    value_info_map = {}
    for container in (model.graph.value_info, model.graph.input, model.graph.output):
        value_info_map.update((vi.name, vi) for vi in container)

    initializer_map = {init.name: init for init in model.graph.initializer}

    node_to_init_map = {
        node.name: [
            initializer_map[input_name]
            for input_name in node.input
            if input_name in initializer_map
        ]
        for node in model.graph.node
    }

    return value_info_map, initializer_map, node_to_init_map


def get_consumer_nodes(model: onnx.ModelProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Get all consumer nodes for a given tensor name.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find consumers for.

    Returns:
        list[onnx.NodeProto]: List of nodes that consume the tensor.
    """
    return [n for n in model.graph.node if tensor_name in n.input]


def get_producer_nodes(model: onnx.ModelProto, tensor_name: str) -> list[onnx.NodeProto]:
    """Get all producer nodes for a given tensor name.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find producers for.

    Returns:
        list[onnx.NodeProto]: List of nodes that produce the tensor.
    """
    return [n for n in model.graph.node if tensor_name in n.output]


def get_unique_consumer_node(model: onnx.ModelProto, tensor_name: str) -> onnx.NodeProto:
    """Get a single consumer node and raise exception if there are multiple consumers.

    Args:
        model: The ONNX model to search.
        tensor_name: Name of the tensor to find consumer for.

    Returns:
        onnx.NodeProto: The single consumer node.

    Raises:
        Exception: If there is not exactly one consumer node.
    """
    consumers = get_consumer_nodes(model, tensor_name)
    if len(consumers) != 1:
        raise Exception(f"Expected single consumer for {tensor_name}, found {len(consumers)}")
    return consumers[0]


def get_cast_to_type(cast_node: onnx.NodeProto) -> int:
    """Get the target type from a Cast node.

    Args:
        cast_node: The Cast node to extract type from.

    Returns:
        int: The target type value from the Cast node's 'to' attribute.

    Raises:
        ValueError: If the Cast node does not have a 'to' attribute.
    """
    for attr in cast_node.attribute:
        if attr.name == "to":
            return attr.i
    raise ValueError("Cast node does not have 'to' attribute")


def get_ops_without_low_precision_support(
    model: onnx.ModelProto,
    low_precision_type: str,
    min_opset: int,
) -> list[str]:
    """Get a list of ops without low precision support for the current opset version.

    Args:
        model: ONNX model.
        low_precision_type: Target precision to reduce to ('float16' or 'bfloat16').
        min_opset: Minimum opset version.

    Returns:
        ops_without_support: List of ops without low precision support for the current opset version.
    """
    # Obtain the current model's opset version
    ai_onnx_domain = [
        opset
        for opset in model.opset_import
        if not opset.domain or opset.domain in ["ai.onnx", "ai.onnx.contrib", "trt.plugins"]
    ]
    opset_version = max(ai_onnx_domain[0].version, min_opset)

    # Get all ops precision support information
    precision = "tensor(float16)" if low_precision_type == "float16" else "tensor(bfloat16)"
    schemas_dict = defaultdict(dict)
    for schema in onnx.defs.get_all_schemas_with_history():
        float16_supported = False
        for constr in schema.type_constraints:
            if precision in constr.allowed_type_strs:
                float16_supported = True
                break
        schemas_dict[schema.name].update({schema.since_version: float16_supported})

    # Check that all ops are supported in low precision for the current opset version.
    # Otherwise, exclude from conversion.
    ops_without_support = []
    for op, schema in schemas_dict.items():
        supported_opsets = [k for k, v in schema.items() if v]
        if supported_opsets:
            min_opset = min(supported_opsets)
            if min_opset > opset_version:
                ops_without_support.append(op)
        else:
            ops_without_support.append(op)

    if ops_without_support:
        logging.warning(
            f"{len(ops_without_support)} ops are not supported in {low_precision_type} in opset {opset_version}. "
            f"Skipping those from conversion: {ops_without_support}."
        )

    return ops_without_support
