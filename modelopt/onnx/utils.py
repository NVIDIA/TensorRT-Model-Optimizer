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

"""Utility functions related to onnx."""

import io
import os
import tempfile
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx.helper import get_attribute_value
from onnx_graphsurgeon import Constant, Node, Variable

from modelopt.onnx.logging_config import logger


def get_input_names_from_bytes(model_bytes: bytes, external_inputs_only: bool = True) -> list[str]:
    """This function returns the inputs names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of input names of the model.
    """
    logger.debug("Getting input names from model bytes")
    model = onnx.load_from_string(model_bytes)
    return get_input_names(model, external_inputs_only)


def get_all_input_names(model: onnx.ModelProto) -> list[str]:
    """This function returns the inputs names of the given onnx model."""
    return [graph_input.name for graph_input in model.graph.input]


def _get_initializer_names(model: onnx.ModelProto) -> list[str]:
    return [initializer.name for initializer in model.graph.initializer]


def get_input_names(model: onnx.ModelProto, external_inputs_only: bool = True) -> list[str]:
    """This function returns the external inputs names of the given onnx model.

    Note: external_input_names = input_names - initializer_names

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of external input names of the model.
    """
    input_names = get_all_input_names(model)
    if not external_inputs_only:
        return input_names

    initializer_names = _get_initializer_names(model)
    external_input_names = list(np.setdiff1d(input_names, initializer_names))
    return external_input_names


def get_output_names_from_bytes(model_bytes: bytes) -> list[str]:
    """This function returns the output names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of output names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_output_names(model)


def get_output_names(model: onnx.ModelProto) -> list[str]:
    """This function returns the output names of the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of output names of the model.
    """
    return [output.name for output in model.graph.output]


def get_node_names_from_bytes(model_bytes: bytes) -> list[str]:
    """This function returns all node names from the given onnx model in bytes.

    Args:
        model: onnx model in bytes.

    Returns:
        List of node names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_node_names(model)


def get_node_names(model: onnx.ModelProto) -> list[str]:
    """This function returns all node names from the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of node names of the model.
    """
    return [node.name for node in model.graph.node]


def _get_tensor_shape(tensor: onnx.ValueInfoProto) -> list[int]:
    """This function returns the shape of the input onnx tensor.

    Onnx tensors are of type ValueInfoProto and their dimensions are stored in a
    RepeatedCompositeContainer. Each of these dimensions is of type onnx.Dimension.
    In a loop we access each of the Dimension object and create a shape list to return.
    Dynamic dimensions (i.e. with "dim_param" field) are replaced with 1.

    Args:
        tensor: Onnx tensor object for which the shape needs to be computed.

    Returns:
        Shape of the input tensor.
    """
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")

    dimensions = tensor.type.tensor_type.shape.dim
    shape = []
    for dim in dimensions:
        if dim.HasField("dim_param"):
            shape.append(1)
        if dim.HasField("dim_value"):
            if dim.dim_value == -1:
                shape.append(1)
            else:
                shape.append(dim.dim_value)

    return shape


def get_dynamic_graph_inputs(onnx_model: onnx.ModelProto):
    """This function returns the dynamic inputs of an ONNX model.

    Args:
        onnx_model: ONNX model to obtain dynamic inputs from.

    Returns:
        List of dynamic inputs.
    """
    graph = gs.import_onnx(onnx_model)
    return [
        inp for inp in graph.inputs if -1 in inp.shape or any(isinstance(s, str) for s in inp.shape)
    ]


def _get_all_shapes(container: Any) -> dict[str, list[int]]:
    """This method returns the shape of tensors within a RepeatedCompositeContainer.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names and shape of the tensors within the container.
    """
    results = {}
    for tensor in container:
        results[tensor.name] = _get_tensor_shape(tensor)
    return results


def _get_selected_shapes(container: Any, inputs_to_include: list[str]) -> dict[str, list[int]]:
    """This method returns the shape tensors within a RepeatedCompositeContainer.

    It only computes the shape of the tensors with name containing in `inputs_to_include` list.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names in inputs_to_include and their shapes.
    """
    results = {}
    for tensor in container:
        if tensor.name in inputs_to_include:
            results[tensor.name] = _get_tensor_shape(tensor)
    return results


def get_input_shapes_from_bytes(model_bytes: bytes) -> dict[str, list[int]]:
    """This function returns the input shapes of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        Dictionary of inputs names and shapes.
    """
    model = onnx.load_from_string(model_bytes)
    return get_input_shapes(model)


def get_input_shapes(
    model: onnx.ModelProto, external_inputs_only: bool = True
) -> dict[str, list[int]]:
    """This function returns the inputs shapes for the given onnx model."""
    logger.debug("Getting input shapes from model")
    if external_inputs_only:
        return _get_selected_shapes(model.graph.input, get_input_names(model))
    return _get_all_shapes(model.graph.input)


def get_output_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    """This function returns the output shapes for the given onnx model."""
    return _get_all_shapes(model.graph.output)


def parse_shapes_spec(shapes_spec: str) -> dict[str, list[int]]:
    """Parse shapes spec and returns them in a dictionary.

    Example shapes spec: input0:1x3x256x256,input1:1x3x128x128
    """
    shapes = shapes_spec.split(",")
    shape_dict = {}

    for shape_def in shapes:
        tensor_name, shape_str = shape_def.rsplit(":", 1)
        shape_list = list(map(int, shape_str.split("x")))
        shape_dict[tensor_name.strip()] = shape_list

    return shape_dict


def _get_tensor_type(tensor: onnx.ValueInfoProto) -> int:
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")
    type_ = tensor.type.tensor_type.elem_type
    return type_


def _get_container_types(container, inputs_to_include: list[str] | None = None) -> dict[str, int]:
    results = {}
    for tensor in container:
        if inputs_to_include is not None and tensor.name not in inputs_to_include:
            continue
        t_type = _get_tensor_type(tensor)
        results[tensor.name] = t_type
    return results


def _get_input_types(model: onnx.ModelProto, external_inputs_only: bool = True) -> dict[str, int]:
    inputs_to_include = get_input_names(model, external_inputs_only)
    return _get_container_types(model.graph.input, inputs_to_include)


def _get_output_types(model: onnx.ModelProto) -> dict[str, int]:
    results = _get_container_types(model.graph.output)
    return results


def _convert_types_to_np(types: dict[str, int] | list[int] | int) -> Any:
    if isinstance(types, dict):
        types_np = {}
        for name in types:
            types_np[name] = onnx.helper.tensor_dtype_to_np_dtype(types[name])
        return types_np
    elif isinstance(types, list):
        return [onnx.helper.tensor_dtype_to_np_dtype(type_) for type_ in types]
    else:
        return onnx.helper.tensor_dtype_to_np_dtype(types)


def gen_random_inputs(
    model: onnx.ModelProto, shapes_spec: str | None = None
) -> dict[str, np.ndarray]:
    """This function generates random inputs for an onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.
        shapes_spec: A string representing the shape of each input tensors. The format is
        "<tensor1>:<d1>x<d2>,<tensor2>:<d1>,...". If the shape is not provided for an input tensor, the shape is
        inferred from the onnx model directly, with all the unknown dims filled with 1.

    Returns:
        Dictionary of numpy tensors.
    """
    logger.info("Generating random inputs for model")
    input_dict = {}
    types = _get_input_types(model)
    types_np = _convert_types_to_np(types)
    input_shapes = {} if shapes_spec is None else parse_shapes_spec(shapes_spec)
    logger.debug(f"Using input shapes: {input_shapes}")

    for graph_input in model.graph.input:
        # Generate tensors for external inputs only
        if graph_input.name not in types_np:
            continue

        shape_arr = (
            input_shapes[graph_input.name]
            if graph_input.name in input_shapes
            else _get_tensor_shape(graph_input)
        )

        target_np_type = types_np[graph_input.name]
        if np.issubdtype(target_np_type, np.integer):
            # For integer types, generate random integers in a representative range
            # Example: if it's int32, maybe -1000 to 1000, or 0 to 1000 if non-negative.
            # This needs to be context-aware or have a sensible default.
            # For token IDs (e.g. input_ids), a typical vocab size might be ~50000
            if "ids" in graph_input.name:  # Heuristic for tokenizers
                min_val, max_val = 0, 50000
            elif "mask" in graph_input.name:  # Heuristic for attention masks (0 or 1)
                min_val, max_val = 0, 2  # np.random.randint excludes high, so use 2 for 0,1
            else:  # General integer case, small range
                min_val, max_val = 0, 100
            input_dict[graph_input.name] = np.random.randint(
                min_val, max_val, size=shape_arr
            ).astype(target_np_type)
        elif np.issubdtype(target_np_type, np.floating):
            # For float types, np.random.uniform() is fine, but ensure a decent range
            # if default (0,1) is not good enough. E.g., np.random.uniform(-1.0, 1.0, size=shape_arr)
            input_dict[graph_input.name] = np.random.uniform(
                low=0.0, high=1.0, size=shape_arr
            ).astype(target_np_type)
        else:  # Fallback for other types (e.g. bool)
            input_dict[graph_input.name] = np.random.uniform(size=shape_arr).astype(target_np_type)

    return input_dict


def remove_weights_data(onnx_bytes: bytes) -> bytes:
    """Removes raw weight data from the onnx model."""
    logger.info("Removing weights data from ONNX model")
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer
    weights_removed = 0

    for idx, init in enumerate(inits):
        # Only remove arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                # Setting up some metadata to randomize the weights later
                np_tensor = np.frombuffer(init.raw_data, dtype=dtype)
                meta = model.metadata_props.add()
                meta.key = init.name + "_avg"
                meta.value = str(np.average(np_tensor))

                meta = model.metadata_props.add()
                meta.key = init.name + "_var"
                meta.value = str(np.var(np_tensor))

                # Note that, onnx.checker will fail due to data cleaning
                # We should not check the model till weights are reassigned
                model.graph.initializer[idx].raw_data = b""
                weights_removed += 1

    logger.debug(f"Removed weights data from {weights_removed} tensors")
    buffer = io.BytesIO()
    onnx.save_model(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def randomize_weights(onnx_path: str) -> None:
    """Assigns random values to the onnx model weights."""
    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()
        onnx_bytes = randomize_weights_onnx_bytes(onnx_bytes)

    with open(onnx_path, "wb") as f:
        # Write the modified onnx model to the same path
        f.write(onnx_bytes)


def randomize_weights_onnx_bytes(onnx_bytes: bytes, seed: int = 0) -> bytes:
    """Assigns random values to the onnx model weights."""
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer
    np.random.seed(seed)
    weight_metadata = {item.key: item.value for item in model.metadata_props}

    for idx, init in enumerate(inits):
        # Randomize only the arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                avg = weight_metadata.get(init.name + "_avg", None)
                var = weight_metadata.get(init.name + "_var", None)
                if avg and var:
                    numpy_array = np.random.normal(float(avg), float(var), size=init.dims).astype(
                        dtype
                    )
                    tensor = numpy_helper.from_array(numpy_array, init.name)
                    model.graph.initializer[idx].CopyFrom(tensor)

    buffer = io.BytesIO()
    onnx.save_model(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def validate_onnx(onnx_bytes: bytes) -> bool:
    """Returns True if the onnx_bytes is valid, else False."""
    logger.info("Validating ONNX model")
    if not onnx_bytes:
        logger.error("Empty ONNX bytes provided")
        return False

    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        return onnx_model is not None
    except Exception:
        return False


def validate_batch_size(onnx_bytes: bytes, batch_size: int) -> bool:
    """Returns True if all the model inputs has batch dimension equal to batch_size."""
    input_shapes = list(get_input_shapes_from_bytes(onnx_bytes).values())
    return all(shape[0] == batch_size for shape in input_shapes)


def get_batch_size(model: onnx.ModelProto) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    input_shapes = list(get_input_shapes(model).values())
    batch_size = input_shapes[0][0]
    for shape in input_shapes:
        if batch_size != shape[0]:
            # The model does not have the batch dimension
            return 1

    return batch_size


def get_batch_size_from_bytes(onnx_bytes: bytes) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    model = onnx.load_from_string(onnx_bytes)
    return get_batch_size(model)


def save_onnx_bytes_to_dir(onnx_bytes: bytes, onnx_dir: str, onnx_name: str) -> None:
    """Saves the onnx bytes to a directory with specified file name."""
    os.makedirs(onnx_dir, exist_ok=True)
    file_path = os.path.join(onnx_dir, onnx_name + ".onnx")

    try:
        with open(file_path, "wb") as f:
            f.write(onnx_bytes)
        logger.info(f"Onnx model saved as {file_path}")
    except Exception as e:
        logger.error(f"Onnx model exporting as {file_path} failed, error {e!s}")


def name_onnx_nodes(graph: onnx.GraphProto) -> bool:
    """Assigns name to the onnx nodes if not present and return the modified status."""
    is_modified = False
    node_names = {node.name for node in graph.node}
    start_id = len(node_names)
    for node in graph.node:
        if not node.name:
            new_name = f"{node.op_type}_{start_id}"
            while new_name in node_names:
                start_id += 1
                new_name = f"{node.op_type}_{start_id}"

            node.name = new_name
            node_names.add(new_name)
            is_modified = True

    return is_modified


def duplicate_shared_constants(onnx_model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    """Duplicate constant tensors if they are shared."""
    graph = gs.import_onnx(onnx_model)
    name_dict = defaultdict(lambda: 0)

    def _get_unique_name(old_name):
        name_dict[old_name] += 1
        return old_name + "_" + str(name_dict[old_name])

    # Get tensors with shared constant inputs
    tensors = []
    for node in graph.nodes:
        for inp_idx, tensor in enumerate(node.inputs):
            # constant is shared across multiple nodes
            if isinstance(tensor, Constant) and len(tensor.outputs) > 1:
                tensors.append({"tensor": tensor, "inp_node": node, "inp_idx": inp_idx})

    # Duplicate shared tensors
    for tensor_dict in tensors:
        tensor = tensor_dict["tensor"]
        new_tensor = Constant(
            name=_get_unique_name(tensor.name),
            values=tensor.values,
        )
        tensor_dict["inp_node"].inputs[tensor_dict["inp_idx"]] = new_tensor

    onnx_model = gs.export_onnx(graph)
    is_modified = bool(tensors)
    return onnx_model, is_modified


def check_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Checks if the given model is valid."""
    if model.ByteSize() > (2 * (1024**3)):  # 2GB limit
        with tempfile.TemporaryDirectory() as temp_dir:
            # ONNX also looks in CWD, so we need to use a unique id
            unique_id = str(uuid.uuid4())[:8]
            onnx_tmp_path = os.path.join(temp_dir, f"model_{unique_id}.onnx")
            save_onnx(model, onnx_tmp_path, save_as_external_data=True)
            onnx.checker.check_model(onnx_tmp_path)
            return onnx.load(onnx_tmp_path)
    else:
        onnx.checker.check_model(model)
        return model


def find_lowest_common_ancestor(node1: Node, node2: Node) -> tuple[str | None, int, int]:
    """Function to find the lowest common ancestor of two nodes.

    Args:
        node1: First node name.
        node2: Second node name.

    Returns:
        LCA node.
        Distance from first node.
        Distance from second node.
    """

    def _find_ancestors(node: Node):
        ancestors = {node.name: 0}
        stack = [(node, 0)]
        while stack:
            cur_node, distance = stack.pop()
            for parent_node in get_parent_nodes(cur_node):
                if parent_node.name not in ancestors:
                    ancestors[parent_node.name] = distance + 1
                    stack.append((parent_node, distance + 1))

        return ancestors

    ancestors1 = _find_ancestors(node1)
    ancestors2 = _find_ancestors(node2)

    # Find the lowest common ancestor
    common_ancestors = set(ancestors1.keys()).intersection(ancestors2.keys())
    if common_ancestors:
        lowest_common_ancestor = common_ancestors.pop()
        distance = ancestors1[lowest_common_ancestor]
        for t in common_ancestors:
            if ancestors1[t] < distance:
                distance = ancestors1[t]
                lowest_common_ancestor = t
        distance1 = ancestors1[lowest_common_ancestor]
        distance2 = ancestors2[lowest_common_ancestor]
        return lowest_common_ancestor, distance1, distance2
    else:
        return None, -1, -1  # No common ancestor found


def get_parent_nodes(node: Node) -> list[Node]:
    """Returns list of input producer nodes for the given node."""
    # If the tensor is not a constant or graph input and has a producer,
    # the producer is a parent of node `node`
    parents = [tensor.inputs[0] for tensor in node.inputs if len(tensor.inputs) == 1]

    return parents


def get_child_nodes(node: Node) -> list[Node]:
    """Returns list of output consumer nodes for the given node."""
    children = [consumer for tensor in node.outputs for consumer in tensor.outputs]
    return children


def get_variable_inputs(node: Node) -> list[Variable]:
    """Returns the variable inputs of the given Node."""
    var_inputs = [
        tensor
        for tensor in node.inputs
        if isinstance(tensor, Variable)
        and (not tensor.inputs or (tensor.inputs and tensor.inputs[0].op != "Constant"))
    ]
    return var_inputs


def save_onnx(model: onnx.ModelProto, onnx_path: str, save_as_external_data: bool = False):
    """Save an ONNX model to given path. If a model is larger than 2GB, will save with external data."""
    size_threshold = 2 * (1024**3)  # 2GB
    try:
        model_proto = model.SerializeToString()
        model_size = len(model_proto)
        save_as_external_data = save_as_external_data or model_size > size_threshold
        logger.debug(
            f"Model size: {model_size} bytes, using external data: {save_as_external_data}"
        )

    except ValueError as e:
        if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(e):
            logger.warning("Model exceeds 2GB limit, switching to external data storage")
            save_as_external_data = True
        else:
            logger.error(f"Failed to serialize model: {e!s}")
            raise

    # Set ir_version to 10, remove it once ORT supports ir_version 11
    model.ir_version = 10

    if save_as_external_data:
        external_data_path = os.path.basename(onnx_path) + "_data"
        if os.path.exists(external_data_path):
            logger.warning(f"Removing existing external data file: {external_data_path}")
            os.remove(external_data_path)

        onnx.save_model(
            model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
            size_threshold=1024,
        )
    else:
        onnx.save(model, onnx_path)


def udpate_domain(onnx_model: onnx.ModelProto, op_type: str, domain: str) -> onnx.ModelProto:
    """Updates the domain of all the nodes of the specified op_type to the specified domain."""
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            node.domain = domain

    return onnx_model


def bfloat16_to_float32(bf16_array):
    """Converts a bfloat16 array (as raw data) to a float32 array."""
    uint32_array = bf16_array.astype(np.uint32) << 16
    return uint32_array.view(np.float32)


def read_f16_tensor_as_fp32(tensor):
    """Reads a float16 or bfloat16 tensor as a float32 numpy ndarray."""
    if tensor.data_type == onnx.TensorProto.BFLOAT16:
        raw_data = tensor.raw_data
        uint16_array = np.frombuffer(raw_data, dtype=np.uint16)
        float32_array = bfloat16_to_float32(uint16_array)
        tensor_shape = tuple(dim for dim in tensor.dims)
        return float32_array.reshape(tensor_shape)

    # Read FLOAT16 tensor and return
    return onnx.numpy_helper.to_array(tensor).astype(np.float32)


def has_attribute(node: onnx.NodeProto, attr_name: str) -> bool:
    """Checks if the given node has the specified attribute."""
    return any(attr.name == attr_name for attr in node.attribute)


def get_attribute(node: onnx.NodeProto, attr_name: str) -> Any:
    """Returns the value of the specified attribute."""
    for attr in node.attribute:
        if attr.name == attr_name:
            return get_attribute_value(attr)
    raise ValueError(f"Attribute {attr_name} not found in node {node.name}")


def infer_shapes(model: onnx.ModelProto, **kwargs):
    """Infers shapes of the onnx graph, handles large models."""
    if model.ByteSize() > (2 * (1024**3)):  # 2GB limit
        with tempfile.TemporaryDirectory() as temp_dir:
            # ONNX also looks in CWD, so we need to use a unique id
            unique_id = str(uuid.uuid4())[:8]
            onnx_orig_path = os.path.join(temp_dir, f"model_{unique_id}.onnx")
            onnx_inferred_path = os.path.join(temp_dir, f"inferred_{unique_id}.onnx")
            save_onnx(model, onnx_orig_path, save_as_external_data=True)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path, **kwargs)
            model = onnx.load(onnx_inferred_path)
        return model
    else:
        return onnx.shape_inference.infer_shapes(model, **kwargs)


def onnx_type_str_to_enum(dtype: str) -> int:
    """Converts ONNX type in string format to onnx.TensorProto format.

    Example: 'tensor(float16)' becomes onnx.TensorProto.FLOAT16

    Args:
        dtype: ONNX type in string format.

    Returns:
        int: ONNX type in enum format.
    """
    dtype = dtype.split("tensor(")[-1].split(")")[0]
    dtype = "FLOAT" if dtype == "float32" else dtype.upper()
    return getattr(onnx.TensorProto, dtype)
