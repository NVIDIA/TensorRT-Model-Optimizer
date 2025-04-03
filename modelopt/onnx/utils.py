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
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import onnx
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx_graphsurgeon as gs
from onnx import numpy_helper
from onnx_graphsurgeon import Constant, Node, Variable


def get_input_names_from_bytes(model_bytes: bytes, external_inputs_only: bool = True) -> list[str]:
    """This function returns the inputs names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of input names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_input_names(model, external_inputs_only)


def get_all_input_names(model: onnx.onnx_ml_pb2.ModelProto) -> list[str]:
    """This function returns the inputs names of the given onnx model."""
    return [graph_input.name for graph_input in model.graph.input]


def _get_initializer_names(model: onnx.onnx_ml_pb2.ModelProto) -> list[str]:
    return [initializer.name for initializer in model.graph.initializer]


def get_input_names(
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> list[str]:
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


def get_output_names(model: onnx.onnx_ml_pb2.ModelProto) -> list[str]:
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


def get_node_names(model: onnx.onnx_ml_pb2.ModelProto) -> list[str]:
    """This function returns all node names from the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of node names of the model.
    """
    return [node.name for node in model.graph.node]


def _get_tensor_shape(tensor: onnx.onnx_ml_pb2.ValueInfoProto) -> list[int]:
    """This function returns the shape of the input onnx tensor.

    Onnx tensors are of type ValueInfoProto and their dimensions are stored in a
    RepeatedCompositeContainer. Each of these dimensions is of type onnx.onnx_ml_pb2.Dimension.
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
        inp
        for inp in graph.inputs
        if -1 in inp.shape or any([isinstance(s, str) for s in inp.shape])
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
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> dict[str, list[int]]:
    """This function returns the inputs shapes for the given onnx model."""
    if external_inputs_only:
        return _get_selected_shapes(model.graph.input, get_input_names(model))
    return _get_all_shapes(model.graph.input)


def get_output_shapes(model: onnx.onnx_ml_pb2.ModelProto) -> dict[str, list[int]]:
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


def _get_tensor_type(tensor: onnx.onnx_ml_pb2.ValueInfoProto) -> int:
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")
    type_ = tensor.type.tensor_type.elem_type
    return type_


def _get_container_types(
    container, inputs_to_include: Union[None, list[str]] = None
) -> dict[str, int]:
    results = {}
    for tensor in container:
        if inputs_to_include is not None:
            if tensor.name not in inputs_to_include:
                continue
        t_type = _get_tensor_type(tensor)
        results[tensor.name] = t_type
    return results


def _get_input_types(
    model: onnx.onnx_ml_pb2.ModelProto, external_inputs_only: bool = True
) -> dict[str, int]:
    inputs_to_include = get_input_names(model, external_inputs_only)
    return _get_container_types(model.graph.input, inputs_to_include)


def _get_output_types(model: onnx.onnx_ml_pb2.ModelProto) -> dict[str, int]:
    results = _get_container_types(model.graph.output)
    return results


def _convert_types_to_np(types: Union[dict[str, int], list[int], int]) -> Any:
    if isinstance(types, dict):
        types_np = {}
        for name in types.keys():
            types_np[name] = onnx.helper.tensor_dtype_to_np_dtype(types[name])
        return types_np
    elif isinstance(types, list):
        return [onnx.helper.tensor_dtype_to_np_dtype(type_) for type_ in types]
    else:
        return onnx.helper.tensor_dtype_to_np_dtype(types)


def gen_random_inputs(
    model: onnx.onnx_ml_pb2.ModelProto, shapes_spec: str = None
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
    input_dict = {}
    types = _get_input_types(model)
    types_np = _convert_types_to_np(types)
    input_shapes = {} if shapes_spec is None else parse_shapes_spec(shapes_spec)

    for graph_input in model.graph.input:
        # Generate tensors for external inputs only
        if graph_input.name not in types_np:
            continue

        shape_arr = (
            input_shapes[graph_input.name]
            if graph_input.name in input_shapes
            else _get_tensor_shape(graph_input)
        )

        input_dict[graph_input.name] = np.random.uniform(size=shape_arr).astype(
            types_np[graph_input.name]
        )

    return input_dict


def remove_weights_data(onnx_bytes: bytes) -> bytes:
    """Removes raw weight data from the onnx model."""
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer

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
    if not onnx_bytes:
        return False

    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        return onnx_model is not None
    except Exception:
        return False


def validate_batch_size(onnx_bytes: bytes, batch_size: int) -> bool:
    """Returns True if all the model inputs has batch dimension equal to batch_size."""
    input_shapes = list(get_input_shapes_from_bytes(onnx_bytes).values())
    for shape in input_shapes:
        if shape[0] != batch_size:
            return False

    return True


def get_batch_size(model: onnx.onnx_ml_pb2.ModelProto) -> int:
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
        print(f"Onnx model saved as {file_path}")
    except Exception as e:
        print(f"Onnx model exporting as {file_path} failed, error {str(e)}")


def name_onnx_nodes(graph: onnx.onnx_ml_pb2.GraphProto) -> bool:
    """Assigns name to the onnx nodes if not present and return the modified status."""
    is_modified = False
    node_names = set([node.name for node in graph.node])
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
            if isinstance(tensor, Constant):
                if len(tensor.outputs) > 1:  # constant is shared across multiple nodes
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
    onnx_model.ir_version = 9
    is_modified = True if tensors else False
    return onnx_model, is_modified


def is_valid_onnx_model(file_path):
    """Checks if the given file is a valid ONNX model."""
    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        return False

    try:
        # Load the ONNX model
        model = onnx.load(file_path)

        # Check the model
        onnx.checker.check_model(model)
        print(f"ONNX model at {file_path} is valid.")
        return True
    except C.ValidationError as e:
        print(f"The file is not a valid ONNX model. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return False


def find_lowest_common_ancestor(node1: Node, node2: Node) -> tuple[Optional[str], int, int]:
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
    parents = []
    for tensor in node.inputs:
        # If the tensor is not a constant or graph input and has a producer,
        # the producer is a parent of node `node`
        if len(tensor.inputs) == 1:
            parents.append(tensor.inputs[0])

    return parents


def get_child_nodes(node: Node) -> list[Node]:
    """Returns list of output consumer nodes for the given node."""
    children = []
    for tensor in node.outputs:
        for consumer in tensor.outputs:  # Traverse all consumer of the tensor
            children.append(consumer)

    return children


def get_variable_inputs(node: Node) -> list[Variable]:
    """Returns the variable inputs of the given Node."""
    var_inputs = []
    for tensor in node.inputs:
        if isinstance(tensor, Variable):
            if not tensor.inputs or (tensor.inputs and tensor.inputs[0].op != "Constant"):
                var_inputs.append(tensor)
    return var_inputs


def save_onnx(
    model: onnx.onnx_ml_pb2.ModelProto, onnx_path: str, save_as_external_data: bool = False
):
    """Save an ONNX model to given path. If a model is larger than 2GB, will save with external data."""
    size_threshold = 2 * 1024 * 1024 * 1024
    try:
        model_proto = model.SerializeToString()
        model_size = len(model_proto)
        save_as_external_data = save_as_external_data or model_size > size_threshold

    except ValueError as e:
        if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(e):
            save_as_external_data = True
        else:
            raise

    if save_as_external_data:
        external_data_path = os.path.basename(onnx_path) + "_data"
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


def udpate_domain(
    onnx_model: onnx.onnx_ml_pb2.ModelProto, op_type: str, domain: str
) -> onnx.onnx_ml_pb2.ModelProto:
    """Updates the domain of all the nodes of the specified op_type to the specified domain."""
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            node.domain = domain

    return onnx_model
