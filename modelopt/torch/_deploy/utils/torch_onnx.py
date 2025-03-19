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

"""Utility functions related to Onnx."""

import base64
import inspect
import json
import os
import shutil
from typing import Any, Union

import onnx
import torch
import torch.nn as nn
from onnx import ModelProto
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelopt.onnx.quantization.qdq_utils import qdq_to_dq
from modelopt.onnx.utils import (
    get_input_names,
    get_input_shapes,
    get_node_names,
    get_output_names,
    get_output_shapes,
)
from modelopt.torch.utils import flatten_tree, standardize_named_model_args
from modelopt.torch.utils._pytree import TreeSpec

from ..utils.onnx_optimizer import Optimizer
from .onnx_utils import _get_onnx_external_data_tensors, check_model_uses_external_data

ModelMetadata = dict[str, Any]
ModelType = Any
ValueInfoType = Any

# a few constants...
DEFAULT_ONNX_OPSET = 13
ONNX_EXPORT_OUT_PREFIX = "out"
TWO_GB = 2 * 1024 * 1024 * 1024


class OnnxBytes:
    """A class to save and load onnx models as bytes."""

    def __init__(self, onnx_load_path: str) -> None:
        """Loads the model from the specified path.

        If the model is loaded without external data format, then it is saved as a dictionary where
        the key is the model name and the value is the model bytes.
        If the model is loaded with external data format, then the model is saved as a dictionary
        where the keys include all the file names in the model directory and the value are the corresponding file bytes.
        For external data format, we assume that the external data for the model is saved in the same directory
        as the model file.

        Args:
            onnx_load_path: The path to load the .onnx model file.
        """
        self.onnx_load_path = os.path.abspath(onnx_load_path)
        self.onnx_model = {}
        self.model_name = ""
        print("Loading onnx model from path:", self.onnx_load_path)
        onnx_model = onnx.load(self.onnx_load_path, load_external_data=False)

        # Check for external data
        external_data_format = False
        for initializer in onnx_model.graph.initializer:
            if initializer.external_data:
                external_data_format = True

        if external_data_format:
            onnx_model_dir = os.path.dirname(self.onnx_load_path)
            for onnx_model_file in os.listdir(onnx_model_dir):
                with open(os.path.join(onnx_model_dir, onnx_model_file), "rb") as f:
                    self.onnx_model[onnx_model_file] = f.read()
                if onnx_model_file.endswith(".onnx"):
                    if self.model_name != "":
                        raise ValueError("Multiple onnx files found in the directory")
                    self.model_name = onnx_model_file.replace(".onnx", "")
        else:
            onnx_model_file = os.path.basename(self.onnx_load_path)
            if not onnx_model_file.endswith(".onnx"):
                raise ValueError("The file should be a .onnx file")
            with open(self.onnx_load_path, "rb") as f:
                self.onnx_model[onnx_model_file] = f.read()
            self.model_name = onnx_model_file.replace(".onnx", "")

    def write_to_disk(self, onnx_save_path: str) -> None:
        """Writes the onnx model to the specified path."""
        if os.path.exists(onnx_save_path):
            print(f"Removing existing directory: {onnx_save_path}")
            shutil.rmtree(onnx_save_path)
        os.makedirs(onnx_save_path)
        print("Writing onnx model to path:", onnx_save_path)
        for onnx_model_file, onnx_model_bytes in self.onnx_model.items():
            with open(os.path.join(onnx_save_path, onnx_model_file), "wb") as f:
                f.write(onnx_model_bytes)

    def to_bytes(self) -> bytes:
        """Returns the bytes of the object that can be restored using the OnnxBytes.from_bytes method."""
        serialized_model = {}
        for file_name, file_bytes in self.onnx_model.items():
            serialized_model[file_name] = base64.b64encode(file_bytes).decode("utf-8")

        # Create a dictionary with all necessary attributes
        data = {
            "onnx_load_path": self.onnx_load_path,
            "model_name": self.model_name,
            "onnx_model": serialized_model,
        }

        return json.dumps(data).encode("utf-8")

    def get_onnx_model_file_bytes(self) -> bytes:
        """Returns the bytes of the onnx model file."""
        return self.onnx_model[self.model_name + ".onnx"]

    @classmethod
    def from_bytes(cls, onnx_bytes: bytes) -> "OnnxBytes":
        """Returns the OnnxBytes object from the bytes."""
        data = json.loads(onnx_bytes.decode("utf-8"))

        # Create a new instance without calling __init__ and set the attributes
        instance = cls.__new__(cls)
        instance.onnx_load_path = data["onnx_load_path"]
        instance.model_name = data["model_name"]
        instance.onnx_model = {}
        for file_name, encoded_bytes in data["onnx_model"].items():
            instance.onnx_model[file_name] = base64.b64decode(encoded_bytes)

        return instance


def _to_expected_onnx_type(val: Any) -> Any:
    """Convert the given value to the expected onnx type.

    During the onnx export process, plain numeric types (floats and ints) are converted to torch
    tensors. This function pre-converts the given val to a tensor in case val is a int or float for
    easier handling of such input values during the onnx export process.
    """
    if isinstance(val, (int, float)):
        return torch.tensor(val).to(type(val))
    return val


def generate_onnx_input(
    model_metadata: ModelMetadata, input: Union[Any, tuple], ignore_nesting: bool = False
) -> dict[str, Any]:
    """Generate input for onnx model from model's forward signature and provided input.

    Args:
        model_metadata: The model's metadata.
        input: A tuple of args/kwargs or torch.Tensor feed into the model's ``forward()`` method,
            see :meth:`standardize_model_args() <modelopt.torch.utils.network.standardize_model_args>`
            for more info on the convention.
        ignore_nesting: If True, only the last part of the nested input name will be considered.
            eg. if the input name is x.y.z, only z will be considered.

    Returns:
        Args flattened into one dictionary with serialized keys compatible with provided onnx.

    .. note::

        This function performs a sanity check on the provided input data to filter out args that
        are constants (instead of input nodes) in the onnx graph.


    Some more relevant background of why we want to flatten the input pytree here:

        * In the onnx export process, nested python data structures (like nested lists, tuples,
            dictionaries) are being recursed into until leaf objects corresponding to tensors are
            encountered.

        * This is used to flatten the input in an onnx model to a list of tensors.

        * However, this is a fairly complex process for the user to understand in case their models
            takes a nested data structure. They have to understand how to manually flatten the data
            structure in the *correct* order in order for them to run inference on a device_model or
            onnx model.

        * With this function this additional complexity can be abstracted away from the user.

        * Example: if the original model took ``[x, {"y":y, "z" : [z1,z2]}]`` they can still provide
            this nested data structure instead of the expected onnx input list of ``[x, y, z1, z2]``
            --> flattening and unflattering is handled internally.
    """
    # get named args and set of params where we added default values
    named_args, args_with_default = standardize_named_model_args(model_metadata["signature"], input)

    # retrieve onnx input names
    onnx_input_names = model_metadata["input_onnx_names"]
    input_none_names = model_metadata["input_none_names"]

    # capture flattened names of args from default values
    named_default_args = {k: v for k, v in named_args.items() if k in args_with_default}
    _, tree_spec_default_args = flatten_tree(named_default_args)

    # capture flattened args without default args that do not appear in onnx graph
    values, tree_spec = flatten_tree(named_args)
    if not ignore_nesting:
        flat_kv = {k: v for k, v in zip(tree_spec.names, values)}
    else:
        flat_kv = {k.split(".")[-1]: v for k, v in zip(tree_spec.names, values)}

    # We wanna consider four types of flattened args:
    # 1. Args that appear in the onnx graph
    # 2. Args that are not their default value
    # 3. Args that were provided as None during conversion but are not None right now
    # 4. Args that were provided as None during conversion and are None right now

    args_in_onnx = {k for k in flat_kv if k in onnx_input_names}
    args_not_default = {k for k in flat_kv if k not in tree_spec_default_args.names}
    args_not_none = {k for k, v in flat_kv.items() if k in input_none_names and v is not None}
    args_none = {k for k, v in flat_kv.items() if k in input_none_names and v is None}

    # identify unexpected args from these 4 types
    unexpected_args = ((args_not_default - args_none) | args_not_none) - args_in_onnx
    if unexpected_args:
        print(
            "The following args were provided that do not appear in the onnx graph of your model "
            "since they are treated as constants in the onnx graph:"
            + "\t\n".join(unexpected_args)
            + "\nConsider removing these args from your input that are constants in the onnx model "
            "or recompiling your onnx model with new constant values!"
        )

    # return the args that are relevant for the onnx graph in the right type
    return {k: _to_expected_onnx_type(v) for k, v in flat_kv.items() if k in args_in_onnx}


def optimize(name, onnx_graph, verbose=False):
    """Optimizes onnx graph."""
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    # TODO: fold constants is not working for some models from deploy_models(NestedOutModel, ArgsKwargsModel1)
    # opt.fold_constants()
    # opt.info(name + ": fold_constants")
    onnx_graph = opt.infer_shapes(return_onnx=True)
    opt.info(name + ": shape inference")
    return onnx_graph


def split_args_kwargs(args_tuple):
    """Splits args_tuple into positional arguments and keyword arguments."""
    split_index = len(args_tuple)

    for i, item in enumerate(reversed(args_tuple)):
        if not isinstance(item, dict):
            split_index = len(args_tuple) - i
            break

    pos_args = args_tuple[:split_index]
    kw_args = {}
    for d in args_tuple[split_index:]:
        kw_args.update(d)

    kw_args = None if kw_args == {} else kw_args

    # remove empty dict if it is the last element
    if pos_args[-1] == {}:
        pos_args = pos_args[:-1]

    return pos_args, kw_args


def get_onnx_bytes_and_metadata(
    model: nn.Module,
    dummy_input: Union[Any, tuple],
    onnx_load_path: str = "",
    dynamic_axes: dict = {},
    remove_exported_model: bool = True,
    dynamo_export: bool = False,
    onnx_opset: int = DEFAULT_ONNX_OPSET,
    dq_only: bool = False,
) -> tuple[bytes, ModelMetadata]:
    """Get onnx model in bytes from input pytorch model together with the input/output of model.

    Arguments:
        model: PyTorch model to export to onnx.
        dummy_input: A tuple of args/kwargs or torch.Tensor, see
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info on the convention.
        onnx_load_path: The path to load the onnx model.
        dynamic_axes: A dictionary of dynamic shapes used for exporting the torch model to onnx.
        remove_exported_model: If True, the onnx model will be cleared from the disk after the
            export process
        dynamo_export: If True, the model is exported using dynamo_export. See
            `torch.onnx.dynamo_export https://pytorch.org/docs/stable/onnx_dynamo.html` for more details.
        onnx_opset: The onnx opset version to use for exporting the model.
        dq_only: If True, the exported ONNX model is converted to a dq_only model.

    Returns:
        bytes: Onnx model in bytes.
        ModelMetadata: The model's meta data.

    Raises:
        ValueError: If nn.Module is not passed as model.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("Only PyTorch model compilation is supported.")

    # unwrap DDP and DP models
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    # Standardize model args and also tensorize them so they also appear in the onnx graph!
    # Floats/ints are tensorized when they are provided, but not tensorized when they are not
    # provided which is somewhat inconsistent (we always tensorize them!)
    named_args, _ = standardize_named_model_args(model, dummy_input)
    named_args = {k: _to_expected_onnx_type(v) for k, v in named_args.items()}

    # Also standardize dummy_input again so we can use it
    dummy_input = tuple(named_args.values())
    if dummy_input and isinstance(dummy_input[-1], dict):
        dummy_input = dummy_input + ({},)  # we need to add an extra dict for the fake kwargs!

    dummy_input_args, dummy_input_kwargs = (), None
    if dynamo_export:
        dummy_input_args, dummy_input_kwargs = split_args_kwargs(dummy_input)

    # Get input tree spec, see generate_onnx_input for more info as well on this
    flat_input, tree_spec_input = flatten_tree(named_args)

    # input names are the names of the flattened input tree spec but without None values
    input_names = [k for k, v in zip(tree_spec_input.names, flat_input) if v is not None]

    # we also want to record the input names that are None so we can remove them from the input
    # during inference.
    input_none_names = list(set(tree_spec_input.names) - set(input_names))

    # Get output once (we export in inference mode - so also using inference mode here!)
    with torch.inference_mode():
        # We enable mixed precision for cases where some outputs may have a different data type than others
        with torch.autocast("cuda"):
            output = model(*named_args.values())

    # Get output tree spec
    flat_output, tree_spec_output = flatten_tree(output, prefix=ONNX_EXPORT_OUT_PREFIX)

    # output names are the names of the flattened input tree spec but without None values
    output_names = [k for k, v in zip(tree_spec_output.names, flat_output) if v is not None]

    model_name = model.__class__.__name__
    onnx_build_folder = os.path.abspath("./build/onnx/")
    onnx_path = os.path.join(onnx_build_folder, model_name)
    os.makedirs(onnx_path, exist_ok=True)
    onnx_save_path = os.path.join(onnx_path, f"{model_name}.onnx")

    # If the onnx_load path is specified by the user or if an onnx model exists at the default path
    # then the model is loaded from this path and returned along with the metadata
    if os.path.exists(onnx_save_path) and onnx_load_path == "":
        print(f"Overriding onnx load path to {onnx_save_path}")
        onnx_load_path = onnx_save_path

    if onnx_load_path != "":
        onnx_model = OnnxBytes(onnx_load_path)
        onnx_model_graph = onnx.load(os.path.join(onnx_load_path))
        model_metadata = create_model_metadata(
            tree_spec_input, tree_spec_output, input_none_names, onnx_model_graph, model
        )
        return onnx_model.to_bytes(), model_metadata

    # Export onnx model from pytorch model
    # As the maximum size of protobuf is 2GB, we cannot use io.BytesIO() buffer during export.
    with torch.inference_mode():
        if not dynamo_export:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_save_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=onnx_opset,
                dynamic_axes=dynamic_axes,
            )
        else:
            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            if dummy_input_kwargs:
                torch.onnx.dynamo_export(
                    model, *dummy_input_args, export_options=export_options, **dummy_input_kwargs
                ).save(onnx_save_path)
            else:
                torch.onnx.dynamo_export(
                    model,
                    *dummy_input_args,
                    export_options=export_options,
                ).save(onnx_save_path)
    # Check that export worked
    assert len(os.listdir(onnx_path)) > 0, "Torch to onnx export failed."

    # Load the onnx graph for optimizaiton
    onnx_graph = onnx.load(onnx_save_path, load_external_data=True)

    # Optimize the onnx graph
    onnx_opt_graph = optimize(model.__class__.__name__, onnx_graph)

    # Remove training_mode attribute from BatchNormalization nodes
    onnx_opt_graph = remove_node_training_mode(onnx_opt_graph, "BatchNormalization")

    model_metadata = create_model_metadata(
        tree_spec_input, tree_spec_output, input_none_names, onnx_opt_graph, model
    )

    # Change the ONNX IR version to 9 to be compatible with ONNXRuntime
    onnx_opt_graph.ir_version = 9

    if dq_only:
        onnx_opt_graph = qdq_to_dq(onnx_opt_graph)

    # If the onnx model contains external data store the external tensors in one file and save the onnx model
    if has_external_data(onnx_save_path):
        tensor_paths = _get_onnx_external_data_tensors(onnx_opt_graph)
        onnx.save_model(
            onnx_opt_graph,
            onnx_save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{model_name}.onnx_data",
            size_threshold=1024,
        )
        for tensor in tensor_paths:
            tensor_path = os.path.join(onnx_path, tensor)
            os.remove(tensor_path)
    else:
        onnx.save_model(onnx_opt_graph, onnx_save_path)

    onnx_bytes = OnnxBytes(onnx_save_path)

    if remove_exported_model:
        shutil.rmtree(os.path.dirname(onnx_build_folder))
    return onnx_bytes.to_bytes(), model_metadata


def has_external_data(onnx_model_path: str):
    """Check if the onnx model has external data."""
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    return check_model_uses_external_data(onnx_model)


def create_model_metadata(
    tree_spec_input: TreeSpec,
    tree_spec_output: TreeSpec,
    input_none_names: list[str],
    onnx_graph: ModelProto,
    model: nn.Module,
) -> ModelMetadata:
    """Create model metadata from the given input.

    Args:
        tree_spec_input: pytree spec describing the structure of the pytree for the model input.
        tree_spec_output: pytree spec describing the structure of the pytree for the model output.
        input_none_names: List of input names with values that are None.
        onnx_opt_graph: Graph of the onnx model.
        model: Pytorch model.

    Returns:
        ModelMetadata: The DeviceModel metadata.
    """
    return {
        "input_tree_spec": tree_spec_input,
        "input_shapes": get_input_shapes(onnx_graph),
        "input_onnx_names": get_input_names(onnx_graph),
        "input_none_names": input_none_names,
        "output_tree_spec": tree_spec_output,
        "output_shapes": get_output_shapes(onnx_graph),
        "output_onnx_names": get_output_names(onnx_graph),
        "signature": inspect.signature(model.forward),
        "onnx_node_names": get_node_names(onnx_graph),
        "is_bytes_pickled": onnx_graph.ByteSize() > TWO_GB,
        "config": model.config if hasattr(model, "config") else None,
    }


def get_onnx_bytes(*args, **kwargs) -> bytes:
    """Return onnx bytes only.

    See ``get_onnx_bytes_and_metadata()`` for more info.
    """
    onnx_bytes = get_onnx_bytes_and_metadata(*args, **kwargs)[0]
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    return onnx_bytes_obj.get_onnx_model_file_bytes()


def remove_node_training_mode(onnx_model: ModelProto, node_op_type: str) -> ModelProto:
    """Remove training_mode attribute from selected node type.

    Args:
        onnx_model: The onnx model.
        node_op_type: The node type to remove training_mode attribute from.

    Returns:
        The onnx model with the training_mode attribute removed.
    """
    for node in onnx_model.graph.node:
        if node.op_type == node_op_type:
            for attribute in node.attribute:
                if attribute.name == "training_mode":
                    if attribute.i == 1:
                        node.output.remove(node.output[1])
                        node.output.remove(node.output[1])
                    attribute.i = 0

    return onnx_model
