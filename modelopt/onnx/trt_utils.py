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

"""This module contains TensorRT utils."""

import ctypes
import platform

import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.utils import get_dynamic_graph_inputs, parse_shapes_spec, save_onnx

try:
    import tensorrt as trt

    TRT_PYTHON_AVAILABLE = True
except ImportError:
    TRT_PYTHON_AVAILABLE = False


def get_custom_layers(
    onnx_path: str | onnx.ModelProto,
    trt_plugins: list[str] | None,
    strongly_typed: bool = False,
) -> tuple[list[str], dict]:
    """Gets custom layers in ONNX model.

    Args:
        onnx_path: Path or ModelProto of the input ONNX model.
        trt_plugins: list with paths to custom TensorRT plugins.
        strongly_typed: Boolean indicating whether to run TensorRT inference in stronglyTyped mode or not.

    Returns:
        List of custom layers.
        Dictionary containing tensors information: {'tensor_name': {'shape': tensor.shape, 'dtype': tensor.dtype}}
    """
    logger.debug("Checking for custom TensorRT ops")

    # Initialize TensorRT plugins
    if trt_plugins:
        logger.debug(f"Loading TensorRT plugins: {trt_plugins}")
        for plugin in trt_plugins:
            ctypes.CDLL(plugin)

    # Create builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)
    network = (
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        if strongly_typed
        else builder.create_network()
    )
    logger.debug("Created TensorRT builder and network")

    # Parse ONNX file
    parser = trt.OnnxParser(network, trt_logger)
    parser_func = parser.parse_from_file if isinstance(onnx_path, str) else parser.parse
    onnx_path = onnx_path if isinstance(onnx_path, str) else onnx_path.SerializeToString()
    if not parser_func(onnx_path):
        error_str = [str(parser.get_error(error)) for error in range(parser.num_errors)]
        raise Exception(f"Failed to parse ONNX file: {''.join(error_str)}")

    # Obtain layer info
    custom_layers = []
    all_tensor_info = {}
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)

        # Obtain plugin layer names
        if "PLUGIN" in str(layer.type):
            custom_layers.append(layer.name)
            logger.debug(f"Found custom layer: {layer.name}")

        # Collect all tensors' type and shape.
        # Replace dynamic axis representation from -1 to 'unk' for 'onnxsim' support if enabled.
        for i in range(layer.num_inputs):
            input_tensor = layer.get_input(i)
            if input_tensor:
                all_tensor_info[input_tensor.name] = {
                    "shape": ["unk" if (s == -1) else s for s in input_tensor.shape],
                    "dtype": input_tensor.dtype,
                }

        for i in range(layer.num_outputs):
            output_tensor = layer.get_output(i)
            if output_tensor and not all_tensor_info.get(output_tensor.name, None):
                all_tensor_info[output_tensor.name] = {
                    "shape": ["unk" if (s == -1) else s for s in output_tensor.shape],
                    "dtype": output_tensor.dtype,
                }

    logger.info(f"Found {len(custom_layers)} custom layers and {len(all_tensor_info)} tensors")
    return custom_layers, all_tensor_info


def infer_types_shapes(graph: gs.Graph, all_tensor_info: dict) -> None:
    """Updates tensor shapes in ORT graph.

    Args:
        graph: ONNX model's GS graph.
        all_tensor_info: Dictionary containing tensors information.

    Returns:
        None. In-memory modification of graph.
    """
    logger.debug("Inferring types and shapes for graph tensors")

    def _map_trt_to_python_type(trt_type: trt.DataType):
        try:
            return trt.nptype(trt_type)
        except TypeError as e:
            logger.warning(f"{e}. TRT datatype: {trt_type}. Setting to None")
            return None

    updated_tensors = 0
    for node in graph.nodes:
        for out in node.outputs:
            if out.name in all_tensor_info:
                out.shape = all_tensor_info[out.name]["shape"]
                out.dtype = out.dtype or _map_trt_to_python_type(all_tensor_info[out.name]["dtype"])
                updated_tensors += 1

    logger.info(f"Updated {updated_tensors} tensors with type and shape information")
    graph.cleanup().toposort()


def set_trt_plugin_domain(model: onnx.ModelProto, custom_ops: list[str]) -> onnx.ModelProto:
    """Set TensorRT plugin domain info in the graph.

    Args:
        model: ONNX model to set custom op domain.
        custom_ops: list of custom ops.

    Returns:
        onnx.ModelProto: ONNX model with domain set in custom ops.
    """
    logger.info(f"Found custom operators: {custom_ops}")
    trt_plugin_domain = "trt.plugins"
    trt_plugin_version = 1

    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op in custom_ops:
            # Add TRT domain to each custom node
            node.domain = trt_plugin_domain

    # Add TRT domain and version to the graph
    model = gs.export_onnx(graph)
    model.opset_import.append(onnx.helper.make_opsetid(trt_plugin_domain, trt_plugin_version))
    logger.info(f"Added TRT plugin domain {trt_plugin_domain} version {trt_plugin_version}")
    return model


def infer_types_shapes_tensorrt(
    model: onnx.ModelProto,
    trt_plugins: list[str] = [],
    all_tensor_info: dict = {},
    strongly_typed: bool = False,
) -> onnx.ModelProto:
    """Update tensor types and shapes from TensorRT inference data.

    Args:
        model: ONNX model to infer types and shapes.
        trt_plugins: list of TensorRT plugin library paths in .so format (compiled shared library).
        all_tensor_info: dictionary with tensor data from TensorRT run.
        strongly_typed: boolean indicating if the TensorRT run should be stronglyTyped or not.

    Returns:
        onnx.ModelProto: ONNX model with inferred types and shapes.
    """
    # Obtain tensor data if not given
    if not all_tensor_info:
        _, all_tensor_info = get_custom_layers(model, trt_plugins, strongly_typed)

    # Ensure that all tensors in the graph have type and shape info
    graph = gs.import_onnx(model)
    infer_types_shapes(graph, all_tensor_info)
    model = gs.export_onnx(graph)
    return model


def load_onnx_model(
    onnx_path: str,
    trt_plugins: list[str] | None = None,
    override_shapes: str | None = None,
    use_external_data_format: bool = False,
    intermediate_generated_files: list[str] | None = None,
) -> tuple[onnx.ModelProto, bool, list[str], str, bool]:
    """Load ONNX model. If 'tensorrt' is installed, check if the model has custom ops and ensure it's supported by ORT.

    Args:
        onnx_path: Path to the input ONNX model.
        trt_plugins: List with paths to custom TensorRT plugins.
        override_shapes: Override model input shapes with static shapes.
        use_external_data_format: If True, separate data path will be used to store the weights of the quantized model.
        intermediate_generated_files: List of paths of intermediate ONNX files, generated during quantization.

    Returns:
        Loaded ONNX model supported by ORT.
        Boolean indicating whether the model has custom ops or not.
        List of custom ops in the ONNX model.
        Path to new intermediary ONNX model.
        Boolean indicating whether we should use external data format for the intermediate and quantized models.
    """
    custom_ops = []
    has_custom_op = False

    # Load the model and weights
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    size_threshold = 2 * (1024**3)  # 2GB
    use_external_data_format = onnx_model.ByteSize() > size_threshold or use_external_data_format

    # If inputs are dynamic and override shapes are given, set them as static
    dynamic_inputs = get_dynamic_graph_inputs(onnx_model)
    static_shaped_onnx_path = None
    if len(dynamic_inputs) > 0:
        input_names = [inp.name for inp in dynamic_inputs]
        logger.info(f"Model has dynamic inputs: {input_names}")

        if override_shapes:
            override_shapes_arr = parse_shapes_spec(override_shapes)
            for graph_input in onnx_model.graph.input:
                if graph_input.name not in input_names:
                    continue
                inp_shapes = override_shapes_arr[graph_input.name]
                logger.info(f"Setting '{graph_input.name}' shape to {inp_shapes}")
                for idx, (d, s) in enumerate(
                    zip(graph_input.type.tensor_type.shape.dim, inp_shapes)
                ):
                    if not d.dim_value:
                        graph_input.type.tensor_type.shape.dim[idx].dim_value = s

            static_shaped_onnx_path = onnx_path.replace(".onnx", "_static.onnx")
            save_onnx(onnx_model, static_shaped_onnx_path, use_external_data_format)
            intermediate_generated_files.append(static_shaped_onnx_path)  # type: ignore[union-attr]

    if TRT_PYTHON_AVAILABLE and platform.system() != "Windows":
        # Check if there's a custom TensorRT op in the ONNX model. If so, make it ORT compatible by adding
        # `trt.plugins to the ONNX graph.
        custom_layers, all_tensor_info = get_custom_layers(
            static_shaped_onnx_path or onnx_path, trt_plugins
        )
        has_custom_op = bool(custom_layers)

        if has_custom_op:
            logger.debug(f"Found custom layers: {custom_layers}")
            custom_ops = {
                node.op_type for node in onnx_model.graph.node if node.name in custom_layers
            }

            # Set TensorRT plugin domain info in the graph for ORT compatibility
            onnx_model = set_trt_plugin_domain(onnx_model, custom_ops)

            # Infer types and shapes in the graph for ORT compatibility
            onnx_model = infer_types_shapes_tensorrt(onnx_model, trt_plugins or [], all_tensor_info)

    return (
        onnx_model,
        has_custom_op,
        custom_ops,
        static_shaped_onnx_path or onnx_path,
        use_external_data_format,
    )
