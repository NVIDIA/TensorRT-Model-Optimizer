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

"""Performs INT8 quantization of an ONNX model, and returns the ONNX ModelProto."""

import logging
import os
import tempfile

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnxconverter_common import convert_float_to_float16
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization.calib_utils import import_scales_from_calib_cache
from modelopt.onnx.quantization.graph_utils import (
    build_non_residual_input_map,
    classify_partition_nodes,
    expand_node_names_from_patterns,
    filter_quantizable_kgen_heads,
    find_nodes_from_matmul_to_exclude,
    find_nodes_to_exclude,
    get_concat_eliminated_tensors,
    remove_partial_input_qdq,
)
from modelopt.onnx.quantization.ort_patching import _quantize_static as quantize_static
from modelopt.onnx.quantization.ort_utils import configure_ort
from modelopt.onnx.quantization.partitioning import (
    find_fusible_partitions,
    find_non_quantizable_partitions_from_patterns,
    find_quantizable_nodes,
    get_skiped_output_layers,
)
from modelopt.onnx.quantization.qdq_utils import replace_scale_values

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)


def _find_nodes_to_quantize(
    graph: Graph,
    quantizable_op_types: list[str],
    verbose: bool,
) -> tuple[list[Node], list[tuple[Node, Node, str]]]:
    # Build a map of add nodes to their non-residual inputs, i.e. fusible with Conv group
    logging.info("Building non-residual Add input map ...")
    non_residual_inputs, _ = build_non_residual_input_map(graph)

    logging.info(
        "Searching for hard-coded patterns like MHA, LayerNorm, etc. to avoid quantization."
    )
    non_quantizable_hard_coded_partitions = find_non_quantizable_partitions_from_patterns(graph)

    logging.info("Building KGEN/CASK targeted partitions ...")
    # partitioned_nodes keeps track of nodes that are already part of some partition.
    # Certain nodes of those partitions are quantizable. For example, heads.
    partitioned_nodes = set(sum(non_quantizable_hard_coded_partitions, []))
    cask_fusible_partitions, kgen_partitions = find_fusible_partitions(
        graph,
        partitioned_nodes,
        non_residual_inputs,
    )
    if verbose:
        logging.info(
            "CASK fusible partitions:"
            f" {[[node.name for node in partition] for partition in cask_fusible_partitions]}"
        )
        logging.info(
            "KGEN partitions:"
            f" {[[node.name for node in partition] for partition in kgen_partitions]}"
        )

    logging.info("Classifying the partition nodes ...")
    _, quantizable_partition_nodes, no_quantize_inputs = classify_partition_nodes(
        cask_fusible_partitions,
    )
    quantizable_kgen_heads, no_quantize_kgen_inputs = filter_quantizable_kgen_heads(
        cask_fusible_partitions,
        kgen_partitions,
        quantizable_op_types,
    )

    quantizable_nodes = quantizable_kgen_heads + quantizable_partition_nodes
    paritially_quantizable_nodes = [dst for _, dst, _ in no_quantize_inputs]

    # Quantize all inputs of partially quantizable nodes by ORT
    # but remove QDQ from non-quantizable inputs in the post-processing step
    quantizable_nodes.extend(paritially_quantizable_nodes)

    quantizable_nodes.extend(
        find_quantizable_nodes(graph, quantizable_nodes, partitioned_nodes, quantizable_op_types)
    )

    skip_list = get_skiped_output_layers(graph, paritially_quantizable_nodes)
    quantizable_nodes = [node for node in quantizable_nodes if node.name not in skip_list]

    return quantizable_nodes, no_quantize_inputs + no_quantize_kgen_inputs


def quantize(
    onnx_path: str,
    calibration_method: str = "entropy",
    calibration_data_reader: CalibrationDataReader = None,
    calibration_cache_path: str = None,
    calibration_shapes: str = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    op_types_to_quantize: list[str] = None,
    op_types_to_exclude: list[str] = None,
    nodes_to_quantize: list[str] = None,
    nodes_to_exclude: list[str] = None,
    use_external_data_format: bool = True,
    intermediate_generated_files: list[str] = [],
    verbose: bool = False,
    trt_extra_plugin_lib_paths: str = None,
    high_precision_dtype: str = "fp32",
    passes: list[str] = None,
    **kwargs,
) -> onnx.onnx_pb.ModelProto:
    """Applies INT8 quantization to an ONNX file using the compiler friendly heuristics.

    Quantization of ['Add', 'AveragePool', 'BatchNormalization', 'Clip', 'Conv', 'ConvTranspose',
    'Gemm', 'GlobalAveragePool', 'MatMul', 'MaxPool', 'Mul'] op types are supported.
    """
    logging.info("Quantization Mode: int8")

    # Take the onnx graph
    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
    graph = gs.import_onnx(onnx_model)
    graph.toposort()

    # Change the default configuration of ORT quantization
    op_types_to_quantize = op_types_to_quantize or []
    op_types = set([node.op for node in graph.nodes])
    trt_guided_options, quantizable_op_types = configure_ort(
        list(op_types), op_types_to_quantize, trt_extra_plugin_lib_paths, calibration_eps
    )
    logging.info(
        f"Quantizable op types in the model: {[t for t in quantizable_op_types if t in op_types]}"
    )

    # Collect node names to include in quantization
    no_quantize_inputs = []
    nodes_to_quantize = expand_node_names_from_patterns(graph, nodes_to_quantize)
    if not nodes_to_quantize:
        # If nodes_to_quantize is not passed, use user supplied op_types_to_quantize list
        nodes_to_quantize = [node.name for node in graph.nodes if node.op in op_types_to_quantize]

        # If op_types_to_quantize is not provided, use default QDQ placement algorithm
        if not nodes_to_quantize:
            quantizable_nodes, no_quantize_inputs = _find_nodes_to_quantize(
                graph,
                quantizable_op_types,
                verbose,
            )
            nodes_to_quantize = [node.name for node in quantizable_nodes]

    # Read the calibration cache and quantize nodes for which activation scale values are cached
    if calibration_cache_path:
        act_scales_dict = import_scales_from_calib_cache(calibration_cache_path)
        logging.info(f"Using calibration cache from {calibration_cache_path}")
        iq_quantized_nodes = []
        quantized_tensors = [
            tensor_name.replace("_scale", "") for tensor_name in act_scales_dict.keys()
        ]
        for node in graph.nodes:
            for node_input in node.inputs:
                if node_input.name in quantized_tensors:
                    iq_quantized_nodes.append(node.name)

        logging.info(
            f"Skipping quantization of nodes: {set(nodes_to_quantize) - set(iq_quantized_nodes)}"
        )
        nodes_to_quantize = list(set(nodes_to_quantize).intersection(iq_quantized_nodes))

    enable_gemv_detection_for_trt = kwargs.get("enable_gemv_detection_for_trt", True)
    if enable_gemv_detection_for_trt:
        # Either of m or n in matmul is 1, this matmul cannot utilize TensorCores.
        # The perf of adding Q/DQ layers is not good in TRT. Thus, in this case,
        # do not add Q/DQ layers to this matmul.
        matmul_nodes_to_exclude = find_nodes_from_matmul_to_exclude(
            onnx_path,
            use_external_data_format,
            intermediate_generated_files,
            calibration_data_reader,
            calibration_eps,
            verbose,
        )
        nodes_to_exclude.extend(matmul_nodes_to_exclude)

    # Collect node names to exclude from quantization
    nodes_to_exclude = find_nodes_to_exclude(graph, nodes_to_exclude, op_types_to_exclude)

    # Update the list of nodes to quantize
    nodes_to_quantize = [
        node_name for node_name in nodes_to_quantize if node_name not in nodes_to_exclude
    ]

    logging.info(f"Total number of nodes: {len(graph.nodes)}")
    if not nodes_to_quantize:
        logging.info(
            "No node or node type is selected for quantization or model does not have them!"
        )
        return
    elif verbose:
        logging.info(f"Selected {len(nodes_to_quantize)} nodes to quantize: {nodes_to_quantize}")

    if passes and "concat_elimination" in passes:
        group_qdq_tensors = get_concat_eliminated_tensors(onnx_model, nodes_to_quantize)
        if group_qdq_tensors:
            trt_guided_options["group_qdq_tensors"] = group_qdq_tensors
            if verbose:
                logging.info("concat_elimination enable")

    # Create a temp file for intermediate model
    tmp_onnx_file, tmp_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(tmp_onnx_file)

    # Use ORT api to quantize the onnx model
    quantize_static(
        onnx_path,
        tmp_onnx_path,
        calibration_data_reader,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        per_channel=True,
        extra_options=trt_guided_options,
        use_external_data_format=use_external_data_format,
        calibrate_method=(
            CalibrationMethod.Entropy
            if calibration_method == "entropy"
            # With ActivationSymmetric as True, MinMax calibration is equivalent to max calibration
            else CalibrationMethod.MinMax
        ),
    )

    intermediate_generated_files.append(tmp_onnx_path)
    if use_external_data_format:
        intermediate_generated_files.append(tmp_onnx_path + ".data")

    # Post-processing of the onnx model after ORT quantization
    onnx_model = onnx.load(tmp_onnx_path)
    graph = gs.import_onnx(onnx_model)
    remove_partial_input_qdq(graph, no_quantize_inputs)
    onnx_model = gs.export_onnx(graph)
    if calibration_cache_path:
        replace_scale_values(onnx_model.graph, act_scales_dict)

    if high_precision_dtype == "fp16":
        # We need to convert float to float16 so as to speed up layers like LayerNorm or GroupNorm.
        logging.info("Converting float tensors to float16")
        onnx_model = convert_float_to_float16(
            onnx_model, keep_io_types=True, disable_shape_infer=True
        )

    return onnx_model
