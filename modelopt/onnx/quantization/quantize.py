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

"""This module is used to convert an ONNX model (w/o QDQ nodes) and calibration data into an ONNX model with QDQ nodes.

Quantization typically targets linear operations like Convolution (Conv), Matrix Multiplication (MatMul), etc., as these
transformations often yield significant performance improvements. However, many other operations can also be quantized
(i.e., they have low-precision kernels available), which can provide optimal performance with a minimal drop in
accuracy.

The default operation types that this ONNX post-training quantization (PTQ) tool quantizes in different quantization
modes are as follows:
- INT8 mode: Add, AveragePool, BatchNormalization, Clip, Conv, ConvTranspose, Gemm, GlobalAveragePool, MatMul, MaxPool,
Mul
- INT4 mode: Gemm, MatMul
- FP8 mode: Conv, Gemm, MatMul

This tool inserts Quantize-Dequantize (QDQ) nodes following compiler-friendly patterns and generates an explicit ONNX
model.
"""

import logging
import os
import platform
import shutil
import tempfile
from typing import Any, Optional

import numpy as np
import onnx
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.calib_utils import (
    CalibrationDataProvider,
    CalibrationDataType,
    RandomDataProvider,
)
from modelopt.onnx.quantization.fp8 import quantize as quantize_fp8
from modelopt.onnx.quantization.graph_utils import (
    add_fp16_fp32_cast,
    find_nodes_from_mha_to_exclude,
    print_stat,
)
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.quantization.int8 import quantize as quantize_int8
from modelopt.onnx.quantization.qdq_utils import qdq_to_dq
from modelopt.onnx.quantization.trt_utils import load_onnx_model
from modelopt.onnx.utils import duplicate_shared_constants, name_onnx_nodes, save_onnx

__all__ = ["quantize"]


# Set logging level to info
logging.getLogger().setLevel(logging.INFO)


def _preprocess_onnx(
    onnx_path: str,
    use_external_data_format: bool,
    output_path: str,
    trt_plugins: Optional[str],
    trt_plugins_precision: Optional[list[str]],
) -> tuple[str, list[str], bool]:
    intermediate_generated_files = []
    output_dir = os.path.dirname(output_path)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    # Load the model and weights
    onnx_model, has_custom_op, custom_ops = load_onnx_model(
        onnx_path, trt_plugins, use_external_data_format
    )
    if has_custom_op:
        onnx_path = os.path.join(output_dir, f"{model_name}_ort_support.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logging.info(
            f"Model with ORT support is saved to {onnx_path}. Model contains custom ops: {custom_ops}."
        )
        intermediate_generated_files.append(onnx_path)
    elif platform.system() != "Windows":
        logging.warning(
            "No custom ops found. If that's not correct, please make sure that the 'tensorrt' python "
            "is correctly installed and that the path to 'libcudnn*.so' is in PATH or LD_LIBRARY_PATH. If the "
            "custom op is not directly available as a plugin in TensorRT, please also make sure that "
            "the path to the compiled '.so' TensorRT plugin is also being given via the "
            "'--trt_plugins' flag (requires TRT 10+)."
        )

    # Check if there's a custom TensorRT op in the ONNX model
    has_custom_op = np.any([node.domain == "trt.plugins" for node in onnx_model.graph.node])

    # Per-Channel support with QDQ format requires onnx opset version 13 or above
    ai_onnx_domain = [
        opset
        for opset in onnx_model.opset_import
        if not opset.domain or opset.domain in ["ai.onnx", "ai.onnx.contrib"]
    ]
    opset_version = ai_onnx_domain[0].version
    logging.info(f"Model {onnx_path} with opset_version {opset_version} is loaded.")

    required_opset_version = 13
    if opset_version < required_opset_version and opset_version != 1:
        opset_version = required_opset_version
        onnx_model = onnx.version_converter.convert_version(onnx_model, opset_version)
        onnx_path = os.path.join(output_dir, f"{model_name}_opset{opset_version}.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} with opset_version {opset_version}.")
        intermediate_generated_files.append(onnx_path)

    # Sometimes input onnx model does not contain the node names
    # This tool depends on those names, so we assign names if needed
    graph = onnx_model.graph
    is_named = name_onnx_nodes(graph)
    onnx_model, is_duplicated_constant = duplicate_shared_constants(onnx_model)  # FasterViT-0, eef

    if is_named or is_duplicated_constant:
        onnx_path = os.path.join(output_dir, f"{model_name}_named.onnx")
        save_onnx(onnx_model, onnx_path, use_external_data_format)
        logging.info(f"Model is cloned to {onnx_path} after naming the nodes.")
        intermediate_generated_files.append(onnx_path)

    # If custom op precisions are given, check if they're fp16. If so, add cast_to_fp16 before all inputs and
    # cast_to_fp32 after all outputs.
    if trt_plugins_precision:
        custom_ops_to_cast = []
        for trt_plugin_precision in trt_plugins_precision:
            assert ":" in trt_plugin_precision, (
                "Plugin precision is incorrectly formatted."
                " Please check that it's in the format <op_type>:<precision>."
            )
            op_type, precision = trt_plugin_precision.split(":")
            if precision == "fp16":
                custom_ops_to_cast.append(op_type)
        if custom_ops_to_cast:
            onnx_path = add_fp16_fp32_cast(onnx_path, custom_ops_to_cast)
            logging.info("Adding cast nodes related to custom ops to match requested precisions.")
            intermediate_generated_files.append(onnx_path)
    return onnx_path, intermediate_generated_files, has_custom_op


def quantize(
    onnx_path: str,
    quantize_mode: str = "int8",
    calibration_data: CalibrationDataType = None,
    calibration_method: str = None,
    calibration_cache_path: str = None,
    calibration_shapes: str = None,
    calibration_eps: list[str] = ["cuda:0", "cpu", "trt"],
    op_types_to_quantize: list[str] = None,
    op_types_to_exclude: list[str] = None,
    nodes_to_quantize: list[str] = None,
    nodes_to_exclude: list[str] = None,
    use_external_data_format: bool = False,
    keep_intermediate_files: bool = False,
    output_path: str = None,
    verbose: bool = False,
    trt_plugins: str = None,
    trt_plugins_precision: list[str] = None,
    high_precision_dtype: str = None,
    mha_accumulation_dtype: str = "fp32",
    disable_mha_qdq: bool = False,
    dq_only: bool = True,
    block_size: Optional[int] = None,
    use_zero_point: bool = False,
    **kwargs: Any,
) -> None:
    """Quantizes the provided ONNX model.

    Args:
        onnx_path:
            Path to the input ONNX model.
        quantize_mode:
            Quantization mode. One of 'int8' (default), 'int4' and 'fp8'.
        calibration_data:
            Calibration data, either a numpy array or list/dict of numpy arrays.
        calibration_method:
            Calibration method choices. Options are int8: 'entropy' (default) and 'max',
            fp8: 'max' (default) and int4: 'awq_clip' (default), 'awq_lite', 'awq_full' and 'rtn_dq'.
        calibration_cache_path:
            Path to pre-calculated activation tensor ranges, also known as calibration cache.
        calibration_eps:
            Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'dml:x', 'cpu', 'trt'], where 'x' is the device id.
        op_types_to_quantize:
            List of op types to quantize. If None (default), all supported operators are quantized.
            This flag does not support regular expression.
        op_types_to_exclude:
            List of op types to exclude from quantization. This flag does not support regular expression.
        nodes_to_quantize:
            List of node names to quantize. If None (default), all supported nodes are quantized.
            This flag supports regular expression.
        nodes_to_exclude:
            List of node names to exclude from quantization. This flag supports regular expression.
        use_external_data_format:
            If True, separate data path will be used to store the weights of the quantized model.
        keep_intermediate_files:
            If True, keep all intermediate files generated during the ONNX model's conversion/calibration.
        output_path:
            Output filename to save the quantized ONNX model.
            If None, save in the same directory as the original ONNX model with .quant suffix.
        verbose:
            If True, print details of node partition, selection etc. throughout the quantization process.
        trt_plugins:
            Specifies custom TensorRT plugin library paths in .so format (compiled shared library).
            For multiple paths, separate them with a semicolon, i.e.: "lib_1.so;lib_2.so".
            If this is not None or the model has custom ops, TensorrtExecutionProvider becomes the first choice as
            calibration execution provider, meaning that the TensorRT is a requirement.
        trt_plugins_precision:
            A space-separated list indicating the precision for each custom op.
            Each item should have the format <op_type>:<precision>, where precision can be fp32 (default) or fp16.
            For example: op_type_1:fp16 op_type_2:fp32.
        high_precision_dtype:
            High precision data type, one of ['fp32', 'fp16']. If high_precision_dtype == 'fp16', model's weight and
            activation will be converted to fp16.
        mha_accumulation_dtype:
            MHA accumulation dtype. One of ['fp32', 'fp16']. 'fp32' by default.
            If quantize_mode == 'fp8' and high_precision_dtype == 'fp32', Cast nodes will be added to
            MHA's bmm1 and bmm2's input and output tensors.
        disable_mha_qdq:
            Don't add Q/DQ layers to MatMuls in MHA pattern.
        dq_only:
            If True (default), only add DQ nodes to the model. If False, add Q/DQ nodes to the model.
        block_size:
            Block size parameter for int4 quantization.
        use_zero_point:
            Use zero-point based quantization, if True.
        kwargs:
            Additional keyword arguments for int4 quantization, including:
            - awqlite_alpha_step (float): Alpha step for lite, range [0, 1].
            - awqclip_alpha_step (float): Min alpha step for clip, range [awqclip_alpha_step, 1].
            - awqclip_alpha_min (float): Alpha step to find best alpha for clip.
            - awqclip_bsz_col (int): Batch size for processing the column dimension in clip.

    Returns:
        None, writes the quantized onnx model in the supplied output_path
        or writes to the same directory with filename like "<model_name>.quant.onnx".
    """
    # quantize_static creates a shape-inferred copy at the input model's directory
    # Needs to check if we have write permission to this directory
    assert onnx_path.endswith(".onnx") or onnx_path.endswith(".pb")
    if not os.access(os.path.dirname(os.path.abspath(onnx_path)), os.W_OK):
        old_dir = os.path.dirname(os.path.abspath(onnx_path))
        tmp_dir = tempfile.mkdtemp()
        logging.info(f"Directory {old_dir} is not writable, store intermediate files in {tmp_dir}")
        onnx_path = os.path.join(tmp_dir, os.path.basename(onnx_path))

        # We assume that the model directory contains only model related weights and protobuf file
        # Anything extra in the model directory will be copied unnecessarily
        for file in os.listdir(old_dir):
            old_file_path = os.path.join(old_dir, file)
            new_file_path = os.path.join(tmp_dir, file)
            if os.path.isfile(old_file_path):
                shutil.copy(old_file_path, new_file_path)

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    if not output_path:
        output_dir = os.path.dirname(onnx_path)
        output_path = os.path.join(output_dir, f"{model_name}.quant.onnx")
        logging.info(f"No output path specified, save quantized model to {output_path}")

    # We need to preprocess the model with naming, weight duplication etc.
    onnx_path, intermediate_generated_files, has_custom_op = _preprocess_onnx(
        onnx_path, use_external_data_format, output_path, trt_plugins, trt_plugins_precision
    )
    # If the model has a custom op and no plugin path was given, assume that this custom op is being implemented
    # by a TRT native plugin. In order to enable the TRT EP, 'trt_extra_plugin_lib_paths' needs to be != None.
    if has_custom_op and not trt_plugins:
        trt_plugins = ""

    # Use random scales if calibration data is not supplied
    if calibration_data is None:
        calibration_data_reader = RandomDataProvider(onnx_path, calibration_shapes)
    else:
        calibration_data_reader = CalibrationDataProvider(
            onnx_path, calibration_data, calibration_shapes
        )

    nodes_to_quantize = nodes_to_quantize or []
    nodes_to_exclude = nodes_to_exclude or []

    # (1) If disable_mha_qdq is set, don't add Q/DQ layers to MatMuls in MHA pattern.
    # (2) else when quantize_mode == "int8", if seq_len > 512, don't add Q/DQ layers to
    # MatMuls in MHA pattern.
    # (2) else when quantize_mode == "fp8", if head_size > 256 or head_size % 16 != 0
    # or mha doesn't meet fp8 fMHA v2 pattern, don't add Q/DQ layers to MatMuls in MHA pattern.
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
        onnx_path,
        use_external_data_format,
        nodes_to_exclude,
        disable_mha_qdq,
        quantize_mode,
        intermediate_generated_files,
        calibration_data_reader,
        calibration_eps,
        verbose,
    )

    if quantize_mode in ["fp8", "int8"]:
        quantize_func = quantize_int8 if quantize_mode == "int8" else quantize_fp8
        default_calibration_method = "entropy" if quantize_mode == "int8" else "max"
        onnx_model = quantize_func(
            onnx_path=onnx_path,
            calibration_method=calibration_method or default_calibration_method,
            calibration_data_reader=calibration_data_reader,
            calibration_cache_path=calibration_cache_path,
            calibration_shapes=calibration_shapes,
            calibration_eps=calibration_eps,
            op_types_to_quantize=op_types_to_quantize,
            op_types_to_exclude=op_types_to_exclude,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            use_external_data_format=use_external_data_format,
            intermediate_generated_files=intermediate_generated_files,
            verbose=verbose,
            trt_extra_plugin_lib_paths=trt_plugins,
            high_precision_dtype=high_precision_dtype,
            mha_accumulation_dtype=mha_accumulation_dtype,
        )
    elif "int4" in quantize_mode:
        onnx_model = quantize_int4(
            onnx_path=onnx_path,
            calibration_method=calibration_method or "awq_clip",
            calibration_data_reader=calibration_data_reader,
            calibration_eps=calibration_eps,
            use_external_data_format=use_external_data_format,
            block_size=block_size,
            nodes_to_exclude=nodes_to_exclude,
            use_zero_point=use_zero_point,
            **kwargs,
        )
    else:
        raise RuntimeError(f"Invalid quantization mode choice: {quantize_mode}")

    if onnx_model:
        # Fuse Q nodes for INT8/FP8 mode
        if quantize_mode in ["int8", "fp8"] and dq_only:
            onnx_model = qdq_to_dq(onnx_model, verbose=verbose)

        # Collect and print stats of the quantized model
        print_stat(gs.import_onnx(onnx_model), verbose)

        # Save the quantized model to the output path
        save_onnx(onnx_model, output_path, use_external_data_format)
        logging.info(f"Quantized onnx model is saved as {output_path}")

    # Check if intermediate files should be deleted
    if not keep_intermediate_files:
        for file in intermediate_generated_files:
            if os.path.exists(file):
                os.remove(file)

    # Check if the quantized model is valid
    try:
        onnx.checker.check_model(output_path)
    except C.ValidationError as e:
        logging.warning("ONNX model checker failed, check your deployment status.")
        logging.warning(e)
