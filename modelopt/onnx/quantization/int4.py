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

"""Performs INT4 WoQ on an ONNX model, and returns the ONNX ModelProto."""

import copy
import gc
import math
import os
import tempfile
import time
from typing import Any, cast

import numpy
import onnx
import onnx.numpy_helper as numpy_helper
import onnx_graphsurgeon as gs
from onnxruntime.quantization.calibrate import CalibrationDataReader
from tqdm import tqdm

import modelopt.onnx.quantization.qdq_utils as qdq
from modelopt.onnx.logging_config import configure_logging, logger
from modelopt.onnx.op_types import is_fusible_scaling_op
from modelopt.onnx.quantization.calib_utils import RandomDataProvider
from modelopt.onnx.quantization.graph_utils import (
    expand_node_names_from_patterns,
    get_tensor_consumer_nodes,
    get_tensor_producer_nodes,
)
from modelopt.onnx.quantization.gs_patching import patch_gs_modules
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.utils import save_onnx

__all__ = ["quantize"]

has_cupy = False
cupy_warning_msg = ""
try:
    import cupy as np

    if not np.cuda.is_available():
        raise ImportError("Unable to use cupy as CUDA is not available.")

    has_cupy = True
except Exception as e:
    # ImportError if cupy is not installed
    # cupy_backends.cuda.api.runtime.CUDARuntimeError if CUDA is not available
    import numpy as np

    cupy_warning_msg = f"Using slower INT4 ONNX quantization using numpy: {e}"


NUM_BITS = 4
INT4_SCALE = 7.0
INT4_MIN = -(2 ** (NUM_BITS - 1))  # -8
INT4_MAX = 2 ** (NUM_BITS - 1) - 1  # 7
UINT4_MIN = 0
UINT4_MAX = 15

# following min-value for clip is taken from AutoAWQ where zero-point based quantization is
# supported and working
CLIP_MIN = 1e-5


def _next_block_size_multiple(x: float, block_size: int) -> float:
    return math.ceil(x / block_size) * block_size


def _pad(w: np.ndarray, block_size: int) -> np.ndarray:
    """Pads `w` to next largest multiple of block_size, on axis 0."""
    if w.shape[0] % block_size == 0:
        return w

    pad_width = _next_block_size_multiple(w.shape[0], block_size) - w.shape[0]
    pads = [(0, 0) for _ in range(len(w.shape))]
    pads[0] = (0, pad_width)
    return np.pad(w, pads, mode="constant", constant_values=0)


def _depad(w: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """Depad axis 0 to original shape."""
    if w.shape == orig_shape:
        return w
    return w[0 : orig_shape[0], ...]


def find_scales(w: np.ndarray, block_size: int, alpha: float = 1.0, use_zero_point: bool = False):
    """Find scale factors for `w` via `s = max(w.block(block_size)) / 7`."""
    w = _pad(w, block_size)
    w = w.T
    s_last_dim = w.shape[-1] // block_size
    s_shape = list(w.shape)
    s_shape[-1] = s_last_dim
    if not use_zero_point:
        w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)
        s = (w_amax * alpha) / INT4_SCALE
        s = s.reshape(s_shape).T
        z = None
    else:
        max_val = w.reshape(-1, block_size).max(axis=-1)
        min_val = w.reshape(-1, block_size).min(axis=-1)
        max_int = UINT4_MAX
        min_int = UINT4_MIN
        s = (max_val - min_val).clip(min=CLIP_MIN) / max_int
        # z = -np.round(temp).clip(min=min_int, max=max_int)    # gives 0 - need to check
        temp = min_val / s
        temp = np.round(temp)
        temp = -temp
        temp = temp.clip(min=min_int, max=max_int)
        z = temp
        assert s.shape == z.shape, "s and z shape mismatch"
        s = s.reshape(s_shape).T
        z = z.reshape(s_shape).T
    return s, z


def rtn(w: np.ndarray, s: np.ndarray, block_size: int, zp: np.ndarray = None) -> np.ndarray:
    """Quantizes `w` with scale factors `s` via Round-to-Nearest.

    Ties are broken by rounding to the nearest even number.
    """
    w_padded = _pad(w, block_size)
    num_blocks = w_padded.shape[0] // s.shape[0]
    if zp is None:
        w_padded = (
            np.rint(w_padded / s.repeat(num_blocks, axis=0))
            .clip(INT4_MIN, INT4_MAX)
            .astype(np.int8)
        )
    else:
        w_padded = (
            (np.rint(w_padded / s.repeat(num_blocks, axis=0)) + zp.repeat(num_blocks, axis=0))
            .clip(UINT4_MIN, UINT4_MAX)
            .astype(np.int8)
        )
    return _depad(w_padded, w.shape)


def dq_tensor(w: np.ndarray, s: np.ndarray, block_size: int, zp: np.ndarray = None) -> np.ndarray:
    """Dequantizes `w` with scale factors `s`."""
    w_padded = _pad(w, block_size)
    num_blocks = w_padded.shape[0] // s.shape[0]
    if zp is None:
        w_padded = w_padded * s.repeat(num_blocks, axis=0)
    else:
        w_padded = (w_padded - zp.repeat(num_blocks, axis=0)) * s.repeat(num_blocks, axis=0)
    return _depad(w_padded, w.shape)


def quantize_rtn(
    onnx_model: onnx.ModelProto,
    gemm_io_type: onnx.TensorProto.DataType,
    block_size: int,
    dq_only: bool = False,
) -> onnx.ModelProto:
    """Quantizes `onnx_model` using the RTN (Round-to-Nearest) algorithm.

    This algorithm computes scale factors by computing s = max(abs(block)) / 8, for each block. The
    quantized weights are computed via Q(w) = round_to_even(w / s), where `round_to_even` denotes
    rounding ties to the nearest even integer (i.e. 1.5, 2.5 both round to 2).

    Always selects the first dimension (0) to block over. This is because we must batch over the Cin
    dimension, and in ONNX, weights are always plugged into the RHS (i.e. y = x @ W).
    """
    logger.info("Starting RTN quantization")
    t_start = time.time()

    graph = gs.import_onnx(onnx_model)
    gemm_nodes = [node for node in graph.nodes if node.op in ["Gemm", "MatMul"]]
    logger.info(f"Found {len(gemm_nodes)} Gemm/MatMul nodes to quantize")

    gemm_tensors = {}
    act_tensors = []
    for gemm in gemm_nodes:
        for in_tensor in gemm.inputs:
            if not isinstance(in_tensor, gs.Constant):
                continue
            if len(in_tensor.values.shape) == 1:
                # 1D blocked quantization not supported.
                continue
            gemm_tensors[in_tensor.name] = in_tensor
            act_tensors.append(gemm.inputs[0])

    gemm_weights = {name: tensor.values for name, tensor in gemm_tensors.items()}
    logger.info(f"Found {len(gemm_weights)} quantizable weights")

    logger.info("Computing scales for gemm weights")
    scales = {}
    for name, w in gemm_weights.items():
        logger.debug(f"Computing scales for weight {name} of shape {w.shape}")
        s, zp = find_scales(np.asarray(w), block_size)
        assert zp is None, "zero-point is not enabled but zp is found non-None"
        scales[name] = s

    # Change the scale type to the expected type, fp16 by default
    for name in scales:
        s = scales[name]
        scales[name] = s.astype(onnx.helper.tensor_dtype_to_np_dtype(gemm_io_type))

    # Change the input activation type to the expected type, fp16 by default
    for act_tensor in act_tensors:
        _change_input_type(onnx_model.graph, act_tensor.name, gemm_io_type)

    # Import the update graph
    graph = gs.import_onnx(onnx_model)

    if dq_only:
        # Calculate actual quantized weights.
        logger.info("Computing quantized weights for DQ-only mode")
        gemm_weights_quantized = {}
        for name, w in gemm_weights.items():
            logger.debug(f"Quantizing weight {name}")
            qw = rtn(np.asarray(w), scales[name], block_size)
            if has_cupy:
                qw = np.asnumpy(qw)
                scales[name] = np.asnumpy(scales[name])
            gemm_weights_quantized[name] = numpy.asarray(qw)

        qdq.insert_dq_nodes(graph, scales, quantized_weights=gemm_weights_quantized)
    else:
        if has_cupy:
            for name in scales:
                scales[name] = np.asnumpy(scales[name])
        qdq.insert_qdq_nodes(graph, scales, weight_map=gemm_tensors)

    logger.info(f"RTN quantization completed in {time.time() - t_start:.2f} seconds")
    return gs.export_onnx(graph)


def quant_tensor(w: np.ndarray, block_size: int, alpha: float = 1.0, use_zero_point: bool = False):
    """Quantize a tensor using alpha etc. and return the quantized tensor."""
    scale, zp = find_scales(w, block_size, alpha, use_zero_point)
    wq = rtn(w, scale, block_size, zp)
    return wq, scale, zp


class AWQClipHelper:
    """AWQ calibration helper class."""

    min_alpha = 0.5
    alpha_step = 0.05

    def __init__(self, w, block_size: int, **kwargs):
        """Initializes AWQClipHelper with a module weight."""
        self.alpha_step = kwargs.get("awqclip_alpha_step", self.alpha_step)
        self.min_alpha = kwargs.get("awqclip_alpha_min", self.min_alpha)
        self.alphas = [
            round(float(k), 2) for k in np.arange(self.min_alpha, 1.0, self.alpha_step)
        ] + [1.0]

        ci, co = w.shape
        self.block_size = block_size if block_size != -1 else w.shape[0]
        w = _pad(w, block_size).T
        self.w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)

        self.loss = {
            k: np.zeros((co, math.ceil(ci / self.block_size)), dtype=np.float32)
            for k in self.alphas
        }
        self.best_loss = np.full_like(self.w_amax, float("inf"))
        self.best_alpha = np.ones_like(self.w_amax)

    def update_best_params(self):
        """Updates the loss dictionary."""
        for alpha, loss in self.loss.items():
            loss = loss.reshape(self.w_amax.shape)
            indices = loss < self.best_loss
            self.best_loss = np.where(indices, loss, self.best_loss)
            self.best_alpha = np.where(indices, alpha, self.best_alpha)


def _clip_search(
    x: np.ndarray, w: np.ndarray, awq_clip: AWQClipHelper, max_tokens: int = 64, **kwargs
):
    """Apply AWQ algorithm on a weight and return optimum alpha.

    This algorithm defines a simple search space for the optimal scales: S = Sx ^ α.
    S is only related to the magnitude of activation Sx, and a single hyper-parameter α is used to balance
    between the protection of salient and non-salient channels. The algorithm finds the best α by a fast grid search
    over the interval of [0, 1] (0 means do not scale; 1 corresponds to the most aggressive scaling).
    Further weight clipping is also applied by minimizing the MSE error.
    """
    # Select max_tokens from input
    co_bsz = kwargs.get("awqclip_bsz_col", 1024)
    x = np.reshape(x, (-1, x.shape[-1]))  # _, ci
    x = x[0 :: max(1, x.shape[0] // max_tokens)]  # max_tokens, ci

    ci, co = w.shape
    block_size = awq_clip.block_size

    # Pad weight and input if necessary
    if ci % block_size != 0:
        w = _pad(w, block_size)
        x = _pad(x.T, block_size).T

    # Make a copy of the original padded weight to quantize with generated scales
    w_copy = copy.deepcopy(w)

    # Reshape weight and input for batch processing over co dimension
    w = w.T  # co, ci
    w = w.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size
    x = x.reshape(1, x.shape[0], -1, block_size)  # 1, max_tokens, n_block, block_size

    # Loop over co dimension of the weight and generate scales
    for co_batch in range(math.ceil(co / co_bsz)):
        slice_s, slice_e = co_batch * co_bsz, min((co_batch + 1) * co_bsz, co)
        weight = w[slice_s:slice_e]
        org_out = np.sum(x * weight, axis=-1)  # co_bsz, max_tokens, n_block

        # Compute loss for each alpha value
        for alpha in awq_clip.loss:
            # Perform QDQ on the whole original weight tensor
            qw, scales, _ = quant_tensor(w_copy, block_size, alpha)
            cur_w = dq_tensor(qw, scales, block_size)

            # Reshape before getting the batch of size co_bsz to multiply with input
            cur_w = cur_w.T  # ci, co -> co, ci
            cur_w = cur_w.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size
            cur_w = cur_w[slice_s:slice_e]

            # Compute loss for each batch
            cur_out = np.sum(x * cur_w, axis=-1)  # co_bsz, max_tokens, n_block
            loss = np.mean(np.power((org_out - cur_out), 2), axis=1)  # co_bsz, n_block
            awq_clip.loss[alpha][slice_s:slice_e] += loss

            if has_cupy:
                np.get_default_memory_pool().free_all_blocks()

    # Update the best alpha value for the weight blocks
    awq_clip.update_best_params()
    if has_cupy:
        np.get_default_memory_pool().free_all_blocks()


def _find_quantizable_weights(
    graph: onnx.GraphProto,
    nodes_to_exclude: list[str],
) -> list[tuple[onnx.ValueInfoProto, onnx.ValueInfoProto, bool, int]]:
    """Finds the quantizable weights from the graph."""
    wa_pack = []
    gemm_nodes = [node for node in graph.node if node.op_type in ["Gemm", "MatMul"]]
    initializer_idxs = {initializer.name: idx for idx, initializer in enumerate(graph.initializer)}
    for gemm in gemm_nodes:
        exclude_this_node = False

        for i in range(len(nodes_to_exclude)):
            if nodes_to_exclude[i] in gemm.name:
                exclude_this_node = True
                break

        if exclude_this_node:
            continue

        if gemm.input[0] in initializer_idxs:
            # Ex. two const input to MatMul_115 in fastvit0.onnx
            # Note. RTN algorithm will quantize these weights though
            continue

        if gemm.input[1] not in initializer_idxs:
            continue

        weight_tensor = graph.initializer[initializer_idxs[gemm.input[1]]]
        if len(weight_tensor.dims) == 1:  # 1D blocked quantization not supported
            continue

        gemm_io_type = cast("int", weight_tensor.data_type)

        act_tensor = onnx.helper.ValueInfoProto()
        act_tensor.name = gemm.input[0]

        # TODO: support transA by transposing activation tensors in _clip_search
        do_transpose = gemm.op_type == "Gemm" and any(
            attr.name == "transB" and attr.i > 0 for attr in gemm.attribute
        )

        wa_pack.append((act_tensor, weight_tensor, do_transpose, gemm_io_type))

    return wa_pack


def _augment_graph(
    graph: onnx.GraphProto,
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
):
    """Extend graph outputs with MatMuls activation input."""
    augmented_outputs = {tensor.name for tensor in graph.output}
    for act_tensor, _, _, _ in wa_pack:
        if act_tensor.name not in augmented_outputs:
            graph.output.append(act_tensor)
            augmented_outputs.add(act_tensor.name)


def _change_input_type(
    graph: onnx.GraphProto, input_name: str, gemm_io_type: onnx.TensorProto.DataType
):
    # Find the corresponding value info in the graph
    done = False
    for value_info in graph.value_info:
        if value_info.name == input_name:
            value_info.type.tensor_type.elem_type = gemm_io_type
            done = True
            break

    if not done:
        # If input not in value_info, it must be a graph input
        for input_info in graph.input:
            if input_info.name == input_name:
                input_info.type.tensor_type.elem_type = gemm_io_type
                break


def _quantize_awq_clip(
    onnx_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool,
    calibration_eps: list[str],
    block_size: int,
    force_fp16: bool = False,
    nodes_to_exclude: list[str] = [],
    **kwargs: Any,
) -> onnx.ModelProto:
    """Quantizes `onnx_model` using the Activation aware quantization a.k.a AWQ algorithm."""
    logger.info("Quantizing model using AWQ clip algorithm")
    t = time.time()
    augmented_model = copy.deepcopy(onnx_model)
    graph = augmented_model.graph

    nodes_to_exclude = expand_node_names_from_patterns(graph, nodes_to_exclude)
    # Collect quantizable weight tensors
    wa_pack = _find_quantizable_weights(graph, nodes_to_exclude)

    # Add input activations to graph output
    _augment_graph(augmented_model.graph, wa_pack)
    logger.info(f"Augmenting took {time.time() - t} seconds")

    scales = {}
    gemm_weights_quantized = {}

    t = time.time()

    # Create a temp file for augmented model
    augmented_onnx_file, augmented_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(augmented_onnx_file)

    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)
    logger.info(f"Saving the model took {time.time() - t} seconds")

    # Creating inference session and preparing inputs for calibration
    session = create_inference_session(augmented_onnx_path, calibration_eps)
    inputs = []
    for inp_d in data_reader:
        inputs.append(inp_d)
        assert isinstance(inp_d, dict)

    # Apply AWQ clip on selected weights
    t = time.time()
    alphas = {}
    for i in tqdm(range(len(wa_pack)), desc="Running clip search..."):
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]

        # First capture all the  activation values after calibration data sweep
        output_dicts = {}
        for inp_d in inputs:
            np_inp_d = {name: numpy.asarray(tensor) for name, tensor in inp_d.items()}
            output = session.run([act_tensor.name], np_inp_d)
            out = np.asarray(output[0])
            output_dicts.setdefault(act_tensor.name, []).append(out)

        # Concatenating the activation tensors over all calib data
        x = np.concatenate(output_dicts[act_tensor.name], axis=0)  # n_token, ci
        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T
        w = np.asarray(w)

        awq_clip = AWQClipHelper(w, block_size, **kwargs)
        _clip_search(x, w, awq_clip, **kwargs)
        alphas[weight_tensor.name] = awq_clip.best_alpha

    logger.info(f"Clip search for all weights took {time.time() - t} seconds")

    del session

    # Compute quantized weights and scales which are needed for DQ nodes
    t = time.time()
    for i in tqdm(range(len(wa_pack)), desc="Quantizing the weights..."):
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]
        gemm_io_type = cast("onnx.TensorProto.DataType", gemm_io_type)

        if force_fp16:
            gemm_io_type = onnx.TensorProto.FLOAT16

        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T
        w = np.asarray(w)

        alpha = alphas.get(weight_tensor.name, 1)
        qw, scale, _ = quant_tensor(w, block_size, alpha)
        if has_cupy:
            qw = np.asnumpy(qw)
            scale = np.asnumpy(scale)
        if do_transpose:
            qw = qw.T
            scale = scale.T
        scales[weight_tensor.name] = scale.astype(
            onnx.helper.tensor_dtype_to_np_dtype(gemm_io_type)
        )
        gemm_weights_quantized[weight_tensor.name] = numpy.asarray(qw).astype(numpy.int8)

        # Change the input activation type to the expected type, fp16 by default
        # TODO: cast input C for Gemm
        _change_input_type(onnx_model.graph, act_tensor.name, gemm_io_type)

    logger.info(f"Quantizing actual weights took {time.time() - t} seconds")

    t = time.time()
    graph_gs = gs.import_onnx(onnx_model)
    dq_node_attributes = {"axis": 0, "block_size": block_size}
    qdq.insert_dq_nodes(
        graph_gs, scales, quantized_weights=gemm_weights_quantized, attributes=dq_node_attributes
    )
    logger.info(f"Inserting DQ nodes took {time.time() - t} seconds")

    logger.info("Exporting the quantized graph")
    t = time.time()
    model = gs.export_onnx(graph_gs)
    # Set ir_version to 10, remove it once ORT supports ir_version 11
    model.ir_version = 10
    logger.info(f"Exporting took {time.time() - t} seconds")

    try:
        os.remove(augmented_onnx_path)
        if use_external_data_format:
            os.remove(augmented_onnx_path + "_data")
    except OSError:
        logger.warn("Augmented ONNX model or external data file was not found")

    return model


class AWQLiteHelper:
    """AWQ Lite calibration helper class."""

    alpha_step = 0.1

    def __init__(self, x, w, block_size: int, **kwargs):
        """Initializes AWQLiteHelper with a module weight."""
        self.alpha_step = kwargs.get("awqlite_alpha_step", self.alpha_step)
        self.block_size = block_size if block_size != -1 else w.shape[0]
        # w = _pad(w, block_size).T
        # self.w_amax = np.abs(w.reshape(-1, block_size)).max(axis=-1)
        self.weight_scale = None if w is None else get_weight_scale(w, self.block_size)
        self.act_scale = None if x is None else get_act_scale(x)
        self.loss = {
            k.item(): 0.0
            for k in np.arange(0, 1.0 + AWQLiteHelper.alpha_step, AWQLiteHelper.alpha_step)
        }
        self.best_scale = None
        self.best_alpha = None

    def update_best_params(self):
        """Updates best-alpha and best-scale."""
        self.best_alpha = min(self.loss, key=self.loss.__getitem__)
        self.best_scale = get_scale(
            self.act_scale,
            self.weight_scale,
            self.best_alpha,
        )


def get_act_scale(x):
    """Get scale tensors for inputs."""
    return x.__abs__().mean(0)


def get_weight_scale(weight, block_size=None):
    """Get scale tensors for weights."""
    org_shape = weight.shape
    slice_after_padding = None
    if block_size:
        if org_shape[0] % block_size != 0:
            slice_after_padding = slice(org_shape[0])
            weight = _pad(weight, block_size)
            org_shape = weight.shape
        weight = weight.reshape(block_size, -1)
    weight_abs_amax = weight.__abs__().max(axis=0, keepdims=True)
    scale = weight.__abs__() / (weight_abs_amax + np.finfo(weight_abs_amax.dtype).tiny)
    scale = scale.reshape(org_shape)
    if slice_after_padding is not None:
        scale = scale[slice_after_padding, ...]
    scale = scale.mean(1)
    return scale


def get_scale(x_max, w_max, alpha):
    """Get AWQ lite scales as described by 's' in the paper."""
    scales = (x_max.__pow__(alpha) / (w_max.__pow__(1 - alpha) + np.finfo(w_max.dtype).tiny)).clip(
        min=1e-4, max=1e4
    )
    scales = scales / np.sqrt(scales.max() * scales.min())
    return scales


def run_awq_scale_search_per_node(
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
    augmented_onnx_path,
    block_size,
    use_zero_point,
    session,
    awq_lite,
    inputs,
    tqdm_msg_append_str,
    enable_weight_clipping,
    enable_fast_path_using_high_sysram,
    output_data,
    clip_alphas,
    **kwargs: Any,
):
    """Method that iterates over each quantizable node for scale search."""
    assert len(awq_lite) == len(wa_pack)

    for i in tqdm(
        range(len(wa_pack)),
        desc="Running AWQ scale search per node" + tqdm_msg_append_str,
    ):
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]

        output_dicts = {}

        if enable_fast_path_using_high_sysram:
            assert len(output_data) > 0, (
                "fast-path is enabled but node-inputs are not pre-determined before grid search"
            )
            node_inputs = []
            for j in range(len(output_data)):
                node_inputs.append(np.asarray(output_data[j][i]))
                # want to free system RAM asap since that data is here copied to GPU for this node
                output_data[j][i] = None
            output_dicts[act_tensor.name] = node_inputs
        else:
            # First capture all the  activation values after calibration data sweep
            for inp_d in inputs:
                np_inp_d = {name: numpy.asarray(tensor) for name, tensor in inp_d.items()}
                output = session.run([act_tensor.name], np_inp_d)
                out = np.asarray(output[0])
                output_dicts.setdefault(act_tensor.name, []).append(out)

        # Concatenating the activation tensors over all calib data
        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T
        w = np.asarray(w)

        x = np.concatenate(output_dicts[act_tensor.name], axis=0).reshape(
            (-1, w.shape[0])
        )  # n_token, ci

        awq_lite[i] = AWQLiteHelper(x, w, block_size, **kwargs)

        out_actual = x.__matmul__(w)

        for alpha in awq_lite[i].loss:
            awq_scale = get_scale(
                awq_lite[i].act_scale,
                awq_lite[i].weight_scale,
                alpha,
            )
            x_scaled = x * 1.0 / awq_scale
            w_scaled = w * awq_scale[:, np.newaxis]

            qw, scale, zp = quant_tensor(w_scaled, block_size, use_zero_point=use_zero_point)
            dqw = dq_tensor(qw, scale, block_size, zp)
            out_curr = x_scaled.__matmul__(dqw)
            loss = np.mean(np.power((out_actual - out_curr), 2))
            del out_curr
            awq_lite[i].loss[alpha] = loss

        awq_lite[i].update_best_params()
        if enable_weight_clipping:
            w = w * (awq_lite[i].best_scale[:, np.newaxis])
            awq_clip = AWQClipHelper(w, block_size, **kwargs)
            _clip_search(x, w, awq_clip, **kwargs)
            clip_alphas[weight_tensor.name] = awq_clip.best_alpha
        del x, w, out_actual, output_dicts
        if has_cupy:
            np.get_default_memory_pool().free_all_blocks()

    return awq_lite, clip_alphas


def get_act_to_weight_map_and_act_to_wa_pack_map(
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
):
    """Method to return subgraph related maps based on activation-name as key.

    This method returns 2 maps:
    (a) map of act-name to input-node's weights dimensions
    (b) map of act-name to wa_pack indices with same act-name
    """
    act_to_wa_pack_map = {}
    act_to_quant_nodes_weight_shape_map = {}
    for i in tqdm(range(len(wa_pack)), desc="Getting activation names maps..."):
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]
        # wa_pack index is stored in map to represent quant nodes
        act_to_wa_pack_map.setdefault(act_tensor.name, []).append(i)
        act_to_quant_nodes_weight_shape_map.setdefault(act_tensor.name, []).append(
            weight_tensor.dims
        )
        if len(act_to_quant_nodes_weight_shape_map[act_tensor.name]) > 1:
            assert (
                weight_tensor.dims[0] == act_to_quant_nodes_weight_shape_map[act_tensor.name][0][0]
            )

    return act_to_wa_pack_map, act_to_quant_nodes_weight_shape_map


def get_x_w_mean_for_subgraph(
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
    wa_pack_idx_list,
    augmented_onnx_path,
    x,
    block_size,
):
    """This method returns x-mean and w-mean."""
    x_sum = np.sum(np.abs(x), axis=0, dtype=np.float32)
    x_m = x_sum / x.shape[0]
    x_m = x_m.astype(x.dtype)
    del x_sum

    w_concatenated = None
    for wa_pack_idx in wa_pack_idx_list:
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[wa_pack_idx]
        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T
        w = np.asarray(w)
        if w_concatenated is None:
            w_concatenated = w
        else:
            w_concatenated = np.concatenate([w_concatenated, w], axis=1)
            del w

    if has_cupy:
        np.get_default_memory_pool().free_all_blocks()

    assert w_concatenated is not None

    org_shape = w_concatenated.shape
    w_concatenated = w_concatenated.reshape(block_size, -1)
    div_by = np.amax(np.abs(w_concatenated), axis=0)
    div_by = div_by + 1e-6  # damping factor 1e-6 is taken from AutoAWQ
    w_concatenated = np.abs(w_concatenated) / div_by
    w_concatenated = w_concatenated.reshape(org_shape)
    w_m = np.mean(w_concatenated, axis=1)

    del w_concatenated
    gc.collect()
    if has_cupy:
        np.get_default_memory_pool().free_all_blocks()

    x_m = x_m.reshape(-1)
    w_m = w_m.reshape(-1)
    assert x_m.shape == w_m.shape

    return x_m, w_m


def run_awq_scale_search_per_subgraph(
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
    act_to_wa_pack_map,
    act_to_quant_nodes_weight_shape_map,
    augmented_onnx_path,
    block_size,
    use_zero_point,
    session,
    awq_lite,
    inputs,
    tqdm_msg_append_str,
    **kwargs: Any,
):
    """Method that iterates over each quantizable subgraph/siblings for scale search."""
    for act, wa_pack_idx_list in tqdm(
        act_to_wa_pack_map.items(),
        desc="Running AWQ scale search per subgraph" + tqdm_msg_append_str,
    ):
        output_dicts = {}

        for inp_d in inputs:
            np_inp_d = {name: numpy.asarray(tensor) for name, tensor in inp_d.items()}
            output = session.run([act], np_inp_d)
            out = np.asarray(output[0])
            output_dicts.setdefault(act, []).append(out)
        assert len(act_to_quant_nodes_weight_shape_map[act]) > 0
        common_dim = act_to_quant_nodes_weight_shape_map[act][0][0]
        x = np.concatenate(output_dicts[act], axis=0).reshape((-1, common_dim))  # n_token, ci

        del output_dicts

        x_m, w_m = get_x_w_mean_for_subgraph(
            wa_pack, wa_pack_idx_list, augmented_onnx_path, x, block_size
        )

        best_error = float("inf")
        best_alpha = None
        best_scale = None

        out_actual = {}

        alpha_step = kwargs.get("awqlite_alpha_step", AWQLiteHelper.alpha_step)
        alpha_values = np.arange(0, 1.0 + alpha_step, alpha_step)

        for alpha in alpha_values:
            loss = 0.0
            # offset and clip value in the following formula is taken from AutoAWQ
            awq_scale = (x_m.__pow__(alpha) / (w_m.__pow__(1 - alpha) + 1e-4)).clip(min=1e-4)
            awq_scale = awq_scale / np.sqrt(awq_scale.max() * awq_scale.min())
            awq_scale[np.isinf(awq_scale)] = 1
            awq_scale[np.isnan(awq_scale)] = 1
            for wa_pack_idx in wa_pack_idx_list:
                _, weight_tensor, do_transpose, _ = wa_pack[wa_pack_idx]
                w = numpy_helper.to_array(
                    weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
                ).copy()
                if do_transpose:
                    w = w.T
                w = np.asarray(w)
                out_act = out_actual.get(wa_pack_idx)
                if out_act is None:
                    out_act = x.__matmul__(w)
                    out_actual[wa_pack_idx] = out_act
                assert out_act is not None
                x_scaled = x * 1.0 / awq_scale
                w_scaled = w * awq_scale[:, np.newaxis]
                qw, scale, zp = quant_tensor(w_scaled, block_size, use_zero_point=use_zero_point)
                dqw = dq_tensor(qw, scale, block_size, zp)
                out_curr = x_scaled.__matmul__(dqw)
                loss += np.mean(np.power((out_act - out_curr), 2))
                del out_curr, out_act
                del w, x_scaled, w_scaled, qw, scale, dqw
                if has_cupy:
                    np.get_default_memory_pool().free_all_blocks()
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_alpha = alpha
                best_scale = awq_scale

        for wa_pack_idx in wa_pack_idx_list:
            assert np.isnan(best_scale).sum() == 0, best_scale
            assert awq_lite[wa_pack_idx] is None
            awq_lite[wa_pack_idx] = AWQLiteHelper(None, None, block_size)
            awq_lite[wa_pack_idx].best_alpha = best_alpha
            awq_lite[wa_pack_idx].best_scale = best_scale
            assert awq_lite[wa_pack_idx] is not None

        del x, out_actual, x_m, w_m
        if has_cupy:
            np.get_default_memory_pool().free_all_blocks()

    return awq_lite


def get_parent_child_nodes_map(
    graph: onnx.GraphProto,
    wa_pack: list[tuple[gs.Tensor, gs.Tensor, bool, int]],
):
    """Get mapping of parent nodes to their MatMul/Gemm nodes with quantizable weights."""
    parent_child_nodes_map = {}
    output_name_to_node = get_tensor_producer_nodes(graph)
    input_name_to_nodes = get_tensor_consumer_nodes(graph)

    for act_tensor, _, _, _ in wa_pack:
        parent_name = output_name_to_node[act_tensor.name].name
        parent_child_nodes_map[parent_name] = []
        for node in input_name_to_nodes[act_tensor.name]:
            if node.op_type in ["Gemm", "MatMul"]:
                parent_child_nodes_map[parent_name].append(node)

    return parent_child_nodes_map, input_name_to_nodes


def _quantize_awq_lite(
    onnx_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool,
    calibration_eps: list[str],
    block_size: int,
    force_fp16: bool = False,
    enable_fast_path_using_high_sysram: bool = False,
    enable_weight_clipping: bool = False,
    use_zero_point: bool = False,
    nodes_to_exclude: list[str] = [],
    **kwargs: Any,
) -> onnx.ModelProto:
    """Quantizes `onnx_model` using the Activation aware quantization a.k.a AWQ algorithm."""
    logger.info("Quantizing model using AWQ lite algorithm")
    t = time.time()

    run_per_subgraph = kwargs.get("awqlite_run_per_subgraph", False)
    fuse_nodes = kwargs.get("awqlite_fuse_nodes", True)

    # TODO - add weight-clipping support in per-subgraph implementation
    assert not run_per_subgraph or not enable_weight_clipping

    # TODO - evaluate/add sysram based fast-path support in per-subgraph implementation
    assert not run_per_subgraph or not enable_fast_path_using_high_sysram

    augmented_model = copy.deepcopy(onnx_model)
    graph = augmented_model.graph

    nodes_to_exclude = expand_node_names_from_patterns(graph, nodes_to_exclude)
    # Collect quantizable weight tensors
    wa_pack = _find_quantizable_weights(graph, nodes_to_exclude)
    if fuse_nodes:
        parent_child_nodes_map, input_name_to_nodes = get_parent_child_nodes_map(
            onnx_model.graph, wa_pack
        )

    # Add input activations to graph output
    _augment_graph(augmented_model.graph, wa_pack)
    logger.info(f"Augmenting took {time.time() - t} seconds")

    zero_points = {}
    scales = {}
    gemm_weights_quantized = {}
    input_tensors = {}
    pre_quant_scale = {}

    t = time.time()

    # Create a temp file for augmented model
    augmented_onnx_file, augmented_onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(augmented_onnx_file)

    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)
    logger.info(f"Saving the model took {time.time() - t} seconds")

    # Creating inference session and preparing inputs for calibration
    session = create_inference_session(augmented_onnx_path, calibration_eps)
    inputs = []
    for inp_d in data_reader:
        inputs.append(inp_d)
        assert isinstance(inp_d, dict)

    gc.collect()

    output_data = []

    if enable_fast_path_using_high_sysram:
        logger.info("Fast-path-using-high-sysram is enabled\n")

        tensor_names_list = []
        for i in tqdm(range(len(wa_pack)), desc="Getting tensor names..."):
            act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]
            tensor_names_list.append(act_tensor.name)

        for i in tqdm(range(len(inputs)), desc="Caching activations..."):
            inp_d = inputs[i]
            np_inp_d = {name: numpy.asarray(tensor) for name, tensor in inp_d.items()}
            output = session.run(tensor_names_list, np_inp_d)
            output_data.append(output)

        del session
        session = None
        gc.collect()

    # Apply AWQ lite on selected weights
    t = time.time()
    awq_lite = [None] * len(wa_pack)
    clip_alphas = {}

    msg = "..."
    if enable_weight_clipping:
        msg = " and clip-range search..."

    act_to_wa_pack_map, act_to_quant_nodes_weight_shape_map = (
        get_act_to_weight_map_and_act_to_wa_pack_map(wa_pack)
    )

    if run_per_subgraph:
        awq_lite = run_awq_scale_search_per_subgraph(
            wa_pack,
            act_to_wa_pack_map,
            act_to_quant_nodes_weight_shape_map,
            augmented_onnx_path,
            block_size,
            use_zero_point,
            session,
            awq_lite,
            inputs,
            msg,
            **kwargs,
        )
    else:
        awq_lite, clip_alphas = run_awq_scale_search_per_node(
            wa_pack,
            augmented_onnx_path,
            block_size,
            use_zero_point,
            session,
            awq_lite,
            inputs,
            msg,
            enable_weight_clipping,
            enable_fast_path_using_high_sysram,
            output_data,
            clip_alphas,
            **kwargs,
        )
    assert len(awq_lite) == len(wa_pack)
    for i in range(len(awq_lite)):
        assert awq_lite[i] is not None

    if enable_weight_clipping:
        assert len(clip_alphas.keys()) == len(wa_pack)

    logger.info("AWQ scale search" + msg.strip(".") + f" took {time.time() - t} seconds")

    if session is not None:
        del session
        session = None
    if has_cupy:
        np.get_default_memory_pool().free_all_blocks()
    del output_data
    gc.collect()

    # Compute quantized weights and scales which are needed for DQ nodes
    t = time.time()
    # Use a common mean scale for weights within a sub-graph
    if fuse_nodes and not run_per_subgraph:
        for wa_pack_idx_list in act_to_wa_pack_map.values():
            group_awq_scale = [
                awq_lite[wa_pack_idx].best_scale[:, np.newaxis] for wa_pack_idx in wa_pack_idx_list
            ]
            mean_awq_scale = np.concatenate(group_awq_scale, axis=1)
            mean_awq_scale = mean_awq_scale.mean(axis=1)
            for wa_pack_idx in wa_pack_idx_list:
                awq_lite[wa_pack_idx].best_scale = mean_awq_scale

    for i in tqdm(range(len(wa_pack)), desc="Quantizing the weights..."):
        act_tensor, weight_tensor, do_transpose, gemm_io_type = wa_pack[i]
        gemm_io_type = cast("onnx.TensorProto.DataType", gemm_io_type)

        if force_fp16:
            gemm_io_type = onnx.TensorProto.FLOAT16

        w = numpy_helper.to_array(
            weight_tensor, base_dir=os.path.dirname(augmented_onnx_path)
        ).copy()
        if do_transpose:
            w = w.T
        w = np.asarray(w)

        w_scaled = w * awq_lite[i].best_scale[:, np.newaxis]
        alpha = clip_alphas.get(weight_tensor.name, 1)
        assert enable_weight_clipping or (alpha == 1), (
            "clip range enabled without enabling weight-clipping param"
        )
        qw, scale, zp = quant_tensor(w_scaled, block_size, alpha, use_zero_point=use_zero_point)
        assert use_zero_point is True or zp is None, "zp is not according to use-zero-point setting"
        if do_transpose:
            qw = qw.T
            scale = scale.T
            if zp is not None:
                zp = zp.T
        if has_cupy:
            qw = np.asnumpy(qw)
            scale = np.asnumpy(scale)
            if zp is not None:
                zp = np.asnumpy(zp)
        scales[weight_tensor.name] = scale.astype(
            onnx.helper.tensor_dtype_to_np_dtype(gemm_io_type)
        )
        weight_dtype = numpy.int8
        if zp is not None:
            zero_points[weight_tensor.name] = numpy.asarray(zp).astype(numpy.uint8)
            weight_dtype = numpy.uint8
        gemm_weights_quantized[weight_tensor.name] = numpy.asarray(qw).astype(weight_dtype)
        input_tensors[weight_tensor.name] = act_tensor.name
        pqs_value = (
            awq_lite[i]
            .best_scale[:, np.newaxis]
            .astype(onnx.helper.tensor_dtype_to_np_dtype(gemm_io_type))
        ).T
        if has_cupy:
            pqs_value = np.asnumpy(pqs_value)
        pre_quant_scale[weight_tensor.name] = pqs_value

        # Change the input activation type to the expected type, fp16 by default
        # TODO: cast input C for Gemm
        _change_input_type(onnx_model.graph, act_tensor.name, gemm_io_type)

    logger.info(f"Quantizing actual weights took {time.time() - t} seconds")

    # Fuse Mul nodes with parent node if possible
    if fuse_nodes:
        logger.info("Fusing pre-quant scale Mul nodes with parent node")
        t = time.time()
        updated_nodes = set()
        name_to_node_map = {node.name: node for node in onnx_model.graph.node}
        initializer_map = {
            initializer.name: initializer for initializer in onnx_model.graph.initializer
        }
        for parent, child_nodes in parent_child_nodes_map.items():
            if parent == "root_0":
                continue
            parent = name_to_node_map[parent]
            if parent.name in updated_nodes:
                continue
            # When fuse_nodes or run_per_subgraph is True,
            # scales computed for each child_nodes will be same.
            # Hence, picking pre_quant_scale corresponding to any child_nodes is acceptable
            input_scale = np.asarray(pre_quant_scale[child_nodes[0].input[1]])
            weight_tensor_names = [node.input[1] for node in child_nodes]
            if (
                is_fusible_scaling_op(parent.op_type)
                and not all(initializer_map.get(inp) is None for inp in parent.input)
                and len(input_name_to_nodes[child_nodes[0].input[0]]) == len(child_nodes)
            ):
                for inp in parent.input:
                    if initializer_map.get(inp) is not None:
                        tensor = initializer_map[inp]
                        tensor_array = numpy_helper.to_array(
                            tensor,
                            base_dir=os.path.dirname(augmented_onnx_path),
                        )
                        new_tensor = np.asarray(tensor_array) / input_scale
                        new_tensor = numpy_helper.from_array(new_tensor.get(), tensor.name)
                        # replace initializer with new scaled array
                        tensor.CopyFrom(new_tensor)
                        for w_name in weight_tensor_names:
                            del pre_quant_scale[w_name]
                updated_nodes.add(parent.name)
            else:
                scale_tensor = onnx.helper.make_tensor(
                    name=parent.output[0] + "_pre_quant_scale",
                    data_type=onnx.helper.np_dtype_to_tensor_dtype(input_scale.dtype),
                    dims=input_scale.shape,
                    vals=(1.0 / input_scale).flatten().tolist(),
                )
                mul_op_name = parent.output[0] + "_pre_quant_scale_out"
                mul_node = onnx.helper.make_node(
                    "Mul",
                    inputs=[child_nodes[0].input[0], scale_tensor.name],
                    outputs=[mul_op_name],
                    name=child_nodes[0].input[0] + "_pre_quant_scale_mul",
                )
                for node in child_nodes:
                    node.input[0] = mul_node.output[0]
                for w_name in weight_tensor_names:
                    del pre_quant_scale[w_name]
                onnx_model.graph.initializer.append(scale_tensor)
                onnx_model.graph.node.append(mul_node)

        logger.info(f"Fusing pre-quant scale Mul nodes took {time.time() - t} seconds")

    logger.info(
        "Inserting DQ nodes and input_pre_quant_scale node using quantized weights and scales"
    )
    t = time.time()
    graph_gs = gs.import_onnx(onnx_model)
    dq_node_attributes = {"axis": 0, "block_size": block_size}
    qdq.insert_dq_nodes(
        graph_gs,
        scales,
        quantized_weights=gemm_weights_quantized,
        attributes=dq_node_attributes,
        zero_points=zero_points if use_zero_point else None,
    )
    if pre_quant_scale:
        qdq.insert_pre_quant_scale_nodes(graph_gs, input_tensors, pre_quant_scale)

    logger.info(f"Inserting nodes took {time.time() - t} seconds")

    logger.info("Exporting the quantized graph")
    t = time.time()
    model = gs.export_onnx(graph_gs)
    # Set ir_version to 10, remove it once ORT supports ir_version 11
    model.ir_version = 10
    logger.info(f"Exporting took {time.time() - t} seconds")

    try:
        os.remove(augmented_onnx_path)
        if use_external_data_format:
            os.remove(augmented_onnx_path + "_data")
    except OSError:
        logger.error("Augmented ONNX model or external data file was not found")

    return model


def quantize(
    onnx_path: str | onnx.ModelProto,
    calibration_method: str = "awq_lite",
    calibration_data_reader: CalibrationDataReader = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    use_external_data_format: bool = True,
    use_zero_point: bool = False,
    block_size: int | None = None,
    nodes_to_exclude: list[str] | None = [r"/lm_head"],
    log_level: str = "INFO",
    **kwargs: Any,
) -> onnx.ModelProto:
    """Applies INT4 Weight-Only-Quantization (WoQ) to an ONNX model.

    Currently, only ``MatMul`` nodes quantization is supported.

    Args:
        onnx_path: Input ONNX model (base model)
        calibration_method: It determines the quantization algorithm. Few important algorithms are:

                - *awq_lite*: Applies AWQ scaling (Alpha search) followed by INT4 quantization.
                - *awq_clip*: Executes weight clipping and INT4 quantization.
        calibration_data_reader: It can be assigned a list of model inputs. If it is ``None``, then
                a randomly generated model input will be used for calibration in AWQ implementation.
        calibration_eps: It denotes ONNX Execution Providers (EPs) to use for base model calibration.
                This list of EPs is then passed to create-session API of the onnxruntime (ORT) to
                perform base model calibration.

                .. note::

                    Make sure that ORT package for chosen calibration-EPs is setup properly along
                    with their dependencies.
        use_external_data_format: If True, save tensors to external file(s) for quantized model.
        use_zero_point: If True, enables zero-point based quantization.
        block_size: Block size parameter for int4 quantization. Default value of 128 is used for
                    ``block_size`` parameter.
        nodes_to_exclude: List of node-names (or substrings of node-names) denoting the nodes to
                    exclude from quantization.

                .. note::

                    By default, ``lm-head`` node is NOT quantized.
        kwargs: It denotes additional keyword arguments for int4 quantization. It includes:

                - **awqlite_alpha_step** (float): Step size to find best Alpha in awq-lite.Range: [0, 1].
                                              Default: 0.1.
                - **awqclip_alpha_step** (float): Step size to find best Alpha in awq-clip.
                                              Default: 0.05
                - **awqclip_alpha_min** (float): Minimum threshold for weight-clipping in awq-clip.
                                             Default: 0.5.
                - **awqclip_bsz_col** (int): Batch size for processing the column dimension in awq-clip.
                                         Default: 1024.
        log_level: The logging level to use (default: logging.INFO)
    **Returns**: A quantized ONNX model in ONNX ModelProto format.
    """
    configure_logging(level=log_level.upper())
    logger.info(f"Starting INT4 quantization with method: {calibration_method}")
    t_start = time.time()

    if cupy_warning_msg:
        logger.warning(cupy_warning_msg)

    # Check if block_size is None and set default to 128
    if block_size is None:
        block_size = 128
        logger.info(f"Using default block size: {block_size}")

    gemm_io_type: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT

    # set config params
    nodes_to_exclude = nodes_to_exclude or []
    logger.debug(f"Excluding nodes matching patterns: {nodes_to_exclude}")

    # Patch GS modules to support INT4.
    patch_gs_modules()

    if isinstance(onnx_path, str):
        logger.info(f"Loading ONNX model from path: {onnx_path}")
        onnx_model = onnx.load(onnx_path, load_external_data=True)
    else:
        onnx_model = onnx_path

    # Initialize calibration_data_reader if not provided
    if calibration_data_reader is None:
        calibration_data_reader = RandomDataProvider(onnx_model)

    if "trt" in calibration_method:
        qdq.use_trt_qdq_ops()

    if calibration_method in ["rtn", "rtn_dq", "rtn_trt", "rtn_trt_dq"]:
        onnx_model = quantize_rtn(
            onnx_model, gemm_io_type, block_size, dq_only="dq" in calibration_method
        )
    elif calibration_method in ["awq_lite", "awq_full"]:
        do_weight_clipping = False
        if calibration_method == "awq_full":
            do_weight_clipping = True
            logger.info("Using AWQ full with weight clipping")
        else:
            logger.info("Using AWQ lite")
        onnx_model = _quantize_awq_lite(
            onnx_model,
            calibration_data_reader,
            use_external_data_format,
            calibration_eps,
            block_size,
            nodes_to_exclude=nodes_to_exclude,
            use_zero_point=use_zero_point,
            enable_weight_clipping=do_weight_clipping,
            **kwargs,
        )
    elif calibration_method in ["awq_clip", "awq_clip_trt"]:
        onnx_model = _quantize_awq_clip(
            onnx_model,
            calibration_data_reader,
            use_external_data_format,
            calibration_eps,
            block_size,
            nodes_to_exclude=nodes_to_exclude,
            **kwargs,
        )
    else:
        raise RuntimeError(f"Unsupported calibration method: '{calibration_method}'")

    logger.info(f"INT4 Quantization completed in {time.time() - t_start:.2f} seconds")
    return onnx_model
