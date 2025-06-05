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

"""Provides some basic utilities that can be used in quantize() methods."""

from collections.abc import Sequence

import numpy as np

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


def pack_float32_to_4bit_optimized(array: np.ndarray | Sequence, signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two concecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    done here mainly rely on moving some common code out of the per-element function calls or loops, thereby making
    them per-input-array, instead of per-input-element. The remaining logic should largely remain as is.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0]))

    inp_arr_len = array_flat.size
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    array_flat = np.clip(array_flat, clip_low, clip_high)
    array_flat = np.rint(array_flat).astype(dtype)
    assert len(array_flat) % 2 == 0, "array length must be even at this point"
    assert len(array_flat) == inp_arr_len, "output-length must match the input-length"
    output_list = [
        ((array_flat[i + 1] << 4) | (array_flat[i] & 0x0F)) for i in range(0, inp_arr_len, 2)
    ]
    arr = np.array(output_list)
    return arr.astype(np.uint8)


def pack_float32_to_4bit_cpp_based(array: np.ndarray | Sequence, signed: bool) -> np.ndarray:
    """Convert an array of float32 value to a 4bit data-type and pack every two concecutive elements in a byte.

    This is the optimized version of pack_float32_to_4bit() utility in ONNX helper file. The basic optimizations
    here is to implement this round_and_pack logic in C++, which is supposed to be faster.

    Args:
        array: array of float to convert and pack
        signed: Whether the 4 bit variant is signed or unsigned

    Returns:
        Packed array with size `ceil(array.size/2)` (single dimension).
    """
    from .extensions import round_and_pack_ext

    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=np.float32)

    # - Currently, FP32, FP64, UINT8 and INT8 dtypes have C++ implementation of the round-and-pack
    # logic.
    # - With above, FP16 should also get "implicitly" supported due to possible type promotion to higher
    # precision float types (e.g. float32 or float64). So, it is mentioned as supported below.
    # - We can add support for other dtypes as and when needed.
    use_python_version = False
    if round_and_pack_ext is None or array.dtype not in [
        "float",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
    ]:
        use_python_version = True

    array_flat = array.ravel()
    is_odd_volume = np.prod(array.shape) % 2 == 1
    if is_odd_volume:
        array_flat = np.append(array_flat, np.array([0], array_flat.dtype))

    inp_arr_len = array_flat.size

    assert inp_arr_len % 2 == 0, "input array length must be even at this point"

    if use_python_version:
        numpy_out = pack_float32_to_4bit_optimized(array_flat, signed)
    else:
        numpy_out = np.zeros([1, int(inp_arr_len / 2)], dtype=np.int8)
        numpy_out = numpy_out.ravel()
        ret = round_and_pack_ext.round_and_pack(
            signed, array_flat, array_flat.size, numpy_out, numpy_out.size
        )
        assert ret == inp_arr_len / 2, "Unexpected output length"
        numpy_out = numpy_out.astype(np.uint8)

    return numpy_out


def get_weights_scaling_factor_2(input: np.ndarray):
    """Returns per tensor weight scaling factor."""
    per_block_scale_amax = np.max(np.abs(input)) / 6.0
    per_block_quant_scale = per_block_scale_amax / 448.0
    if per_block_quant_scale == 0:
        per_block_quant_scale = 1.0
    return np.float32(per_block_quant_scale)


def get_weights_scaling_factor(
    input: np.ndarray, block_size: int, weights_scaling_factor_2: np.float32
):
    """Returns quantized per block weight scaling factor."""
    # Get per_block amax
    [n, k] = input.shape
    assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

    assert k % block_size == 0, (
        "Weight shape is not divisible for block size for block quantiation."
    )

    input = input.reshape(n, k // block_size, block_size)
    # Get per block amax
    per_block_amax = np.max(np.abs(input), axis=-1)
    # Get per-block-scale
    per_block_scale = per_block_amax / 6.0
    # Quantize per_block_scale to FP8
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    # Set all zero values in scale to 1.0
    zero_mask = per_block_scale == 0
    q_per_block_scale[zero_mask] = 1.0
    return q_per_block_scale.astype(np.float32)


def quantize(
    input: np.ndarray,
    block_size: int,
    weights_scaling_factor: np.ndarray,
    weights_scaling_factor_2: np.ndarray,
):
    """Converting a tensor to a quantized format based on NVFP4 quantization."""
    # Reshape the weight and scale factors
    input = input.reshape((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights
    scaled_weight = input / (
        np.expand_dims(weights_scaling_factor, axis=-1) * weights_scaling_factor_2
    )

    # Reshape weights to original and return
    return scaled_weight.reshape(scaled_weight.shape[0], -1)


def get_amax(weight: np.ndarray, quant_axis: int, block_size: int) -> np.ndarray:
    """Returns the amax of the weight tensor along the specified axis for a given block size.

    Only 2D and 3D tensors are supported.

    Args:
        weight: The weight tensor.
        quant_axis: The axis to quantize.
        block_size: The block size.

    Returns:
        The amax of the weight tensor.
    """
    rank = weight.ndim
    if quant_axis == -1:
        quant_axis = rank - 1
    assert rank in [2, 3], "Weight must be a 2D or 3D tensor"
    if rank == 3:
        d0, d1, d2 = weight.shape
        if quant_axis == 2:
            assert d2 % block_size == 0, (
                f"Weight dimension {d2} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0, d1 * d2 // block_size, block_size)).max(axis=2)
        elif quant_axis == 1:
            assert d1 % block_size == 0, (
                f"Weight dimension {d1} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0 * d1 // block_size, block_size, d2)).max(axis=1)
        else:
            raise ValueError(f"Unsupported weight axis: {quant_axis}")
    else:
        d0, d1 = weight.shape
        if quant_axis == 1:
            assert d1 % block_size == 0, (
                f"Weight dimension {d1} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(d0 * d1 // block_size, block_size)).max(axis=1)
        elif quant_axis == 0:
            assert d0 % block_size == 0, (
                f"Weight dimension {d0} must be divisible by block size {block_size}"
            )
            amax = np.abs(weight.reshape(block_size, d0 * d1 // block_size)).max(axis=0)
        else:
            raise ValueError(f"Unsupported weight axis: {quant_axis}")
    return amax


def compute_e8m0(
    amax: np.ndarray, weight_shape: tuple[int, ...], quant_axis: int, block_size: int
) -> np.ndarray:
    """Computes the e8m0 value for the weight tensor.

    Args:
        amax: The amax of the weight tensor.
        weight_shape: The shape of the weight tensor.
        quant_axis: The axis to compute the e8m0 value.
        block_size: The block size.

    Returns:
        The e8m0 value for the weight tensor.
    """
    rank = len(weight_shape)
    if quant_axis == -1:
        quant_axis = rank - 1
    q_max = 448  # Largest value of FP8
    e8m0_bias = 127
    amax = amax / q_max
    min_value = -127.0
    e8m0_unbiased_exp = np.ceil(np.maximum(np.log2(amax), min_value))
    e8m0 = e8m0_unbiased_exp + e8m0_bias
    if not np.all((e8m0 >= 0) & (e8m0 <= 255)):
        raise ValueError(f"e8m0 values out of range: min={np.min(e8m0)}, max={np.max(e8m0)}")
    if rank == 2:
        d0, d1 = weight_shape
        if quant_axis == 1:
            e8m0 = e8m0.reshape(d0, d1 // block_size)
        elif quant_axis == 0:
            e8m0 = e8m0.reshape(d0 // block_size, d1)
        else:
            raise ValueError(f"Unsupported axis: {quant_axis}")
    elif rank == 3:
        d0, d1, d2 = weight_shape
        if quant_axis == 2:
            e8m0 = e8m0.reshape(d0, d1, d2 // block_size)
        elif quant_axis == 1:
            e8m0 = e8m0.reshape(d0, d1 // block_size, d2)
        else:
            raise ValueError(f"Unsupported axis: {quant_axis}")
    else:
        raise ValueError(f"Unsupported weight shape: {weight_shape}")
    return e8m0
