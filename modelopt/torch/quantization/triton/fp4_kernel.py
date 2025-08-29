#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NVFP4 Fake Quantization Triton Implementation.

This module provides high-performance GPU implementations of NVFP4 fake quantization
operations using Triton kernels.
"""

import torch
import triton
import triton.language as tl

__all__ = ["fp4_fake_quant_block"]


@triton.jit
def fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
):
    """Applies FP4 fake quantization on input data using per-block scaling factors.

    Args:
        x_ptr (tl.pointer): Pointer to the input tensor (BF16/FP32)
        y_ptr (tl.pointer): Pointer to the output buffer
        M (int): Number of rows in the matrix
        N (int): Number of columns in the matrix
        global_scale_ptr (tl.pointer): Pointer to the global scaling factor tensor
        BLOCK_SIZE (tl.constexpr): Size of each FP4 quantization block
        TILE_SIZE (tl.constexpr): Size of the processing block
        NUM_FP4_BLOCKS (tl.constexpr): Number of FP4 blocks within TILE_SIZE
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Load global scale from tensor
    global_scale = tl.load(global_scale_ptr).to(tl.float32)

    # Calculate offsets
    offs_m = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load input data
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    # Reshape for block processing
    x_reshaped = tl.reshape(x, (TILE_SIZE, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(x_reshaped)

    # Calculate max values for each FP4 block
    block_max = tl.max(x_abs, axis=2, keep_dims=True)
    # global_scale = global_amax / (448 * 6)
    block_max_quant = (
        tl.minimum((block_max / (6.0 * global_scale)), 448.0).to(tl.float8e4nv).to(tl.float32)
        * global_scale
    )

    # Broadcast max values
    block_max_quant_broadcast = tl.broadcast_to(
        block_max_quant, (TILE_SIZE, NUM_FP4_BLOCKS, BLOCK_SIZE)
    )
    # Set scale to 1 if block amax is 0
    block_max_quant_broadcast = tl.where(
        block_max_quant_broadcast < 1e-5, 1.0, block_max_quant_broadcast
    )
    abs_scaled = x_abs / block_max_quant_broadcast

    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )

    # Apply signs and rescale
    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(x_reshaped >= 0, x_rescaled, -x_rescaled)

    # Reshape back and store
    x_rescaled = tl.reshape(x_rescaled, (TILE_SIZE, TILE_SIZE))
    tl.store(y_ptr + offs, x_rescaled, mask=mask)


def fp4_fake_quant_block(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    tile_size: int = 128,
) -> torch.Tensor:
    """Applies FP4 fake quantization on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (M, N)
        global_amax (torch.Tensor): Global max value of the input tensor
            This needs to be a tensor to be cuda-graph compatible
        block_size (int): Size of FP4 quantization blocks
        tile_size (int): Size of processing blocks

    Returns:
        torch.Tensor: Quantized tensor of the same shape as input
    """
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.reshape(-1, x_shape[-1]).contiguous()

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    grid = lambda meta: (
        triton.cdiv(M, meta["TILE_SIZE"]),
        triton.cdiv(N, meta["TILE_SIZE"]),
    )
    global_scale = global_amax.float() / (6.0 * 448.0)
    num_fp4_blocks = tile_size // block_size
    fp4_fake_quant_kernel[grid](
        x,
        y,
        M,
        N,
        global_scale,
        TILE_SIZE=tile_size,
        BLOCK_SIZE=block_size,
        NUM_FP4_BLOCKS=num_fp4_blocks,
    )
    y = y.reshape(x_shape).contiguous().to(dtype=x_dtype)
    return y


@triton.jit
def fp4_dequantize_kernel(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """Dequantizes FP4 packed data using per-block scaling factors.

    Args:
        packed_ptr (tl.pointer): Pointer to packed uint8 tensor (M x N//2)
        scale_ptr (tl.pointer): Pointer to per-block scale tensor (M x N//BLOCK_SIZE)
        output_ptr (tl.pointer): Pointer to output tensor (M x N)
        global_scale_ptr (tl.pointer): Pointer to global scale tensor
        N (int): Number of columns in unpacked tensor
        BLOCK_SIZE (tl.constexpr): Size of each FP4 quantization block
        TILE_SIZE (tl.constexpr): Size of the processing tile (in packed elements)
    """
    # Get program ID for processing packed elements
    pid = tl.program_id(0)

    # Calculate packed element offsets (each packed element contains 2 FP4 values)
    packed_start = pid * TILE_SIZE
    packed_offs = packed_start + tl.arange(0, TILE_SIZE)

    # Calculate 2D coordinates for packed data
    packed_row_idx = packed_offs // (N // 2)
    packed_col_idx = packed_offs % (N // 2)

    # Create mask for packed data bounds checking
    packed_mask = packed_col_idx < (N // 2)

    # Load global scale
    global_scale = tl.load(global_scale_ptr)

    # Load packed data
    packed_data = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    # Unpack packed FP4 values (uint8) to float16x2
    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[packed_data],
        dtype=tl.uint32,
        is_pure=True,
        pack=4,
    )
    val_low = (
        (x_f16x2_packed & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )
    val_high = (
        (x_f16x2_packed >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )

    # Calculate output positions for both values
    out_col_low = packed_col_idx * 2
    out_col_high = packed_col_idx * 2 + 1
    out_offs_low = packed_row_idx * N + out_col_low
    out_offs_high = packed_row_idx * N + out_col_high

    # Calculate block indices for scaling
    block_col_low = out_col_low // BLOCK_SIZE
    block_col_high = out_col_high // BLOCK_SIZE
    scale_offs_low = packed_row_idx * (N // BLOCK_SIZE) + block_col_low
    scale_offs_high = packed_row_idx * (N // BLOCK_SIZE) + block_col_high

    # Load scaling factors
    scale_low = tl.load(scale_ptr + scale_offs_low, mask=packed_mask & (out_col_low < N), other=1.0)
    scale_high = tl.load(
        scale_ptr + scale_offs_high, mask=packed_mask & (out_col_high < N), other=1.0
    )

    # Apply scaling
    result_low = val_low * scale_low.to(tl.float32) * global_scale
    result_high = val_high * scale_high.to(tl.float32) * global_scale

    # Store results
    out_mask_low = packed_mask & (out_col_low < N)
    out_mask_high = packed_mask & (out_col_high < N)

    tl.store(output_ptr + out_offs_low, result_low, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_high, mask=out_mask_high)


def fp4_dequantize(
    packed_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    block_size: int = 16,
    tile_size: int = 128,
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    """Dequantizes FP4 packed tensor using per-block scaling factors.

    Args:
        packed_tensor (torch.Tensor): Packed uint8 tensor of shape (M, N//2)
        scale_tensor (torch.Tensor): Per-block scale tensor of shape (M, N//block_size)
        global_scale (torch.Tensor): Global scaling factor tensor
        block_size (int): Size of FP4 quantization blocks
        tile_size (int): Size of processing tiles

    Returns:
        torch.Tensor: Dequantized tensor of shape (M, N)
    """
    packed_N = packed_tensor.shape[-1]
    N = packed_N * 2
    # Create output tensor with proper shape handling
    output_shape = list(packed_tensor.shape)
    output_shape[-1] = N
    output = torch.empty(output_shape, dtype=dtype, device=packed_tensor.device)

    # Calculate total number of elements and grid size
    grid = lambda meta: (triton.cdiv(packed_tensor.numel(), meta["TILE_SIZE"]),)

    fp4_dequantize_kernel[grid](
        packed_tensor,
        scale_tensor,
        global_scale,
        output,
        N,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
    )

    return output
