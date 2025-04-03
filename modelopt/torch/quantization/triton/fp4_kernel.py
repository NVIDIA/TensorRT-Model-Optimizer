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
    global_scale,
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
        global_scale (float): Global scaling factor
        BLOCK_SIZE (tl.constexpr): Size of each FP4 quantization block
        TILE_SIZE (tl.constexpr): Size of the processing block
        NUM_FP4_BLOCKS (tl.constexpr): Number of FP4 blocks within TILE_SIZE
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate offsets
    offs_m = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load input data
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    # Reshape for block processing
    x_reshaped = tl.reshape(x, (TILE_SIZE, NUM_FP4_BLOCKS, BLOCK_SIZE))

    # Calculate max values for each FP4 block
    block_max = tl.max(tl.abs(x_reshaped), axis=2, keep_dims=True)
    # global_scale = global_amax / (448 * 6)
    block_max_quant = (
        tl.clamp((block_max / (6.0 * global_scale)), -448.0, 448.0).to(tl.float8e4nv).to(tl.float32)
        * global_scale
    )

    # Broadcast max values
    block_max_quant_broadcast = tl.broadcast_to(
        block_max_quant, (TILE_SIZE, NUM_FP4_BLOCKS, BLOCK_SIZE)
    )

    x_scaled = x_reshaped / block_max_quant_broadcast

    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    abs_scaled = tl.abs(x_scaled)
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
    sign = tl.where(x_scaled >= 0, 1.0, -1.0)

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = x_rescaled * sign

    # Reshape back and store
    x_rescaled = tl.reshape(x_rescaled, (TILE_SIZE, TILE_SIZE))
    tl.store(y_ptr + offs, x_rescaled, mask=mask)


def fp4_fake_quant_block(
    x: torch.Tensor,
    global_amax: float,
    block_size: int = 16,
    tile_size: int = 128,
) -> torch.Tensor:
    """Applies FP4 fake quantization on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (M, N)
        global_scale (float): Global scaling factor
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

    grid = lambda meta: (triton.cdiv(M, meta["TILE_SIZE"]), triton.cdiv(N, meta["TILE_SIZE"]))
    global_scale = (global_amax / 6.0) / 448.0
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
