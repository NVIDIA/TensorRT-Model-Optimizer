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


"""Hadamard transformation utilities for rotation-based quantization.

This module provides functions for applying Hadamard transformations to tensors,
with optimized implementations for both CPU and CUDA devices.
"""

# Adapted from https://github.com/spcl/QuaRot/blob/main/quarot/functional/hadamard.py

import math
import os
from functools import cache

import scipy.linalg as linalg

try:
    import fast_hadamard_transform
except ImportError:
    print("fast_hadamard_transform not found. It's currently not needed for pre-fuse rotation.")
    fast_hadamard_transform = None
import torch

HADK_WEIGHTS = os.path.join(os.path.dirname(__file__), "hadK.pth")


@cache
def get_had_k(n, transpose=False):
    """Get pre-computed Hadamard matrix (had_k) and its dimension (k) for a given size.

    This function loads pre-computed Hadamard matrices for specific dimensions that
    are commonly used in LLaMA and other transformer models. The dimension must be
    divisible by one of the supported k values (172, 156, 140, 108, 60, 52, 40, 36,
    28, 20, 12) or be a power of 2.

    Args:
        n (int): The target dimension size. Must be divisible by a supported k value
            or be a power of 2.
        transpose (bool, optional): Whether to return the transposed hadamard matrix.
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            - had_k (torch.Tensor or None): Pre-computed Hadamard matrix of shape (k, k),
              or None if k=1 (pure power-of-2 case).
            - k (int): The base dimension of the Hadamard matrix.

    Raises:
        AssertionError: If n is not properly divisible by a supported k value with
            a power-of-2 quotient, or if n is not a power of 2 when no k divides it.
    """
    had_k, k = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)
        k = 172
        had_k = torch.load(HADK_WEIGHTS)[172]
        had_k = had_k.T if transpose else had_k
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)
        k = 156
        had_k = torch.load(HADK_WEIGHTS)[156]
        had_k = had_k.T if transpose else had_k
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)
        k = 140
        had_k = torch.load(HADK_WEIGHTS)[140]
        had_k = had_k.T if transpose else had_k
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)
        k = 108
        had_k = torch.load(HADK_WEIGHTS)[108]
        had_k = had_k.T if transpose else had_k
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)
        k = 60
        had_k = torch.load(HADK_WEIGHTS)[60]
        had_k = had_k.T if transpose else had_k
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)
        k = 52
        had_k = torch.load(HADK_WEIGHTS)[52]
        had_k = had_k.T if transpose else had_k
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        k = 36
        had_k = torch.load(HADK_WEIGHTS)[36]
        had_k = had_k.T if transpose else had_k
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        k = 28
        had_k = torch.load(HADK_WEIGHTS)[28]
        had_k = had_k.T if transpose else had_k
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        k = 40
        had_k = torch.load(HADK_WEIGHTS)[40]
        had_k = had_k.T if transpose else had_k
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        k = 20
        had_k = torch.load(HADK_WEIGHTS)[20]
        had_k = had_k.T if transpose else had_k
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        k = 12
        had_k = torch.load(HADK_WEIGHTS)[12]
        had_k = had_k.T if transpose else had_k
    else:
        assert is_pow2(n)
        k = 1
    if had_k is not None:
        assert had_k.shape[0] == k

    return had_k, k


def matmul_had_u(x, transpose=False):
    """Apply Hadamard transformation (matrix U) to input tensor x on CPU.

    This function performs an optimized Hadamard transformation using a divide-and-conquer
    approach, optionally using pre-computed Hadamard matrices for non-power-of-2 dimensions.

    Args:
        x (torch.Tensor): Input tensor of any shape. The Hadamard transformation is
            applied along the last dimension.
        transpose (bool, optional): Whether to use the transposed Hadamard matrix.
            Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input x, scaled by
            1/sqrt(n) where n is the size of the last dimension.
    """
    n = x.shape[-1]
    had_k, k = get_had_k(n, transpose)
    input = x.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > k:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if k > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     had_k.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = had_k.view(1, k, k).to(input) @ input

    return input.view(x.shape) / torch.tensor(n).sqrt()


def matmul_had_ut(x):
    """Apply transposed Hadamard transformation to input tensor x on CPU.

    This is a convenience wrapper around matmul_had_u with transpose=True.

    Args:
        x (torch.Tensor): Input tensor of any shape. The transposed Hadamard
            transformation is applied along the last dimension.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input x.
    """
    return matmul_had_u(x, transpose=True)


# See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
def random_hadamard_matrix(size, device):
    """Generate a random Hadamard matrix using diagonal randomization.

    Creates a randomized Hadamard matrix by applying a random diagonal matrix
    (with entries of +1 or -1) followed by a Hadamard transformation.
    See https://cornell-relaxml.github.io/quip-sharp/ for more details.

    Args:
        size (int): The dimension of the square Hadamard matrix to generate.
        device (torch.device): The device on which to create the matrix.

    Returns:
        torch.Tensor: A random Hadamard matrix of shape (size, size) in float64.
    """
    q = torch.randint(0, 2, (size,)).to(torch.float64)
    q = q * 2 - 1
    q = torch.diag(q)
    return matmul_had_u(q).to(device)


def random_base_hadamard_matrix(size, device):
    """Generate a random base Hadamard matrix using Sylvester's construction.

    This function creates a Hadamard matrix using scipy's implementation of
    Sylvester's construction method, which is only defined for power-of-2 sizes.

    Args:
        size (int): The dimension of the square Hadamard matrix. Must be a power of 2.
        device (torch.device): The device on which to create the matrix.

    Returns:
        torch.Tensor: A normalized Hadamard matrix of shape (size, size) in float64,
            scaled by 1/sqrt(size).

    Raises:
        AssertionError: If size is not a power of 2.
    """
    assert is_pow2(size), "Size must be a power of 2 for hadamard matrix!"
    return torch.tensor(linalg.hadamard(size), dtype=torch.float64).to(device) / torch.sqrt(size)


def matmul_had_u_cuda(x, had_k, transpose=False):
    """Apply Hadamard transformation to input tensor x on CUDA using fast implementation.

    This function uses the fast_hadamard_transform library for efficient CUDA-based
    Hadamard transformations. It supports both power-of-2 dimensions (had_k=None)
    and non-power-of-2 dimensions using pre-computed Hadamard matrices.

    Args:
        x (torch.Tensor): Input tensor of any shape. The Hadamard transformation is
            applied along the last dimension.
        had_k (torch.Tensor or None): Pre-computed Hadamard matrix of shape (k, k).
            If None, assumes the last dimension is a power of 2.
        transpose (bool, optional): Placeholder parameter for API compatibility with
            matmul_diag. Currently not used in the implementation. Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input x, scaled by
            1/sqrt(n) where n is the size of the last dimension.

    Raises:
        AssertionError: If had_k is None and the last dimension of x is not a power of 2.
    """
    n = x.shape[-1]
    if had_k is None:
        assert is_pow2(n), "Input dimension must be a power of 2!"
        return fast_hadamard_transform.hadamard_transform(
            x.contiguous(),
            1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
        )
    # if transpose:
    #     had_k = had_k.T.contiguous()
    k = had_k.shape[0]
    input = x.view(*x.shape[:-1], k, n // k)
    input = fast_hadamard_transform.hadamard_transform(
        input.contiguous(),
        1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
    )
    # input = had_k.to(input.device).to(input.dtype) @ input
    input = had_k @ input
    return input.contiguous().view(x.shape)


def is_pow2(n):
    """Check if a number is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)
