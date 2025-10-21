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
def get_hadK(n, transpose=False):
    """Get pre-computed Hadamard matrix (hadK) and its dimension (K) for a given size.

    This function loads pre-computed Hadamard matrices for specific dimensions that
    are commonly used in LLaMA and other transformer models. The dimension must be
    divisible by one of the supported K values (172, 156, 140, 108, 60, 52, 40, 36,
    28, 20, 12) or be a power of 2.

    Args:
        n (int): The target dimension size. Must be divisible by a supported K value
            or be a power of 2.
        transpose (bool, optional): Whether to return the transposed hadamard matrix.
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            - hadK (torch.Tensor or None): Pre-computed Hadamard matrix of shape (K, K),
              or None if K=1 (pure power-of-2 case).
            - K (int): The base dimension of the Hadamard matrix.

    Raises:
        AssertionError: If n is not properly divisible by a supported K value with
            a power-of-2 quotient, or if n is not a power of 2 when no K divides it.
    """
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)
        K = 172
        hadK = torch.load(HADK_WEIGHTS)[172]
        hadK = hadK.T if transpose else hadK
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)
        K = 156
        hadK = torch.load(HADK_WEIGHTS)[156]
        hadK = hadK.T if transpose else hadK
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)
        K = 140
        hadK = torch.load(HADK_WEIGHTS)[140]
        hadK = hadK.T if transpose else hadK
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)
        K = 108
        hadK = torch.load(HADK_WEIGHTS)[108]
        hadK = hadK.T if transpose else hadK
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)
        K = 60
        hadK = torch.load(HADK_WEIGHTS)[60]
        hadK = hadK.T if transpose else hadK
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)
        K = 52
        hadK = torch.load(HADK_WEIGHTS)[52]
        hadK = hadK.T if transpose else hadK
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = torch.load(HADK_WEIGHTS)[36]
        hadK = hadK.T if transpose else hadK
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        K = 28
        hadK = torch.load(HADK_WEIGHTS)[28]
        hadK = hadK.T if transpose else hadK
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = torch.load(HADK_WEIGHTS)[40]
        hadK = hadK.T if transpose else hadK
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = torch.load(HADK_WEIGHTS)[20]
        hadK = hadK.T if transpose else hadK
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = torch.load(HADK_WEIGHTS)[12]
        hadK = hadK.T if transpose else hadK
    else:
        assert is_pow2(n)
        K = 1
    if hadK is not None:
        assert hadK.shape[0] == K

    return hadK, K


def matmul_hadU(X, transpose=False):
    """Apply Hadamard transformation (matrix U) to input tensor X on CPU.

    This function performs an optimized Hadamard transformation using a divide-and-conquer
    approach, optionally using pre-computed Hadamard matrices for non-power-of-2 dimensions.

    Args:
        X (torch.Tensor): Input tensor of any shape. The Hadamard transformation is
            applied along the last dimension.
        transpose (bool, optional): Whether to use the transposed Hadamard matrix.
            Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input X, scaled by
            1/sqrt(n) where n is the size of the last dimension.
    """
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    """Apply transposed Hadamard transformation to input tensor X on CPU.

    This is a convenience wrapper around matmul_hadU with transpose=True.

    Args:
        X (torch.Tensor): Input tensor of any shape. The transposed Hadamard
            transformation is applied along the last dimension.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input X.
    """
    return matmul_hadU(X, transpose=True)


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
    Q = torch.randint(0, 2, (size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


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


# def matmul_hadU_cuda(X, hadK, K):
#     n = X.shape[-1]
#     if K == 1:
#         return fast_hadamard_transform.hadamard_transform(
#             X.contiguous(), 1.0 / torch.tensor(n).sqrt()
#         )
#     # if transpose:
#     #     hadK = hadK.T.contiguous()
#     input = X.view(-1, K, n // K)
#     input = fast_hadamard_transform.hadamard_transform(
#         input.contiguous(), 1.0 / torch.tensor(n).sqrt()
#     )
#     input = hadK.to(input.device).to(input.dtype) @ input
#     return input.reshape(X.shape)


def matmul_hadU_cuda(X, hadK, transpose=False):
    """Apply Hadamard transformation to input tensor X on CUDA using fast implementation.

    This function uses the fast_hadamard_transform library for efficient CUDA-based
    Hadamard transformations. It supports both power-of-2 dimensions (hadK=None)
    and non-power-of-2 dimensions using pre-computed Hadamard matrices.

    Args:
        X (torch.Tensor): Input tensor of any shape. The Hadamard transformation is
            applied along the last dimension.
        hadK (torch.Tensor or None): Pre-computed Hadamard matrix of shape (K, K).
            If None, assumes the last dimension is a power of 2.
        transpose (bool, optional): Placeholder parameter for API compatibility with
            matmul_diag. Currently not used in the implementation. Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input X, scaled by
            1/sqrt(n) where n is the size of the last dimension.

    Raises:
        AssertionError: If hadK is None and the last dimension of X is not a power of 2.
    """
    n = X.shape[-1]
    if hadK is None:
        assert is_pow2(n), "Input dimension must be a power of 2!"
        return fast_hadamard_transform.hadamard_transform(
            X.contiguous(),
            1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
        )
    # if transpose:
    #     hadK = hadK.T.contiguous()
    K = hadK.shape[0]
    input = X.view(*X.shape[:-1], K, n // K)
    input = fast_hadamard_transform.hadamard_transform(
        input.contiguous(),
        1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
    )
    # input = hadK.to(input.device).to(input.dtype) @ input
    input = hadK @ input
    return input.contiguous().view(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    """Apply transposed Hadamard transformation to input tensor X on CUDA.

    This is a convenience wrapper around matmul_hadU_cuda with transpose=True.
    Note: The transpose parameter currently has no effect in the implementation.

    Args:
        X (torch.Tensor): Input tensor of any shape. The Hadamard transformation is
            applied along the last dimension.
        hadK (torch.Tensor or None): Pre-computed Hadamard matrix of shape (K, K).
        K (int): The base dimension (currently unused, kept for API compatibility).

    Returns:
        torch.Tensor: Transformed tensor with the same shape as input X.
    """
    return matmul_hadU_cuda(X, hadK, K, transpose=True)


def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    """Apply Hadamard transformation to the weights of a Linear layer in-place.

    This function modifies the weights of a torch.nn.Linear module by applying
    a Hadamard transformation. The transformation can be applied to either the
    input features (default) or output features dimension. For non-standard
    dimensions, it can apply chunked transformations.

    Args:
        module (torch.nn.Linear): The linear layer whose weights will be transformed.
        had_dim (int, optional): The dimension size for chunked Hadamard transformation.
            If -1 (default), applies full Hadamard transformation to the entire dimension.
            If positive, must be a power of 2, and the transformation is applied to
            chunks of this size.
        output (bool, optional): If True, applies transformation to the output dimension
            (rows of weight matrix). If False, applies to input dimension (columns).
            Defaults to False.

    Raises:
        AssertionError: If module is not a torch.nn.Linear instance, or if had_dim
            is not -1 or a power of 2.
        NotImplementedError: If had_dim is positive and output is False (not yet implemented).

    Note:
        This operation modifies the module's weight data in-place and temporarily moves
        the weights to CUDA for computation before moving them back to the original device.
    """
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    # Apply Hadamard to the last had_dim chunks of the weights
    elif output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = (
            fast_hadamard_transform.hadamard_transform(
                W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim),
                scale=1 / math.sqrt(had_dim),
            )
            .reshape(transposed_shape)
            .t()
        )
    else:
        raise NotImplementedError("Not implemented (or tested) yet!")
        n = W_.shape[1]
        W_ = hadamard_transform(
            W_.reshape(-1, n // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
        ).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    """Check if a number is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)
