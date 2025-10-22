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


"""Rotation utility functions for preprocessing.

This module provides utilities for applying rotation transformations to model weights
and biases, including support for layer normalization fusion, orthogonal matrix generation,
and distributed tensor (DTensor) handling.
"""

import fnmatch
import gc
import re
import typing
from warnings import warn

import torch

from .hadamard_utils import get_base_hadamard_matrix, matmul_had_u_cuda, random_hadamard_matrix


def _iter_modules_by_pattern(
    model: torch.nn.Module, pattern: str, regex: bool = False
) -> list[tuple[str, torch.nn.Module]]:
    """Find all modules matching the given pattern.

    Supports OR logic with pipe separator: '*q_proj|*k_proj' matches either pattern.
    Can use either glob patterns (fnmatch) or regular expressions.

    Args:
        model (torch.nn.Module): The model to search for matching modules.
        pattern (str): Pattern string to match. Use '|' to separate multiple patterns
            for OR logic. For regex=False, uses glob patterns (e.g., '*q_proj').
            For regex=True, uses regular expressions.
        regex (bool, optional): If True, treat pattern as a regular expression.
            If False, treat as glob pattern. Defaults to False.

    Returns:
        list[tuple[str, torch.nn.Module]]: List of (module_name, module) tuples for
            all modules matching the pattern(s).
    """
    results: list[tuple[str, torch.nn.Module]] = []
    patterns = pattern.split("|")

    for name, module in model.named_modules():
        for p in patterns:
            if regex:
                if re.search(p, name):
                    results.append((name, module))
                    break
            elif fnmatch.fnmatch(name, p):
                results.append((name, module))
                break
    return results


# Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/utils/fuse_norm_utils.py
def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """Fuse LayerNorm parameters into adjacent Linear layers.

    This function absorbs the LayerNorm's scale (weight) and bias into the subsequent
    Linear layer(s) by modifying their weights and biases. After fusion, the LayerNorm
    is reset to identity (weight=1, bias=0).

    Args:
        layernorm (torch.nn.Module): The LayerNorm module whose parameters will be fused.
            Should have 'weight' attribute and optionally 'bias' attribute.
        linear_layers (typing.Iterable[torch.nn.Linear]): Iterable of Linear layers to
            fuse the LayerNorm into. Each layer's weight (and bias if present) will be
            modified in-place.

    Note:
        This operation modifies the modules in-place. After fusion, the layernorm
        becomes an identity operation.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        w_ = linear.weight.data.double()
        linear.weight.data = (w_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            print(f"Fusing bias for {linear.weight.shape} and {layernorm.weight.shape}")
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(w_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

    torch.nn.init.ones_(layernorm.weight)
    if hasattr(layernorm, "bias"):
        torch.nn.init.zeros_(layernorm.bias)


@torch.no_grad()
def fuse_layernorms(model: torch.nn.Module, norm_fuse_config: dict) -> None:
    """Fuse LayerNorm scale/bias into neighboring Linear layers as configured.

    Args:
        model: Model to apply fusion to
        norm_fuse_config: Dict with keys:
            - decoder_layer_fuse: List[tuple[str, list[str]]] - decoder layer norm fusions
            - lm_head_fuse: List[tuple[str, str]] - lm head norm fusions
    """
    for name, module in model.named_modules():
        if re.search("layers\\.[0-9]+$", name) is not None:
            for layer_norm, linear_layers in norm_fuse_config.get("decoder_layer_fuse", []):
                print(
                    f"Fusing layer norm {layer_norm} and linear layers {linear_layers} for {name}"
                )
                submodules = {name for name, _ in module.named_modules()}

                if layer_norm in submodules and all(
                    linear in submodules for linear in linear_layers
                ):
                    fuse_ln_linear(
                        module.get_submodule(layer_norm),
                        [module.get_submodule(linear) for linear in linear_layers],
                    )
                else:
                    print(f"{name} submodules: {submodules}")
                    warn(
                        f"{name} has no submodule {layer_norm} or {linear_layers}, skipping fusion. "
                        "This can happen if the model has mixed types of layers, e.g. nemotron H architecture."
                    )
    for layer_norm, linear in norm_fuse_config.get("lm_head_fuse", []):
        print(f"Fusing layer norm {layer_norm} and linear {linear} for model")
        fuse_ln_linear(model.get_submodule(layer_norm), [model.get_submodule(linear)])


def random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal matrix of the specified size on device."""
    return torch.nn.init.orthogonal_(torch.empty(size, size, device=device))


def _normalize_mode(mode: str) -> str:
    """Normalize rotation mode string to canonical form.

    Args:
        mode (str): Mode string to normalize (e.g., "hadamard", "Hadamard", "HADAMARD").

    Returns:
        str: Normalized mode string ("hadamard" if the input contains "hadamard",
            otherwise "random").
    """
    m = mode.lower().strip().replace("_", " ")
    if "hadamard" in m:
        return "hadamard"
    return "random"


def get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
    """Get an orthogonal matrix of the specified size and mode on device.

    Args:
        size (int): The dimension of the square orthogonal matrix to generate.
        mode (str): The type of orthogonal matrix to generate. Supported modes:
            - "orthogonal" or "random": Random orthogonal matrix via QR decomposition.
            - "hadamard": Randomized Hadamard matrix with diagonal randomization.
            - "base_hadamard": Base Hadamard matrix using Sylvester's construction.
            - "fast_hadamard": Fast Hadamard transform (returns None, handled separately).
        device (torch.device): The device on which to create the matrix.

    Returns:
        torch.Tensor: An orthogonal matrix of shape (size, size).

    Raises:
        ValueError: If mode is not one of the supported modes.
    """
    mode = _normalize_mode(mode)
    if mode == "orthogonal":
        return random_orthogonal_matrix(size, device)
    if mode == "hadamard":
        return random_hadamard_matrix(size, device)
    if mode == "base_hadamard":
        return get_base_hadamard_matrix(size, device)
    raise ValueError(f"Unknown rotation matrix mode: {mode}")


def extract_layer_index(module_name: str) -> int | None:
    """Extract layer index from module name.

    Args:
        module_name: Module name like 'model.layers.12.self_attn.o_proj'

    Returns:
        Layer index as integer, or None if not found
    """
    match = re.search(r"layers\.(\d+)", module_name)
    return int(match.group(1)) if match else None


def diag_matmul(
    m1: torch.Tensor, m2: torch.Tensor, transpose=True, block_size=None
) -> torch.Tensor:
    """Perform matrix multiplication m1 @ m2 with optional block-diagonal support.

    This function handles two cases:
    1. Standard matrix multiplication if dimensions match
    2. Block-diagonal multiplication where m1 is replicated as a block-diagonal matrix

    Args:
        m1 (torch.Tensor): Left matrix. Can be a single block of a block-diagonal matrix.
        m2 (torch.Tensor): Right matrix to multiply.
        transpose (bool, optional): If True, transpose m1 before multiplication.
            Defaults to True.
        block_size (int, optional): Block size for block-diagonal operations. Currently
            unused. Defaults to None.

    Returns:
        torch.Tensor: Result of m1 @ m2 (or m1.T @ m2 if transpose=True). If m1 is
            smaller than required, it's treated as a repeated block-diagonal matrix.

    Note:
        For block-diagonal case, m2.shape[-2] must be divisible by m1.shape[1].
    """
    if transpose:
        m1 = m1.t()
    if m1.shape[-1] == m2.shape[-2]:
        return torch.matmul(m1, m2)
    m2_shape = m2.shape
    size = m1.shape[1]
    m2 = m2.view(*m2.shape[:-2], m2.shape[-2] // size, size, m2.shape[-1])
    # print(m1.shape, m2.shape)
    output = m1 @ m2  # torch.matmul(m1, m2)
    return output.contiguous().view(*m2_shape)


def matmul_diag(
    m1: torch.Tensor, m2: torch.Tensor, transpose=False, block_size=None
) -> torch.Tensor:
    """Perform matrix multiplication m1 @ m2 with optional block-diagonal support.

    This function handles two cases:
    1. Standard matrix multiplication if dimensions match
    2. Block-diagonal multiplication where m2 is replicated as a block-diagonal matrix

    Args:
        m1 (torch.Tensor): Left matrix to multiply.
        m2 (torch.Tensor): Right matrix. Can be a single block of a block-diagonal matrix.
        transpose (bool, optional): If True, transpose m2 before multiplication.
            Defaults to False.
        block_size (int, optional): Block size for block-diagonal operations. Currently
            unused. Defaults to None.

    Returns:
        torch.Tensor: Result of m1 @ m2 (or m1 @ m2.T if transpose=True). If m2 is
            smaller than required, it's treated as a repeated block-diagonal matrix.

    Note:
        For block-diagonal case, m1.shape[-1] must be divisible by m2.shape[0].
    """
    if transpose:
        # print("transposed")
        m2 = m2.t()
    if m1.shape[-1] == m2.shape[-2]:
        return torch.matmul(m1, m2)
    m1_shape = m1.shape
    size = m2.shape[0]
    # assert block_size == size
    m1 = m1.view(*m1.shape[:-1], m1.shape[-1] // size, size)
    output = m1 @ m2
    return output.contiguous().view(*m1_shape)


def matmul_diag_fast_hadamard(m1, had_k, transpose=False, block_size=None):
    """Apply fast Hadamard transform: m1 @ H where H is a Hadamard matrix.

    Uses the fast_hadamard_transform library for efficient computation. The Hadamard
    matrix is constructed via Sylvester's method, which produces a symmetric matrix.

    Args:
        m1 (torch.Tensor): Input matrix to multiply with the Hadamard matrix.
        had_k (torch.Tensor or None): Pre-computed Hadamard matrix for non-power-of-2
            dimensions. If None, assumes the last dimension is a power of 2.
        transpose (bool, optional): Placeholder for API compatibility. Not used since
            Hadamard matrices are symmetric. Defaults to False.
        block_size (int, optional): If provided, applies Hadamard transform in blocks
            of this size. Defaults to None.

    Returns:
        torch.Tensor: Result of m1 @ H where H is the Hadamard matrix.

    Note:
        If m1.shape[-1] is not a power of 2, had_k must be provided.
    """
    if block_size is not None:
        shape = m1.shape
        m1 = m1.view(*shape[:-1], shape[-1] // block_size, block_size)

        return matmul_had_u_cuda(m1, None).view(*shape)
    return matmul_had_u_cuda(m1, had_k)


def with_dtensor_support(rotation_fn):
    """Decorator to add DTensor support to rotation functions.

    This wrapper handles the gather-compute-redistribute pattern for DTensors:
    - Uses .full_tensor() to gather the complete tensor
    - Runs rotation only on rank 0
    - Broadcasts result to other ranks
    - Redistributes back to original DTensor placements

    Args:
        rotation_fn: The rotation function to wrap (e.g., _rotate_weight_impl, _rotate_bias_impl)

    Returns:
        Wrapped function with DTensor support
    """

    def wrapper(tensor, *args, **kwargs):
        if not is_dtensor(tensor):
            # Regular tensor path - call the function directly
            return rotation_fn(tensor, *args, **kwargs)

        # DTensor path
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            from torch.distributed.tensor import DTensor

        # Save original DTensor metadata
        original_placements = tensor.placements
        original_device_mesh = tensor.device_mesh

        # Gather the full tensor using .full_tensor() - simpler than manual redistribute
        full_tensor = tensor.full_tensor()

        # Only run rotation on rank 0 to save compute
        current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if current_rank == 0:
            # Run rotation on main GPU
            rotated_tensor = rotation_fn(full_tensor, *args, **kwargs)
        else:
            # Other ranks create placeholder with same shape/dtype/device
            rotated_tensor = torch.empty_like(full_tensor)

        # Broadcast result from rank 0 to all other ranks
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.broadcast(rotated_tensor, src=0)

        # Convert back to DTensor and redistribute to original placements
        dtensor_result = DTensor.from_local(
            rotated_tensor,
            device_mesh=original_device_mesh,
            placements=original_placements,
            run_check=False,
        )

        return dtensor_result

    return wrapper


def _rotate_weight_impl(
    weight, input_spin, output_spin, fast_hadamard=False, block_size=None, use_float64=True
):
    """Core implementation of weight rotation (non-DTensor version).

    Applies rotation matrices to weight tensor: output_spin @ weight @ input_spin.
    Automatically handles device placement and dtype conversions.

    Args:
        weight (torch.Tensor): Weight tensor to rotate, shape (out_features, in_features).
        input_spin (torch.Tensor or None): Input rotation matrix. If None, no input
            rotation is applied (unless fast_hadamard=True with block_size).
        output_spin (torch.Tensor or None): Output rotation matrix. If None, no output
            rotation is applied.
        fast_hadamard (bool, optional): If True, use fast Hadamard transform instead
            of explicit matrix multiplication. Defaults to False.
        block_size (int, optional): Block size for chunked Hadamard transform.
            Defaults to None.
        use_float64 (bool, optional): If True, perform computations in float64 for
            higher precision. Defaults to False (uses float32).

    Returns:
        torch.Tensor: Rotated weight tensor with same shape, dtype, and device as input.

    Note:
        Temporarily moves tensors to CUDA for computation if on CPU.
    """
    device = weight.device
    dtype = weight.dtype
    if device.type == "cpu":
        weight = weight.to("cuda")

    cal_dtype = torch.float64 if use_float64 else torch.float32
    weight = weight.to(cal_dtype)
    if input_spin is not None:
        input_spin = input_spin.to(dtype=cal_dtype, device=weight.device)
    if output_spin is not None:
        output_spin = output_spin.to(dtype=cal_dtype, device=weight.device)
    matmul_spin = matmul_diag_fast_hadamard if fast_hadamard else matmul_diag
    if input_spin is not None or (input_spin is None and fast_hadamard and block_size is not None):
        weight = matmul_spin(
            weight, input_spin, block_size=block_size
        )  # torch.matmul(weight, input_spin)
    if output_spin is not None:
        weight = diag_matmul(output_spin, weight)

    # offload the rotation tensors to cpu
    if input_spin is not None:
        input_spin.to("cpu")
    if output_spin is not None:
        output_spin.to("cpu")

    weight = weight.to(dtype=dtype, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    return weight


@with_dtensor_support
def rotate_weight(
    weight, input_spin, output_spin, fast_hadamard=False, block_size=None, use_float64=True
):
    """Rotating a matrix with DTensor support.

    This function supports both regular tensors and DTensors. For DTensors:
    - Gathers the full tensor using .full_tensor()
    - Runs rotation only on rank 0
    - Broadcasts result to other ranks
    - Redistributes back to original DTensor placements

    Args:
        weight: Weight tensor to rotate (can be DTensor or regular tensor)
        input_spin: Input spin matrix
        output_spin: Output spin matrix
        fast_hadamard: Whether to use fast Hadamard transform
        block_size: Block size for Hadamard transform
        use_float64: Whether to use float64 for computation

    Returns:
        Rotated weight tensor (same type as input)
    """
    return _rotate_weight_impl(
        weight, input_spin, output_spin, fast_hadamard, block_size, use_float64
    )


def _rotate_bias_impl(bias, output_spin, fast_hadamard=False, block_size=None, use_float64=True):
    """Core implementation of bias rotation (non-DTensor version).

    Applies output rotation matrix to bias vector: bias @ output_spin^T.
    Automatically handles device placement and dtype conversions.

    Args:
        bias (torch.Tensor): Bias vector to rotate, shape (out_features,).
        output_spin (torch.Tensor or None): Output rotation matrix. If None, returns
            the bias unchanged.
        fast_hadamard (bool, optional): If True, use fast Hadamard transform instead
            of explicit matrix multiplication. Defaults to False.
        block_size (int, optional): Block size for chunked Hadamard transform.
            Defaults to None.
        use_float64 (bool, optional): If True, perform computations in float64 for
            higher precision. Defaults to False (uses float32).

    Returns:
        torch.Tensor: Rotated bias vector with same shape, dtype, and device as input.

    Note:
        Temporarily moves tensors to CUDA for computation if on CPU.
    """
    dtype = bias.dtype
    device = bias.device
    if device.type == "cpu":
        bias = bias.to("cuda")
    output_spin.to(bias.device)

    cal_dtype = torch.float64 if use_float64 else torch.float32
    bias = bias.to(cal_dtype)
    if output_spin is not None:
        output_spin = output_spin.to(cal_dtype)
    matmul_spin = matmul_diag_fast_hadamard if fast_hadamard else matmul_diag
    bias = bias.unsqueeze(0)
    bias = matmul_spin(bias, output_spin, block_size=block_size, transpose=True)
    bias = bias.squeeze(0)
    output_spin.to("cpu")
    bias = bias.to(dtype=dtype, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    return bias


@with_dtensor_support
def rotate_bias(bias, output_spin, fast_hadamard=False, block_size=None, use_float64=True):
    """Rotate a bias with DTensor support.

    This function supports both regular tensors and DTensors. For DTensors:
    - Gathers the full tensor using .full_tensor()
    - Runs rotation only on rank 0
    - Broadcasts result to other ranks
    - Redistributes back to original DTensor placements

    Args:
        bias: Bias tensor to rotate (can be DTensor or regular tensor)
        output_spin: Output spin matrix
        fast_hadamard: Whether to use fast Hadamard transform
        block_size: Block size for Hadamard transform
        use_float64: Whether to use float64 for computation

    Returns:
        Rotated bias tensor (same type as input)
    """
    return _rotate_bias_impl(bias, output_spin, fast_hadamard, block_size, use_float64)


def get_device(module, model):
    """Get the device of the module.

    Attempts to find the device by checking parameters and buffers of the module.
    Falls back to the model's device if no parameters or buffers are found.

    Args:
        module (torch.nn.Module): The module to get the device from.
        model (torch.nn.Module): The parent model to use as fallback.

    Returns:
        torch.device: The device where the module resides.
    """
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return model.device


def is_dtensor(tensor):
    """Check if the tensor is a DTensor (Distributed Tensor)."""
    return isinstance(tensor, torch.distributed.tensor.DTensor)


class RotationMatrixStore:
    """Store and manage rotation matrices for model layers.

    This class handles initialization, storage, and retrieval of rotation matrices
    used in model quantization. Supports per-layer rotation matrices, pattern-based
    matrix assignment, and various rotation modes (Hadamard, orthogonal, etc.).

    Attributes:
        config (dict): Configuration dictionary specifying rotation matrix properties.
        model (torch.nn.Module): The model to apply rotations to.
        matrices (list): List of (module, name) tuples tracking all rotation matrices.
        _num_decoder_layers (int): Number of decoder layers in the model.
    """

    def __init__(self, config, model):
        """Initialize the rotation matrix store.

        Args:
            config (dict): Configuration dictionary with rotation matrix specifications.
                Each key is a matrix name, and values contain:
                - 'dim' (int or str): Dimension of the rotation matrix
                - 'mode' (str): Type of rotation ('hadamard', 'orthogonal', etc.)
                - 'per_layer' (bool or str): If True/str, create separate matrices per layer
            model (torch.nn.Module): The model to apply rotations to.
        """
        self.config = config
        # list of (module, name), module.name is the rotation matrix
        self.matrices = []
        self.model = model
        self._num_decoder_layers = getattr(model.config, "num_hidden_layers", 0)
        if self._num_decoder_layers == 0:
            for name, _ in model.named_modules():
                if re.search("layers\\.[0-9]+$", name) is not None:
                    self._num_decoder_layers += 1
        # let's initialize the matrices on cpu first, then move to the device of the model
        self.init_matrices(device="cpu")

    def is_fast_hadamard(self, name):
        """Check if the rotation matrix is a fast Hadamard matrix.

        Args:
            name (str): Name of the rotation matrix to check.

        Returns:
            bool: True if the matrix mode is 'fast_hadamard', False otherwise.
        """
        return self.config.get(name).get("mode", "hadamard") == "fast_hadamard"

    def init_matrices(self, device=None):
        """Initialize the rotation matrices based on configuration.

        Creates rotation matrices and attaches them as attributes to the appropriate
        modules (per-layer, pattern-based, or model-level).

        Args:
            device (torch.device, optional): Device to create matrices on. If None,
                uses the device of each target module. Defaults to None.

        Note:
            Matrices are stored as attributes on modules and tracked in self.matrices.
        """
        for name in self.config:
            per_layer = self.config.get(name).get("per_layer", False)
            if isinstance(per_layer, str):
                for moudle_name, module in _iter_modules_by_pattern(self.model, per_layer):
                    print(f"Initializing {name} for {moudle_name}")
                    dim = self.config.get(name).get("dim")
                    if isinstance(dim, str):
                        dim = getattr(self.model.config, dim)
                    setattr(
                        module,
                        name,
                        get_orthogonal_matrix(
                            dim,
                            self.config.get(name).get("mode", "hadamard"),
                            device or get_device(module, self.model),
                        ),
                    )
                    self.matrices.append((module, name))
            elif per_layer:
                for moduel_name, module in _iter_modules_by_pattern(
                    self.model, "layers\\.[0-9]+$", regex=True
                ):
                    print(f"Initializing {name} for layer {moduel_name}")
                    dim = self.config.get(name).get("dim")
                    if isinstance(dim, str):
                        dim = getattr(self.model.config, dim)
                    setattr(
                        module,
                        name,
                        get_orthogonal_matrix(
                            dim,
                            self.config.get(name).get("mode", "hadamard"),
                            device or get_device(module, self.model),
                        ),
                    )
                    self.matrices.append((module, name))
            else:
                print(f"Initializing {name} for root model")
                dim = self.config.get(name).get("dim")
                if isinstance(dim, str):
                    dim = getattr(self.model.config, dim)
                setattr(
                    self.model,
                    name,
                    get_orthogonal_matrix(
                        dim,
                        self.config.get(name).get("mode", "hadamard"),
                        device or get_device(self.model, self.model),
                    ),
                )
                self.matrices.append((self.model, name))

    def get(self, name, module_name=None):
        """Get the rotation matrix by name.

        Args:
            name: Name of the rotation matrix
            module_name: Name of the module that the rotation matrix will be applied to.

        Returns:
            The rotation matrix
        """
        if module_name is not None:
            module = self.model.get_submodule(module_name)
            while not hasattr(module, name) and "." in module_name:
                module_name = module_name.rsplit(".", 1)[0]
                module = self.model.get_submodule(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        return getattr(self.model, name)

    def clear(self):
        """Clear all rotation matrices and free memory.

        Removes all rotation matrix attributes from modules and performs garbage
        collection to free memory.

        Note:
            After calling this method, the rotation matrices are no longer accessible
            from the modules.
        """
        for module, name in self.matrices:
            delattr(module, name)
        gc.collect()
        torch.cuda.empty_cache()
