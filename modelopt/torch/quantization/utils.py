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

"""Quantization utilities."""

from contextlib import ExitStack, contextmanager

import torch
from packaging.version import Version

from modelopt.torch.utils.distributed import ParallelState

__all__ = [
    "reduce_amax",
    "is_quantized",
    "is_quantized_layer_with_weight",
    "is_quantized_column_parallel_linear",
    "is_quantized_row_parallel_linear",
    "replace_function",
    "EXPORT_MODE",
    "export_torch_mode",
    "is_torch_library_supported",
    "get_parallel_state",
]


def reduce_amax(input, axis=None, keepdims=True):
    """Compute the absolute maximum value of a tensor.

    Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    the reduced dimensions are retained with length 1.

    .. note::
        Gradient computation is disabled as this function is never meant learning reduces amax

    Args:
        input: Input tensor
        axis: The dimensions to reduce. None or int or tuple of ints. If None (the default),
            reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
        keepdims: A boolean. If true, retains reduced dimensions with length 1. Default True
        granularity: DEPRECTED. specifies if the statistic has to be calculated at tensor or channel granularity

    Returns:
        The reduced tensor.

    Raises:
        ValueError: Any axis which doesn't make sense or is not supported
        ValueError: If unknown granularity is passed in.
    """
    with torch.no_grad():
        # A memory-efficient implementation that avoids copying input tensor
        if axis is None:
            max_val = torch.max(input)
            min_val = torch.min(input)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            max_val = torch.amax(input, dim=axis, keepdim=keepdims)
            min_val = torch.amin(input, dim=axis, keepdim=keepdims)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
            if output.numel() == 1:
                output.squeeze_()
        return output


def is_quantized(module):
    """Check if a module is quantized."""
    from .nn import TensorQuantizer

    for _module in module.modules():
        if isinstance(_module, TensorQuantizer):
            return True
    return False


def is_quantized_layer_with_weight(module):
    """Check if a module is quantized with weights."""
    return is_quantized(module) and getattr(module, "weight", None) is not None


def is_quantized_linear(module):
    """Check if a module is a quantized linear module."""
    return (
        hasattr(module, "input_quantizer")
        and hasattr(module, "weight_quantizer")
        and getattr(module, "weight", None) is not None
        and module.weight.dim() == 2
    )


def is_quantized_column_parallel_linear(module):
    """Check if a module is a quantized column parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_column_parallel", False)


def is_quantized_row_parallel_linear(module):
    """Check if a module is a quantized row parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_row_parallel", False)


@contextmanager
def replace_function(package, name, new_func):
    """Replace a function with a new one within a context."""
    old_func = getattr(package, name)
    setattr(package, name, new_func)
    setattr(package, "_" + name, old_func)
    yield
    setattr(package, name, old_func)
    delattr(package, "_" + name)


@contextmanager
def multi_context(*cms):
    """Context manager enabling variable number of context managers."""
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


EXPORT_MODE: bool = False


@contextmanager
def export_torch_mode():
    """Context manager enabling the export mode."""
    global EXPORT_MODE
    original_value = EXPORT_MODE
    EXPORT_MODE = True
    try:
        yield
    finally:
        EXPORT_MODE = original_value


def is_torch_export_mode():
    """Check whether in the context of exporting model to torch."""
    return EXPORT_MODE


def is_torch_library_supported():
    """Check if the installed PyTorch version meets or exceeds a specified version."""
    # Require torch version >= 2.2.0
    # Adding checks for `impl` and `impl_abstract` as they are experimental features
    return (
        Version(torch.__version__) >= Version("2.2.0")
        and hasattr(torch.library, "impl")
        and hasattr(torch.library, "impl_abstract")
    ) or (
        Version(torch.__version__) >= Version("2.4.0")
        and hasattr(torch.library, "impl")
        and hasattr(torch.library, "register_fake")
    )


def get_parallel_state(model, name=None) -> ParallelState:
    """Get the parallel state.

    Args:
        model: Pytorch model.
        name: The name of the submodule of the model to get the parallel state from. If None,
            the parallel state of the model is returned.
    """
    if name is None:
        return getattr(model, "_parallel_state", ParallelState())

    # If the submodule does not have a parallel state, get the parallel state of the parent module
    module = model.get_submodule(name)
    if hasattr(module, "_parallel_state"):
        return module._parallel_state
    parent_module = model.get_submodule(name.rpartition(".")[0])
    return getattr(parent_module, "_parallel_state", ParallelState())
