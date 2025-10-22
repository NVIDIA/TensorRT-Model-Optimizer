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

from __future__ import annotations

from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import Replicate

from modelopt.torch.quantization.qtensor.base_qtensor import QFSDPParam, QTensorWrapper
from modelopt.torch.utils import get_unwrapped_name, print_rank_0

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "EXPORT_MODE",
    "convert_quantization_axis_to_reduce_axis",
    "export_torch_mode",
    "is_quantized",
    "is_quantized_column_parallel_linear",
    "is_quantized_linear",
    "is_quantized_row_parallel_linear",
    "reduce_amax",
    "replace_function",
    "weight_attr_names",
]


def reduce_block_amax(input_tensor: torch.Tensor, block_sizes: dict):
    """Computes the amax of the input tensor using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for each dimension.
                            Example: `{-1: 128, -2: 128}` reduces over 2D blocks.

    Returns:
        torch.Tensor: The reduced tensor with amax computed per block.

    Example:
        Input Shape: [256, 512]
        Block Sizes: {-1: 128, -2: 128}
        Process:
            - Block along last dim → Shape [256, 4, 128]
            - Compute block-wise amax → Shape [256, 4]
            - Block along second-to-last dim → Shape [2, 128, 4]
            - Compute block-wise amax → Shape [2, 4]
    """
    with torch.no_grad():
        amax = input_tensor.clone()

        for dim, block_size in block_sizes.items():
            # Convert negative dimensions to positive
            dim = dim if dim >= 0 else len(amax.shape) + dim
            assert amax.shape[dim] % block_size == 0, (
                f"Tensor dimension {amax.shape[dim]}, {amax.shape[dim]} is not divisible by {block_size}"
            )

            # Compute new shape for blocking
            outer_dim = amax.shape[dim] // block_size
            new_shape = [
                *list(amax.shape[:dim]),
                outer_dim,
                block_size,
                *list(amax.shape[dim + 1 :]),
            ]

            # Reshape into blocks
            amax = amax.reshape(new_shape)

            # Reduce along the newly created block dimension
            # Shift by 1 because we added an extra dimension
            amax = reduce_amax(amax, dim + 1, keepdims=False, squeeze_scalar=False)

        return amax


def reduce_block_padding(input: torch.Tensor, block_sizes: dict, pad_value: float = 0):
    """Padding the input using block-based reduction for each dimension.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        block_sizes (dict): A dictionary specifying the block size for padding each dimension.
                            Example: `{-1: 128, -2: 128}` pads the input over 2D blocks.
    """
    with torch.no_grad():
        padded_tensor = input
        num_dims = padded_tensor.dim()

        # Process each specified dimension independently
        for dim, block in block_sizes.items():
            # Convert negative dimension to positive index
            pos_dim = dim if dim >= 0 else num_dims + dim

            # Calculate how many elements are missing along that dimension
            current_size = padded_tensor.size(pos_dim)
            remainder = current_size % block
            pad_amt = 0 if remainder == 0 else block - remainder

            if pad_amt > 0:
                # F.pad expects a pad tuple of length 2*num_dims.
                pad = [0] * (2 * num_dims)
                # For dimension pos_dim, the right padding is at index: (num_dims - 1 - pos_dim)*2 + 1.
                pad_index = (num_dims - 1 - pos_dim) * 2
                pad[pad_index + 1] = (
                    pad_amt  # Set padding on the right side of the target dimension
                )

                padded_tensor = F.pad(padded_tensor, pad, value=pad_value)

        return padded_tensor


def convert_quantization_axis_to_reduce_axis(input, axis):
    """Convert the quantization axis to the reduce axis.

    Args:
        input (torch.Tensor): The input tensor.
        axis (int, tuple, list of None): The quantization axis. None means per-tensor quantization.

    Returns:
        list: The axis to reduce. None suggests all dimensions should be reduced.
    """
    if axis is None:
        return None
    axis = axis if isinstance(axis, (list, tuple)) else [axis]
    # Handle positive and negative axis.
    reduce_axis = [i for i in range(input.dim()) if i not in axis and (i - input.dim()) not in axis]
    return reduce_axis


@torch.no_grad()
def reduce_amax(input, axis=None, keepdims=True, squeeze_scalar=True):
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

    Returns:
        The reduced tensor.
    """
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
        if squeeze_scalar and output.numel() == 1:
            output.squeeze_()
    return output


def weight_attr_names(module: nn.Module) -> Generator[str, None, None]:
    """Get the weight param attribute names in a converted module, non-recursive.

    We consider the following two cases for each weight param attribute:
    - The standard weight attribute (e.g. nn.Linear).
    - The custom `weight_attr_name`. (e.g. Llama4TextExperts has weight attributes `gate_up_proj` and `down_proj`)
    """
    from .nn import SequentialQuantizer, TensorQuantizer

    # the standard weight and quantizer case
    weight = getattr(module, "weight", None)
    weight_quantizer = getattr(module, "weight_quantizer", None)
    if isinstance(weight, nn.Parameter) and isinstance(
        weight_quantizer, (TensorQuantizer, SequentialQuantizer)
    ):
        yield "weight"

    # other weight and quantizer case
    for name, _ in module.named_parameters(recurse=False):
        weight = getattr(module, name, None)
        weight_quantizer = getattr(module, f"{name}_weight_quantizer", None)
        if isinstance(weight, nn.Parameter) and isinstance(
            weight_quantizer, (TensorQuantizer, SequentialQuantizer)
        ):
            yield name


"""The whole set of quantizer related attribute names for a given weight name."""
QuantizerAttrNames = namedtuple(
    "QuantizerAttrNames",
    (
        "weight_quantizer",
        "input_quantizer",
        "output_quantizer",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "output_scale",
    ),
)


def quantizer_attr_names(weight_name: str = "weight") -> QuantizerAttrNames:
    """Get all the quantizer related attribute names for a given weight name."""
    prefix = f"{weight_name}_" if weight_name != "weight" else ""
    return QuantizerAttrNames(
        weight_quantizer=f"{prefix}weight_quantizer",
        input_quantizer=f"{prefix}input_quantizer",
        output_quantizer=f"{prefix}output_quantizer",
        weight_scale=f"{prefix}weight_scale",
        weight_scale_2=f"{prefix}weight_scale_2",
        input_scale=f"{prefix}input_scale",
        output_scale=f"{prefix}output_scale",
    )


def is_quantized(module):
    """Check if a module is quantized."""
    from .nn import TensorQuantizer

    return any(isinstance(_module, TensorQuantizer) for _module in module.modules())


def is_quantized_linear(module):
    """Check if a module is a quantized linear module."""
    from .nn import QuantModule, TensorQuantizer

    return (
        isinstance(module, QuantModule)
        and isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
        and hasattr(module, "weight_quantizer")
        and (
            (getattr(module, "weight", None) is not None and module.weight.dim() == 2)
            # module.weight0 check is required to support TEGroupedLinear
            or (getattr(module, "weight0", None) is not None and module.weight0.dim() == 2)
        )
    )


def is_quantized_column_parallel_linear(module):
    """Check if a module is a quantized column parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_column_parallel", False)


def is_quantized_row_parallel_linear(module):
    """Check if a module is a quantized row parallel linear module."""
    return is_quantized_linear(module) and getattr(module, "_is_row_parallel", False)


def is_quantized_parallel_linear(module):
    """Check if a module is a quantized parallel linear module."""
    return is_quantized_column_parallel_linear(module) or is_quantized_row_parallel_linear(module)


@contextmanager
def calibrate_with_adapters(model, args):
    """Disables LoRA adapters during calibration, then re-enables them afterward."""
    is_lora = getattr(args, "lora", None)
    if is_lora:
        print_rank_0("Disabling LoRA adapters during calibration...")
        model.disable_adapters()

    yield

    if is_lora:
        print_rank_0("Enabling LoRA adapters after calibration...")
        model.enable_adapters()


def disable_lora_quantizers_in_config(config, layers):
    """Turns off input, weight, and output quantizers for LoRA weights and LoRALinear layers in config."""
    config["quant_cfg"]["*lora*"] = {"enable": False}
    for layer in layers:
        config["quant_cfg"][f"*{layer}.input_quantizer"] = {"enable": False}
        config["quant_cfg"][f"*{layer}.weight_quantizer"] = {"enable": False}
        config["quant_cfg"][f"*{layer}.output_quantizer"] = {"enable": False}
    return config


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


def is_pow2(n):
    """Check if a number is the power of 2."""
    return (n != 0) and (n & (n - 1) == 0)


def _get_fsdp2_mesh(module: nn.Module):
    """Get the mesh info of the model."""
    try:
        from torch.distributed._composable_state import _get_module_state
    except ImportError:
        return None

    fsdp_state = _get_module_state(module)
    if (
        fsdp_state._fsdp_param_group
        and fsdp_state._fsdp_param_group.post_forward_mesh_info is not None
    ):
        return fsdp_state._fsdp_param_group.post_forward_mesh_info.mesh


def _get_module_name(module: nn.Module, root_model: nn.Module):
    name_to_module = dict(root_model.named_modules())
    target_module_name = next((name for name, m in name_to_module.items() if m is module), None)
    return target_module_name


def _get_enclosing_fsdp_module(module: nn.Module, root_model: nn.Module):
    """Get the enclosing FSDP module for a given module."""
    if isinstance(module, FSDPModule):
        return module

    name_to_module = dict(root_model.named_modules())
    target_module_name = _get_module_name(module, root_model)

    if target_module_name is None:
        raise ValueError(f"Module {module} not found in the root model {root_model}.")

    current_name = target_module_name
    while "." in current_name:
        parent_name = ".".join(current_name.split(".")[:-1])
        parent_module = name_to_module.get(parent_name)
        if parent_module and isinstance(parent_module, FSDPModule):
            return parent_module
        current_name = parent_name

    if isinstance(root_model, FSDPModule):
        return root_model


@contextmanager
def fsdp2_weight_access_and_writeback_context(module: nn.Module, root_model: nn.Module):
    """Context manager for FSDP2 weight access and writeback.

    Note this context will gather the weight across FSDP/HSDP shards. If TP is implemented with DTensor,
    the weight will be a local tensor of the TP DTensor under this context.
    """
    assert isinstance(root_model, torch.distributed.fsdp.FSDPModule), "We only support FSDP2"

    assert not hasattr(module, "_hf_hook"), "We dont support FSDP2 with HF accelerate hooks"
    assert isinstance(module.weight, torch.distributed.tensor.DTensor)
    fsdp_module = _get_enclosing_fsdp_module(module, root_model)
    assert fsdp_module is not None, "Module is not wrapped by FSDP"
    fsdp_device_mesh = _get_fsdp2_mesh(fsdp_module)
    fsdp_dim = fsdp_device_mesh.ndim

    original_placements = module.weight.placements
    original_device_mesh = module.weight.device_mesh
    original_weight = module.weight
    # Assuming the first fsdp_dim dimensions are for FSDP/HSDP, we only collect the tensor over FSDP/HSDP dimension,
    # the TP will be handled by the TP reduction.
    if fsdp_dim != original_device_mesh.ndim:
        assert fsdp_device_mesh.mesh_dim_names == original_device_mesh.mesh_dim_names[:fsdp_dim], (
            "FSDP2 mesh should be a slice of DTesnor's device mesh."
        )

    weight_collected = original_weight.redistribute(
        placements=[Replicate()] * fsdp_dim + list(original_placements[fsdp_dim:]),
        device_mesh=original_device_mesh,
    )
    new_weight = nn.Parameter(weight_collected.to_local())
    module._parameters["weight"] = new_weight

    yield

    original_weight.to_local().data.copy_(
        weight_collected.redistribute(
            placements=original_placements, device_mesh=original_device_mesh
        ).to_local()
    )
    module._parameters["weight"] = original_weight


@contextmanager
def enable_weight_access_and_writeback(module, root_model):
    """Enable weight access and writeback for a module.

    Useful for modules with weight not intact such as Linear layer in FSDP wrapped model or
    HF accelerate CPU off-loaded models.
    """
    if _get_enclosing_fsdp_module(module, root_model) is not None:
        context = fsdp2_weight_access_and_writeback_context(module, root_model)
    elif is_quantized_parallel_linear(module) and hasattr(module, "_hf_tp_plan"):
        # HF transformers TP sharded linear layer
        context = module.enable_weight_access_and_writeback()
    elif hasattr(module, "_hf_hook"):
        from .plugins.accelerate import weight_access_and_writeback_context

        context = weight_access_and_writeback_context(module)
    else:
        context = nullcontext()

    with context:
        yield


def get_quantizer_state_dict(model: nn.Module):
    """Get the state dict of the quantizers in the model."""
    # We should not call model.state_dict() here.
    # With FSDP, model.state_dict() will hang if it is not called from all processes
    from .nn import TensorQuantizer

    quantizer_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            quantizer_state_dict[get_unwrapped_name(name, model)] = module.state_dict()
    return quantizer_state_dict


def set_quantizer_state_dict(model: nn.Module, quantizer_state_dict: dict):
    """Set the state dict of the quantizers in the model."""
    from .nn import TensorQuantizer

    for name, module in model.named_modules():
        key = get_unwrapped_name(name, model)
        if isinstance(module, TensorQuantizer) and key in quantizer_state_dict:
            module.load_state_dict(quantizer_state_dict[key])


def patch_fsdp_mp_dtypes():
    """Patch FSDP2 to handle mixed dtypes properly during quantization."""

    def _init_mp_dtypes(self) -> None:
        """This function is directly copied from the latest version of torch FSDP."""
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)

        trainable_params: list[FSDPParam] = [
            p for p in self.fsdp_params if p.sharded_param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}

        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )

        self._orig_dtype = next(iter(orig_dtypes)) if len(trainable_params) else None

        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(f"FSDP expects uniform reduce dtype but got {reduce_dtypes}")

        self._reduce_dtype = next(iter(reduce_dtypes)) if len(trainable_params) else None

    # Apply the patch
    original_init_mp_dtypes = (
        torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup._init_mp_dtypes
    )
    torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup._init_mp_dtypes = (
        _init_mp_dtypes
    )
    return original_init_mp_dtypes


def get_prefixed_param_names(parent_model, target_module):
    """Get parameter names for a target module prefixed with the parent model name.

    This function is used to get full parameter name from FSDPParam module_info which stores the
    unprefixed parameter name.

    """
    target_ids = {id(p) for p in target_module.parameters()}
    return next(
        (
            name.rsplit(".", 1)[0]
            for name, param in parent_model.named_parameters()
            if id(param) in target_ids
        ),
        None,  # default value if no match
    )


def create_fsdp_param_mapping(fsdp_param_list, model):
    """Builds a mapping from module name to their corresponding FSDPParam.

    Args:
        fsdp_param_list (list): List of FSDPParam.
        model (nn.Module): FSDP root module.

    Returns:
        dict: Full parameter name → FSDP parameter.
    """
    return {
        get_prefixed_param_names(model, param._module_info.module): param
        for param in fsdp_param_list
    }


@contextmanager
def no_requires_grad():
    """Context manager to temporarily set requires_grad to False.

    This is used to allow us to call init_sharded_parameter() on the compressed weights. Currently FSDP2 creates
    a new parameter with default requires_grad and then update the requires_grad attribute as needed. This
    triggers an error when torch.nn.Parameter is called on compressed weights as requires_grad cannot be set to True
    for integer tensors.
    """
    original_new = torch.nn.Parameter.__new__

    def patched_new(cls, data=None, requires_grad=True):
        return original_new(cls, data, requires_grad=False)

    torch.nn.Parameter.__new__ = patched_new
    try:
        yield
    finally:
        torch.nn.Parameter.__new__ = original_new


@contextmanager
def enable_fake_quant(module):
    """Temporarily set the fake_quant attribute of a module to True.

    This is used to prevent weight compression from being triggered during an unshard() call.
    """
    original_fake_quant = []
    for m in module.modules():
        if hasattr(m, "weight_quantizer"):
            original_fake_quant.append(m.weight_quantizer._fake_quant)
            m.weight_quantizer._fake_quant = True
    yield
    for m in module.modules():
        if hasattr(m, "weight_quantizer"):
            m.weight_quantizer._fake_quant = original_fake_quant.pop(0)


@contextmanager
def fsdp2_aware_weight_update(root_model, modules_to_update):
    """Context manager to update the FSDPParam list if an update is made to a submodule of an FSDPModule."""
    try:
        if isinstance(root_model, FSDPModule):
            # Get FSDP root module, if none is returned, then the update is not made to a submodule of an FSDPModule
            if not isinstance(modules_to_update, list):
                modules_to_update = [modules_to_update]

            root_modules = set()
            for module in modules_to_update:
                root_module = _get_enclosing_fsdp_module(module, root_model)
                root_modules.add(root_module)

            # Ensure all modules in root_modules are the same
            assert len(root_modules) == 1, "All modules must be in the same root FSDPModule"
            root_module = next(iter(root_modules))

            # Check if root module state is sharded and unshard if needed
            if fully_shard.state(root_module)._fsdp_param_group.is_sharded:
                with enable_fake_quant(root_module):
                    root_module.unshard()

            # Get FSDPParam list
            fsdp_param_group = fully_shard.state(root_module)._fsdp_param_group
            fsdp_param_mapping = create_fsdp_param_mapping(fsdp_param_group.fsdp_params, root_model)

            # Assert that all the modules in the module list are present in this fsdp_param_group
            for module in modules_to_update:
                name = _get_module_name(module, root_model)
                assert name in fsdp_param_mapping, (
                    f"Module {module} not found in fsdp_param_mapping"
                )
        # Yields for necessary weight updates/processing
        yield
    finally:
        if isinstance(root_model, FSDPModule):
            # Update FSDPParam list
            for module in modules_to_update:
                name = _get_module_name(module, root_model)
                old_fsdp_param = fsdp_param_mapping[name]

                # Update mp policy to reflect the new dtype
                new_mp_policy = MixedPrecisionPolicy(
                    param_dtype=module.weight.dtype,
                    reduce_dtype=None,
                    output_dtype=None,
                    cast_forward_inputs=False,
                )

                with no_requires_grad():
                    # Create a new QFSDPParam or FSDPParam based on weight type
                    param_class = (
                        QFSDPParam if isinstance(module.weight, QTensorWrapper) else FSDPParam
                    )
                    new_param = param_class(
                        module.weight,
                        old_fsdp_param._module_info,
                        old_fsdp_param.mesh_info,
                        old_fsdp_param.post_forward_mesh_info,
                        old_fsdp_param.device,
                        None,
                        new_mp_policy,
                        None,
                    )
                    if not isinstance(new_param, QFSDPParam):
                        new_param.init_dtype_attrs(new_mp_policy)

                    # Update the FSDPParam mapping to keep track of the new FSDPParam
                    fsdp_param_mapping[name] = new_param

                    # Remove the post_load_hook_handle to allow gc to collect the old FSDPParam
                    old_fsdp_param._post_load_hook_handle.remove()

            # Update FSDPParam list with new compressed weights
            fsdp_param_group.fsdp_params = list(fsdp_param_mapping.values())

            # Reshard FSDP root module
            # TODO: Add a check to reshard only if necessary, can help performance during export
            root_module.reshard()
