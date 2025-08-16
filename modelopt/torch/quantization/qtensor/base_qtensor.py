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

"""Base Class for Real Quantized Tensor."""

import enum
import warnings
from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor


class QTensorType(enum.Enum):
    """Enumeration for defining types of quantization."""

    INT4 = 1
    INT8 = 2
    FP8 = 3
    NF4 = 4


__all__ = ["BaseQuantizedTensor", "QTensorWrapper", "pack_real_quantize_weight"]


class BaseQuantizedTensor:
    """Base class for quantized tensors, providing methods for quantization and dequantization.

    This class should be subclassed to implement specific types of quantized tensors. It handles the
    storage of quantized data along with the necessary configurations and original attributes.

    Attributes:
        original_meta_tensor (torch.Tensor): Original meta to keep attributes of original tensors.
        quantized_data (torch.Tensor): Storage for the quantized tensor data. Quantized_data dtype is
            customized per QuantizedTensor implementation.
    """

    _quantized_data: torch.Tensor

    def __init__(
        self,
        original_shape: torch.Size,
        original_dtype: torch.dtype,
        quantized_data: torch.Tensor,
    ):
        """Initialize data attributes."""
        self.metadata = {
            "shape": original_shape,
            "dtype": original_dtype,
        }
        self._quantized_data = quantized_data

    @classmethod
    def quantize(cls, input: torch.Tensor, block_size: int):
        """Pack a fake torch.Tensor into a real quantized tensor.

        Args:
            fake_quant_tensor (torch.Tensor): The fake quantized tensor.

        Returns:
            A real quantized tensor, scales.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def dequantize(self, dtype: torch.Tensor = None, **kwarg):
        """Converts the quantized tensor back to a standard torch.Tensor.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")


class QTensorWrapper(torch.nn.Parameter):
    """A wrapper class for quantized tensors to make them compatible with torch.nn.Parameter.

    Args:
        qtensor (BaseQuantizedTensor): The quantized tensor to be wrapped.
    """

    def __new__(cls, qtensor: BaseQuantizedTensor | torch.Tensor, metadata: dict | None = None):
        """Create a new QTensorWrapper instance."""
        quantized_tensor = (
            qtensor._quantized_data if isinstance(qtensor, BaseQuantizedTensor) else qtensor
        )
        instance = super().__new__(cls, quantized_tensor, requires_grad=False)
        if metadata is None:
            instance.metadata = qtensor.metadata
            instance.metadata["qtensor_class"] = qtensor.__class__
        else:
            assert all(key in metadata for key in ["qtensor_class", "shape", "dtype"]), (
                f"metadata: {metadata}"
            )
            instance.metadata = metadata
        return instance

    def dim(self):
        """Return the number of dimensions of the meta_tensor."""
        return len(self.metadata["shape"])

    def to(self, *args, **kwargs):
        """Override the `to` method to move real quantized tensors to the specified device."""
        changing_device, changing_dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if changing_device:
            self.data = self.data.to(device=changing_device)
        dtype = changing_dtype if changing_dtype else self.metadata["dtype"]
        return QTensorWrapper(
            self.metadata["qtensor_class"](self.metadata["shape"], dtype, self.data)
        )

    def get_qtensor(self):
        """Get the quantized tensor class from QTensorWrapper."""
        return self.metadata["qtensor_class"](
            self.metadata["shape"], self.metadata["dtype"], self.data
        )

    def get_state(self):
        """Get the state of the QTensorWrapper."""
        return {
            "metadata": self.metadata,
            "quantized_data.shape": self.data.shape,
            "quantized_data.dtype": self.data.dtype,
        }


class QFSDPParam(FSDPParam):
    """A Quantized FSDPParam class to make weight updates compatible with BaseQuantizedTensor and QTensorWrapper.

    With this class, we can keep track of the quantized tensor's metadata when compressing the weights
    and recreate the QTensorWrapper with the correct metadata, when unsharding the FSDPModule.

    Args:
        qtensor (BaseQuantizedTensor): The quantized tensor to be wrapped.
    """

    def __init__(self, *args, **kwargs):
        # Store qtensor information
        self.metadata = args[0].metadata
        super().__init__(*args, **kwargs)
        self.init_dtype_attrs(self.mp_policy)

    def _setattr_on_modules(self, param: torch.nn.Parameter) -> None:
        if not isinstance(param, DTensor):
            # Create a QTensorWrapper with the correct metadata during unsharding
            param = QTensorWrapper(param, metadata=self.metadata)
        super()._setattr_on_modules(param)


# Function to dynamically override load_state_dict
def dynamically_update_state_methods(module):
    # Original method
    original_load_from_state_dict = module._load_from_state_dict

    def custom_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Override _load_from_state_dict to handle custom parameters dynamically."""
        args_list = list(args)
        deleted_stat_dict = {}

        # Load parameters
        for name, param in self.named_parameters():
            if prefix + name in state_dict and isinstance(param, QTensorWrapper):
                if param.data.is_meta:
                    # Wrap the loaded tensor in a QTensorWrapper
                    param = QTensorWrapper(state_dict[prefix + name], metadata=param.metadata)
                    self.register_parameter(name, param)
                else:
                    param.copy_(state_dict[prefix + name])
                deleted_stat_dict[prefix + name] = state_dict[prefix + name]
                del state_dict[prefix + name]

        # Set strict=False because weight keys are removed
        kwargs = {}
        if len(args_list) > 3:
            args_list[1] = False
        else:
            kwargs = {"strict": False}
        original_load_from_state_dict(state_dict, prefix, *args_list, **kwargs)
        state_dict.update(**deleted_stat_dict)

    module._load_from_state_dict = custom_load_from_state_dict.__get__(module, type(module))


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


def pack_real_quantize_weight(module, force_quantize: bool = False):
    """Pack real quantized tensors to a compressed format and set proper load_state_dict function."""
    # Import SequentialQuantizer here to avoid circular import
    from ..nn import SequentialQuantizer

    def _compress_and_update_module_weight(module):
        """Compresses and updates module weights if quantizer is enabled. Returns True when compression is applied."""
        if hasattr(module, "weight") and (module.weight is None or module.weight.is_meta):
            # We dont compress meta tensors or None
            return False
        if (
            hasattr(module, "weight_quantizer")
            and module.weight_quantizer.is_enabled
            and not module.weight_quantizer._fake_quant
            and module.weight.element_size() > 1
        ):
            if force_quantize:
                module.weight_quantizer._dequantize = False

            real_quant_tensor = module.weight_quantizer(module.weight)
            module.weight = QTensorWrapper(real_quant_tensor)
            return True

        return False

    def _create_fsdp_param_mapping(fsdp_param_list, model):
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

    def _compress_fsdp_module(fsdp_module):
        """Applies weight compression to an FSDP-wrapped module and updates its sharded parameter group.

        This function unshards the FSDP module to access full weights and compresses each eligible submodule’s weights.
        A new FSDPParam wrapped with `QFSDPParam` is registered to the FSDPParamGroup for future handling of
        sharding and unsharding. The weight_scale buffers registered during compression and the FSDPModule are reharded
        once compression is complete.

        Args:
            fsdp_module (nn.Module): The FSDP-wrapped module to compress.

        Returns:
            None
        """
        # Unshard FSDPmodule by temporarily setting _fake_quant to prevent weight compression from being triggered
        with enable_fake_quant(fsdp_module):
            fsdp_module.unshard()

        # Get the FSDPParamGroup for the FSDPModule
        fsdp_param_group = fully_shard.state(fsdp_module)._fsdp_param_group

        if getattr(fsdp_param_group, "fsdp_params", None) is None:
            warnings.warn(
                f"FSDPParamGroup for {fsdp_module} has no fsdp_params, skipping compression"
            )
            return

        # Create FSDPParam mapping dictionary to keep track of FSDPParams to update/delete
        fsdp_param_mapping = _create_fsdp_param_mapping(fsdp_param_group.fsdp_params, fsdp_module)

        for name, submodule in fsdp_module.named_modules():
            # This is to handle case where the root FSDPModule has parameters.
            # We skip all the parameters that dont belong to the FSDPParamGroup.
            if name not in fsdp_param_mapping:
                continue

            if _compress_and_update_module_weight(submodule):
                old_fsdp_param = fsdp_param_mapping[name]

                # Update mp policy to reflect the new dtype
                new_mp_policy = MixedPrecisionPolicy(
                    param_dtype=submodule.weight.dtype,
                    reduce_dtype=None,
                    output_dtype=None,
                    cast_forward_inputs=False,
                )
                with no_requires_grad():
                    # Create a new QFSDPParam parameter
                    new_param = QFSDPParam(
                        submodule.weight,
                        old_fsdp_param._module_info,
                        old_fsdp_param.mesh_info,
                        old_fsdp_param.post_forward_mesh_info,
                        old_fsdp_param.device,
                        None,
                        new_mp_policy,
                        None,
                    )

                    # Update the FSDPParam mapping to keep track of the new FSDPParam
                    fsdp_param_mapping[name] = new_param
                    # Remove the post_load_hook_handle to allow gc to collect the old FSDPParam
                    old_fsdp_param._post_load_hook_handle.remove()

        # Update FSDPParam list with new compressed weights
        fsdp_param_group.fsdp_params = list(fsdp_param_mapping.values())

        # Reshard FSDP root module
        fsdp_module.reshard()

    with SequentialQuantizer.convert_to_single_quantizer(module), torch.no_grad():
        for _, m in module.named_modules():
            # If FSDP module, we need to additionally process the FSDPParam list
            if isinstance(m, FSDPModule):
                _compress_fsdp_module(m)
            else:
                # Compress weights and update module weight
                _compress_and_update_module_weight(m)
