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

import torch


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

    def __new__(cls, qtensor: BaseQuantizedTensor):
        """Create a new QTensorWrapper instance."""
        quantized_tensor = qtensor._quantized_data
        instance = super().__new__(cls, quantized_tensor, requires_grad=False)
        instance.metadata = qtensor.metadata
        instance.metadata["qtensor_class"] = qtensor.__class__
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


def pack_real_quantize_weight(module, force_quantize: bool = False):
    """Pack real quantized tensors to a compressed format and set proper load_state_dict function."""

    # Function to dynamically override load_state_dict
    def dynamically_update_state_methods(module):
        # Original method
        original_load_from_state_dict = module._load_from_state_dict

        def custom_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            """Override _load_from_state_dict to handle custom parameters dynamically."""
            args_list = list(args)

            deleted_stat_dict = {}
            for name, param in self.named_parameters():
                if isinstance(param, QTensorWrapper) and prefix + name in state_dict:
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

    with torch.no_grad():
        for _, m in module.named_modules():
            if (
                hasattr(m, "weight_quantizer")
                and m.weight_quantizer.is_enabled
                and not m.weight_quantizer._fake_quant
            ):
                if force_quantize:
                    m.weight_quantizer._dequantize = False
                assert not m.weight.is_meta, (
                    "Real quantization does not support tensors on meta device."
                )
                real_quant_tensor = m.weight_quantizer(m.weight)
                m.weight = QTensorWrapper(real_quant_tensor)
                dynamically_update_state_methods(m)
