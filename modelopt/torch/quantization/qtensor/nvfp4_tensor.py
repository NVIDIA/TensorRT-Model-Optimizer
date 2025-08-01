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

"""Implements NVFP4 quantization for efficient tensor storage and computation."""

import numpy as np
import torch

from ..backends.utils import fp4_compatible
from ..qtensor.base_qtensor import BaseQuantizedTensor
from ..utils import reduce_block_padding

# Define conversion tables
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

__all__ = ["NVFP4QTensor"]


class NVFP4QTensor(BaseQuantizedTensor):
    """Implements the INT4 quantization on tensors for more efficient storage or computation.

    Attributes:
        quantized_data (torch.Tensor): The quantized data stored as a packed uint8 tensor.
    """

    e2m1_values_on_device = {}

    @classmethod
    def get_e2m1_values(cls, device):
        """Returns the e2m1 values on the device."""
        if device not in cls.e2m1_values_on_device:
            cls.e2m1_values_on_device[device] = e2m1_values.to(device)
        return cls.e2m1_values_on_device[device]

    @classmethod
    def get_weights_scaling_factor_2_from_quantizer(cls, weight_quantizer):
        """Returns per tensor weight scaling factor from the weight_quantizer amax."""
        # Assert that weight_quantizer has attribute amax
        assert hasattr(weight_quantizer, "_amax"), "Weight quantizer does not have attribute amax"
        return weight_quantizer._amax.float() / 6.0 / 448.0

    @classmethod
    def get_weights_scaling_factor(
        cls,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Returns quantized per block weight scaling factor."""
        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)

        # Get per_block amax
        [n, k] = input.shape[-2:]
        assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

        assert k % block_size == 0, (
            "Weight shape is not divisible for block size for block quantiation."
        )

        input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
        # Get per block amax
        per_block_amax = input.abs().amax(dim=-1).float()
        # Get per-block-scale
        per_block_scale = per_block_amax / 6.0
        # Quantize per_block_scale to FP8
        q_per_block_scale = per_block_scale / weights_scaling_factor_2
        # Set all zero values in scale to 1.0
        q_per_block_scale[per_block_scale == 0] = 1.0
        # Convert to torch.float8_e4m3fn
        if not keep_high_precision:
            q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
        return q_per_block_scale, weights_scaling_factor_2

    @classmethod
    def get_weights_scaling_factor_2(cls, input: torch.Tensor):
        """Returns per tensor weight scaling factor."""
        return input.abs().amax().float() / 6.0 / 448.0

    @classmethod
    def get_activation_scaling_factor(cls, quantizer):
        """Returns the activation scaling factor for export."""
        # TODO: Update to use module and not quantizer
        if not quantizer.is_enabled:
            return None

        amax = quantizer.export_amax()

        if amax is None:
            return None

        activation_scaling_factor = amax.float() / (quantizer.maxbound)
        activation_scaling_factor = activation_scaling_factor / 448.0

        assert torch.all(activation_scaling_factor > 0), (
            f" activation scaling factor {activation_scaling_factor} not positive."
        )

        return activation_scaling_factor

    @staticmethod
    def _cast_fp4(weight: torch.Tensor):
        """Converts tensor to uint4."""
        # Get device
        device = weight.device

        # Define mask to perform rounding
        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
        mask_shape = list(weight.shape)
        mask = mask.expand([*mask_shape, 7])

        sign_bit = (weight < 0).to(torch.uint8)

        weight_abs = weight.abs_()
        # Calculate the ordinal value based on the bounds
        ord = torch.searchsorted(e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
        # All values equal to e2m1_bounds at odd indices are rounded up and even indices are rounded down
        round = torch.any((weight_abs.unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1)
        fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
        return fp4_val

    @classmethod
    def quantize(
        cls,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor: torch.Tensor | None = None,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
        try_tensorrt: bool = False,
    ):
        """Converting a tensor to a quantized format based on NVFP4 quantization.

        Args:
            input (torch.Tensor): The input tensor to be quantized.
            block_size (int): The size of each block for quantization.
            weights_scaling_factor (torch.Tensor): The scaling factor for the weights.
            weights_scaling_factor_2 (torch.Tensor): The scaling factor for the weights.
            keep_high_precision (bool): Whether to keep output scales at high precision.

        Returns:
        tuple: Contains quantized data, quantized per block scaling factor, and per tensor scaling factor.
        """
        # Get original input shape
        input_shape = input.shape
        input_dtype = input.dtype

        # pad the input if needed
        input = reduce_block_padding(input, block_sizes={-1: block_size})

        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)

        # try call trtllm fp4 quantization if possible
        if (
            fp4_compatible()
            and weights_scaling_factor is None
            and try_tensorrt
            and block_size == 16
        ):
            try:
                import tensorrt_llm  # noqa: F401

                # Make sure this utils is available for dequantize
                from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
                    cutlass_fp4_scale_to_modelopt_fp4_scale,  # noqa: F401
                )

                packed_weight, weights_scaling_factor = torch.ops.trtllm.fp4_quantize(
                    input, 1.0 / weights_scaling_factor_2, block_size, False
                )
                # weights_scaling_factor is ready for nvfp4_gemm to use;
                # however, it is different from the non trtllm version, so when dequantize,
                # it will be converted.
                return (
                    cls(input_shape, input_dtype, packed_weight),
                    weights_scaling_factor,
                    weights_scaling_factor_2,
                )
            except ImportError:
                pass

        if weights_scaling_factor is None:
            weights_scaling_factor, _ = cls.get_weights_scaling_factor(
                input, block_size, weights_scaling_factor_2
            )

        # Reshape the weight and scale factors
        input = input.view((*tuple(input.shape[:-1]), -1, block_size))

        # Scale weights
        scaled_weight = input / (
            (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
        )

        # Reshape weights to original
        scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

        if keep_high_precision:
            return scaled_weight
        # Cast weights to fp4
        q_weight = cls._cast_fp4(scaled_weight)
        # Pack weights
        packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
        return (
            cls(input_shape, input_dtype, packed_weight),
            weights_scaling_factor,
            weights_scaling_factor_2,
        )

    def dequantize(self, dtype: torch.dtype = None, **kwarg):
        """Dequantze NVFP4 packed tensor to a target dtype."""
        if dtype is None:
            dtype = self.metadata["dtype"]

        def _unpack_tensor(input: torch.Tensor):
            # Initalize storage for unpacked tensor
            unpacked = torch.empty(
                [input.shape[0], input.shape[1] * 2], dtype=dtype, device=input.device
            )
            unpacked_shape = unpacked.shape

            unpacked[..., 1::2] = input >> 4
            unpacked[..., 0::2] = input & 0x0F

            unpacked = unpacked.reshape(-1)
            unpacked = self.get_e2m1_values(input.device)[unpacked.long()]

            return unpacked.reshape(unpacked_shape)

        # Get scales from kwargs
        if kwarg["scale"].dtype == torch.uint8 and kwarg["scale"].ndim == 1:
            # If quantization is done by trtllm, convert cutlass fp4 scale to modelopt fp4 scale
            try:
                from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
                    cutlass_fp4_scale_to_modelopt_fp4_scale,
                )

                kwarg["scale"] = cutlass_fp4_scale_to_modelopt_fp4_scale(
                    kwarg["scale"], self.metadata["shape"][-2:]
                )
            except ImportError as e:
                raise ImportError(
                    "This tensor is quantized by trtllm, but tensorrt_llm cannot be imported."
                ) from e
        q_per_block_scale = (
            kwarg["scale"].to(torch.float32)
            if kwarg["scale"].dtype == torch.float8_e4m3fn
            else kwarg["scale"]
        )
        block_sizes = kwarg["block_sizes"][-1]
        per_block_quant_scale = kwarg["double_scale"]

        # Dequantize scales
        per_block_scale = q_per_block_scale * per_block_quant_scale

        # Unpack and unscale weights
        deq_data = _unpack_tensor(self._quantized_data)

        deq_data = deq_data.view(
            deq_data.shape[0], deq_data.shape[1] // block_sizes, -1
        ) * per_block_scale.unsqueeze(-1)

        return (
            deq_data.view(-1)[: np.prod(self.metadata["shape"])]
            .reshape(self.metadata["shape"])
            .to(dtype)
        )
