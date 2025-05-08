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

"""Quantized Linear."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import tensor_quant
from .quant_module import QuantLinearConvBase, QuantModuleRegistry, _LegacyQuantLinearConvBaseMixin
from .tensor_quantizer import TensorQuantizer

__all__ = ["Linear", "QuantLinear", "SVDQuantLinear"]


@QuantModuleRegistry.register({nn.Linear: "nn.Linear"})
class _QuantLinear(QuantLinearConvBase):
    """Quantized base class for nn.Linear type classes."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    @staticmethod
    def quantized_linear_fn(package, func_name, self, input, weight, *args, **kwargs):
        """Quantized version of a generic linear functional."""
        output = getattr(package, func_name)(
            self.input_quantizer(input),
            self.weight_quantizer(weight),
            *args,
            **kwargs,
        )
        return self.output_quantizer(output)


class QuantLinear(_LegacyQuantLinearConvBaseMixin, nn.Linear):
    """Quantized version of nn.Linear."""

    default_quant_desc_weight = _QuantLinear.default_quant_desc_weight


Linear = QuantLinear


class SVDQuantLinear(QuantLinearConvBase):
    """Base class for quantized linear modules with SVDQuant."""

    def _not_sequential_quantizers(self):
        return isinstance(self.weight_quantizer, TensorQuantizer) and isinstance(
            self.input_quantizer, TensorQuantizer
        )

    def _apply_pre_quant_scale(self, input: torch.Tensor):
        """Applies pre_quant_scale if present."""
        if self.input_quantizer.pre_quant_scale is not None and self._not_sequential_quantizers():
            input = input * self.input_quantizer.pre_quant_scale
        return input

    def _compute_lora_residual(self, input: torch.Tensor):
        """Compute the LoRA residual if present, otherwise return None."""
        if (
            self._not_sequential_quantizers()
            and self.weight_quantizer.svdquant_lora_a is not None
            and self.weight_quantizer.svdquant_lora_b is not None
        ):
            lora_a = F.linear(input, weight=self.weight_quantizer.svdquant_lora_a)
            lora_b = F.linear(lora_a, weight=self.weight_quantizer.svdquant_lora_b)
            return lora_b
        return None

    def forward(self, input, *args, **kwargs):
        """SVDQuant layer forward function."""
        has_svdquant_lora = (
            self._not_sequential_quantizers()
            and self.weight_quantizer.svdquant_lora_a is not None
            and self.weight_quantizer.svdquant_lora_b is not None
        )
        if has_svdquant_lora:
            input = self._apply_pre_quant_scale(input)
            _svdquant_lora_outputs = self._compute_lora_residual(input)
            with self.input_quantizer.disable_pre_quant_scale():
                output = super().forward(input, *args, **kwargs) + _svdquant_lora_outputs
        else:
            output = super().forward(input, *args, **kwargs)
        return output

    def _setup(self):
        """Overrides and bypass the _setup function."""
        pass

    def fold_weight(self):
        """Fold the weight for faster eval."""
        super().fold_weight()
        if (
            hasattr(self, "weight_quantizer")
            and hasattr(self, "weight")
            and self.weight_quantizer.fake_quant
        ):
            if (
                self._not_sequential_quantizers()
                and self.weight_quantizer.svdquant_lora_a is not None
                and self.weight_quantizer.svdquant_lora_b is not None
            ):
                self.weight.data.copy_(
                    self.weight
                    + self.weight_quantizer.svdquant_lora_b @ self.weight_quantizer.svdquant_lora_a
                )
            _attrs = [
                "_svdquant_lora_a",
                "_svdquant_lora_b",
            ]
            for attr in _attrs:
                if hasattr(self.weight_quantizer, attr):
                    delattr(self.weight_quantizer, attr)
