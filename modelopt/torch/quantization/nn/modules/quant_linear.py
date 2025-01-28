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

import torch.nn as nn

from ... import tensor_quant
from .quant_module import QuantLinearConvBase, QuantModuleRegistry, _LegacyQuantLinearConvBaseMixin

__all__ = ["Linear", "QuantLinear"]


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
