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

"""Support quantization for Transformer Engine layers."""

import torch
import transformer_engine as te
import transformer_engine.pytorch.module.linear as te_linear

from ..nn import QuantModuleRegistry
from .custom import _ParallelLinear


@QuantModuleRegistry.register({te.pytorch.Linear: "te_Linear"})
class _QuantTELinear(_ParallelLinear):
    _functionals_to_replace = [
        (
            te_linear._Linear,
            "apply" if torch.is_grad_enabled() else "forward",
        ),
    ]

    @staticmethod
    def te_quantized_linear_fn(package, func_name, self, *args, **kwargs):
        """Quantized version specifically for TE with weight first, then input."""
        if te.__version__ >= "2.0":
            weight, inputs = args[0], args[1]
            remaining_args = args[2:]
            output = getattr(package, func_name)(
                self.weight_quantizer(weight),
                self.input_quantizer(inputs),
                *remaining_args,
                **kwargs,
            )
        else:
            weight, weight_fp8, inputs = args[0], args[1], args[2]
            remaining_args = args[3:]
            output = getattr(package, func_name)(
                self.weight_quantizer(weight),
                weight_fp8,
                self.input_quantizer(inputs),
                *remaining_args,
                **kwargs,
            )
        return self.output_quantizer(output)

    # Override the quantized linear function
    _quantized_linear_fn = te_quantized_linear_fn
