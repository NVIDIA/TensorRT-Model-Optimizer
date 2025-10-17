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
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
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


# Register the public te.pytorch.GroupedLinear class
@QuantModuleRegistry.register({te_grouped_linear.GroupedLinear: "te_GroupedLinear"})
class _QuantTEGroupedLinear(_ParallelLinear):
    _functionals_to_replace = [
        (te_grouped_linear._GroupedLinear, "forward"),
        (te_grouped_linear._GroupedLinear, "apply"),
    ]

    def _setup(self):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run setup in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        # Memorize the original weight.dtype for modelopt_post_restore given that
        # the dtype can change later.
        super()._setup()
        # Remove self.weight after setup.
        delattr(self, "weight")

    def modelopt_post_restore(self, prefix: str = ""):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run post_restore in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        super().modelopt_post_restore(prefix=prefix)
        # Remove self.weight after post_restore.
        delattr(self, "weight")

    @staticmethod
    def te_grouped_quantized_linear_fn(package, func_name, self, *args):
        idx = 1 if func_name == "_forward" else 0
        inp = args[idx]
        num_gemms = len(args[idx + 1])
        weights_and_biases = args[-2 * num_gemms :]
        weights, biases = weights_and_biases[:num_gemms], weights_and_biases[num_gemms:]
        quantized_inputs = self.input_quantizer(inp)
        quantized_weights = [self.weight_quantizer(weight) for weight in weights]

        output = getattr(package, func_name)(
            *(
                args[0],
                quantized_inputs,
            )
            if func_name == "_forward"
            else (quantized_inputs,),
            *args[idx + 1 : -2 * num_gemms],
            *quantized_weights,
            *biases,
        )
        return self.output_quantizer(output)

    # Override the quantized linear function
    _quantized_linear_fn = te_grouped_quantized_linear_fn
