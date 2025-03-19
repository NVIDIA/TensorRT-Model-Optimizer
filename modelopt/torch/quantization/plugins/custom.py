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

"""Custom plugin base modules and utilities for quantization."""

from functools import partial
from types import ModuleType
from typing import Callable, Iterator

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils.distributed import ParallelState

from ..nn import TensorQuantizer
from ..nn.modules.quant_linear import _QuantLinear
from ..utils import multi_context, replace_function

try:
    from .huggingface import (
        register_dbrx_moe_on_the_fly,
        register_falcon_linears_on_the_fly,
        register_hf_attentions_on_the_fly,
    )
except ImportError:

    def _dummy_register(model):
        pass

    register_falcon_linears_on_the_fly = _dummy_register
    register_dbrx_moe_on_the_fly = _dummy_register
    register_hf_attentions_on_the_fly = _dummy_register


# TODO: This is a temporary solution
# In future implement a decorator to register methods updating QUANT_MODULE on the fly
def register_custom_model_plugins_on_the_fly(model):
    """Registers custom modules as QUANT_MODULE on the fly."""
    register_falcon_linears_on_the_fly(model)
    register_dbrx_moe_on_the_fly(model)
    register_hf_attentions_on_the_fly(model)


class _QuantFunctionalMixin(DynamicModule):
    """Mixin class for quantized functionals.

    Often we need to replace a functional with a quantized version. This class provides a way to do that.
    """

    # List of functionals to replace with quantized versions, e.g. [(package, func_name, quantized_func), ...]
    _functionals_to_replace: list[tuple[ModuleType, str, Callable]] = []

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        return (
            (package, func_name, quantized_func)
            for package, func_name, quantized_func in self._functionals_to_replace
            if hasattr(package, func_name)
        )

    def forward(self, *args, **kwargs):
        with multi_context(
            *(
                replace_function(package, func_name, quantized_func)
                for package, func_name, quantized_func in self.functionals_to_replace
            )
        ):
            return super().forward(*args, **kwargs)


class _ParallelLinear(_QuantFunctionalMixin):
    """Quantized base class for ParallelLinear type classes.

    For this type of modules, we need to quantize the inputs and weights just before calling the actual linear
    functional. This is accomplished by replacing the linear functional with a custom one that quantizes the inputs
    and weights before calling the original functional.
    """

    # List of functionals to replace [(package, func_name), ...]
    _functionals_to_replace: list[tuple[ModuleType, str]] = []

    _parallel_state: ParallelState
    _is_column_parallel = False
    _is_row_parallel = False
    _quantized_linear_fn: Callable = _QuantLinear.quantized_linear_fn

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        for package, func_name in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(
                self.__class__._quantized_linear_fn,
                package,
                "_" + func_name,
                self,
            )
            if hasattr(getattr(package, func_name), "__dict__"):
                quantized_func.__dict__.update(getattr(package, func_name).__dict__)
            yield package, func_name, quantized_func

    def initialize_parallel_state(self):
        self._parallel_state = ParallelState()

    def _setup(self):
        self.initialize_parallel_state()
        self.input_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_output)
        self.output_quantizer.disable()
