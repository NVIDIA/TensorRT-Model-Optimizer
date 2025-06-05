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

"""Customization for Nemo Megatron GPT."""

from collections.abc import Callable, Iterator
from functools import partial
from types import ModuleType

import torch

# New nemo version should depend on megatron.core
from nemo.collections.nlp.modules.common.megatron.attention import CoreAttention

from ..nn import QuantInputBase, QuantModuleRegistry, TensorQuantizer
from .custom import _QuantFunctionalMixin

__all__ = []


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    """Quantized version of BMM2 in nemo CoreAttention."""
    attn, v = input, mat2
    return torch._bmm(attn, self.v_bmm_quantizer(v), *args, **kwargs)


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    """Quantized version of BMM1 in nemo CoreAttention."""
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


class _QuantCoreAttention(_QuantFunctionalMixin):
    """Quantized base class for CoreAttention."""

    _functionals_to_replace = [
        (torch, "bmm", _quantized_bmm),
        (torch, "baddbmm", _quantized_baddbmm),
    ]

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        for package, func_name, quantized_func in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(quantized_func, self)
            yield package, func_name, quantized_func

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)


QuantModuleRegistry.register({CoreAttention: "nemo_core_attention"})(_QuantCoreAttention)
