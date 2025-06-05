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

"""Support quantization for peft LoRA linear layers."""

import torch.nn.functional as F
from peft.tuners.lora.layer import Linear as LoraLinear

from ..nn import QuantModule, QuantModuleRegistry, TensorQuantizer

__all__ = []


@QuantModuleRegistry.register({LoraLinear: "LoraLinear"})
class _QuantLoraLinear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = TensorQuantizer()
        self.output_quantizer = TensorQuantizer()

    def forward(self, x, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters or adapter_names is not None or self.merged:
            return super().forward(x, args, kwargs)

        weight = self.base_layer.weight
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():  # noqa: SIM118
                continue
            lora_a = self.lora_A[active_adapter]
            lora_b = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]

            if not self.use_dora[active_adapter]:
                weight = weight + scaling * lora_b.weight @ lora_a.weight
            else:
                raise NotImplementedError("dora not implemented")

        x = self.input_quantizer(x)
        weight = self.weight_quantizer(weight)
        output = self.output_quantizer(F.linear(x, weight, self.base_layer.bias))
        return output
