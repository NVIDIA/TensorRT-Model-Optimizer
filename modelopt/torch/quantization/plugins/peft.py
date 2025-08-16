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

from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper

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
            return super().forward(x, *args, **kwargs)

        x = self.input_quantizer(x)
        weight = self.base_layer.weight
        is_compressed = isinstance(weight, QTensorWrapper)
        if not is_compressed:
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
            weight = self.weight_quantizer(weight)
            output = F.linear(x, weight, self.base_layer.bias)
        else:
            # For compressed weights, compute base output and LoRA outputs separately
            base_output = self.base_layer(x)

            # Only compute LoRA outputs if there are active adapters
            if self.active_adapters:
                # Start with zero LoRA output
                lora_output = None

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():  # noqa: SIM118
                        continue
                    lora_a = self.lora_A[active_adapter]
                    lora_b = self.lora_B[active_adapter]
                    scaling = self.scaling[active_adapter]

                    if not self.use_dora[active_adapter]:
                        # Compute LoRA output step by step to maintain gradient flow
                        lora_a_output = F.linear(x, lora_a.weight)
                        lora_b_output = F.linear(lora_a_output, lora_b.weight)
                        adapter_output = scaling * lora_b_output

                        lora_output = (
                            adapter_output if lora_output is None else lora_output + adapter_output
                        )

                output = base_output + lora_output
            else:
                output = base_output

        output = self.output_quantizer(output)
        return output
