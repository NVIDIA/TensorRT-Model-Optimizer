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

import pytest
import torch

import modelopt.torch.quantization as mtq

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers.models.llama.configuration_llama import LlamaConfig  # noqa
from transformers.models.llama.modeling_llama import LlamaForCausalLM  # noqa
from peft import LoraConfig, get_peft_model  # noqa
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa


def test_convert_loralinear():
    def create_base_model():
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
            )
        )
        return model

    lora_config = LoraConfig(
        r=1,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_ref = create_base_model()
    model_ref = get_peft_model(model_ref, lora_config)
    model_test = create_base_model()
    model_test = get_peft_model(model_test, lora_config)
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for _, module in model_test.named_modules():
        if isinstance(module, LoraLinear):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = torch.randint(0, 128, (2, 3))
    model_ref.eval()
    model_test.eval()
    out_1 = model_ref(input_ids=x).logits
    out_2 = model_test(input_ids=x).logits

    assert torch.allclose(out_1, out_2)
