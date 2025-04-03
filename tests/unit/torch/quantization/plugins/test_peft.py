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
from _test_utils.torch_model.transformers_models import get_tiny_llama, tf_output_tester

pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear

import modelopt.torch.quantization as mtq


def test_convert_loralinear():
    lora_config = LoraConfig(
        r=1,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_ref = get_tiny_llama()
    model_ref = get_peft_model(model_ref, lora_config)
    model_test = get_tiny_llama()
    model_test = get_peft_model(model_test, lora_config)
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for _, module in model_test.named_modules():
        if isinstance(module, LoraLinear):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    tf_output_tester(model_ref, model_test)
