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

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from _transformers_helper import create_base_model
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


@pytest.mark.skip(reason="under investigation")
def test_peft_save_restore(tmpdir):
    mto.enable_huggingface_checkpointing()

    model_ref = AutoModelForCausalLM.from_pretrained(create_base_model(tmpdir))

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj"],
    )
    model_ref = get_peft_model(model_ref, peft_config)
    mtq.quantize(model_ref, mtq.INT8_DEFAULT_CFG, lambda model: model(**model.dummy_inputs))

    model_ref.save_pretrained(tmpdir + "/modelopt_peft_model")

    model_test = AutoPeftModelForCausalLM.from_pretrained(tmpdir + "/modelopt_peft_model")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)

    assert model_ref_state == model_test_state

    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs).logits
    output_test = model_test(**inputs).logits
    assert torch.allclose(output_ref, output_test)
