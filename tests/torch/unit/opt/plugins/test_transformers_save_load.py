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

transformers = pytest.importorskip("transformers")

from _transformers_helper import create_base_model
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM


@pytest.mark.parametrize("model_cls", [LlamaForCausalLM, AutoModelForCausalLM])
def test_transformers_save_restore(tmpdir, model_cls):
    mto.enable_huggingface_checkpointing()

    model_ref = model_cls.from_pretrained(create_base_model(tmpdir))
    mtq.quantize(model_ref, mtq.INT8_DEFAULT_CFG, lambda model: model(**model.dummy_inputs))
    model_ref.save_pretrained(tmpdir + "/modelopt_model")

    model_test = model_cls.from_pretrained(tmpdir + "/modelopt_model")

    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for name, module in model_test.named_modules():
        if hasattr(module, "_is_hf_initialized"):
            delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)

    assert model_ref_state == model_test_state

    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs).logits
    output_test = model_test(**inputs).logits
    assert torch.allclose(output_ref, output_test)


def test_transformers_save_restore_fold_weight(tmpdir):
    mto.enable_huggingface_checkpointing()

    model_ref = AutoModelForCausalLM.from_pretrained(create_base_model(tmpdir))
    mtq.quantize(model_ref, mtq.INT8_DEFAULT_CFG, lambda model: model(**model.dummy_inputs))
    mtq.fold_weight(model_ref)
    model_ref.save_pretrained(tmpdir + "/modelopt_model")

    model_test = AutoModelForCausalLM.from_pretrained(tmpdir + "/modelopt_model")

    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for name, module in model_test.named_modules():
        if hasattr(module, "_is_hf_initialized"):
            delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)

    assert model_ref_state == model_test_state

    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs).logits
    output_test = model_test(**inputs).logits
    assert torch.allclose(output_ref, output_test)
