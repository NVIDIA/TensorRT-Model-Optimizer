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

import os

import pytest
import torch
from _test_utils.torch_model.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM

import modelopt.torch.speculative as mtsp


def test_medusa_model_convert_save_and_restore(tmp_path):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    model_ref = LlamaForCausalLM.from_pretrained(tiny_llama_dir)

    config = {
        "medusa_num_heads": 2,
        "medusa_num_layers": 1,
    }
    mtsp.convert(model_ref, mode=[("medusa", config)])
    assert isinstance(model_ref, mtsp.plugins.HFMedusaModel)

    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")
    assert os.path.exists(tiny_llama_dir / "modelopt_model/modelopt_state.pth")

    model_test = AutoModelForCausalLM.from_pretrained(tiny_llama_dir / "modelopt_model")
    assert isinstance(model_test, mtsp.plugins.HFMedusaModel)
    tf_modelopt_state_and_output_tester(model_ref, model_test)


def test_eagle_model_convert_save_and_restore(tmp_path):
    model_ref = get_tiny_llama(num_hidden_layers=8)

    config = {
        "eagle_num_layers": 1,
        "use_aux_hidden_state": True,
    }
    mtsp.convert(model_ref, mode=[("eagle", config)])
    assert isinstance(model_ref, mtsp.plugins.HFEagleModel)

    model_ref.save_pretrained(tmp_path / "modelopt_model")
    assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

    model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
    assert isinstance(model_test, mtsp.plugins.HFEagleModel)
    tf_modelopt_state_and_output_tester(model_ref, model_test)


# fmt: off
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_eagle_model_prepare_eagle_inputs(dtype):
    dummy_model = get_tiny_llama(num_hidden_layers=4)

    config = {
        "eagle_num_layers": 1,
        "use_aux_hidden_state": True,
    }
    mtsp.convert(dummy_model, mode=[("eagle", config)])

    eagle_input_ids_0 = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    position_ids_0 = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)


    #This is concatenated from 3 intermediate base model layers
    cat_aux_hidden_states = torch.randn(1, 4, 32, dtype=dtype)

    #This is eagle output from previous eagle forward pass
    dummy_eagle_output_hidden_states = torch.randn(1, 4, 32, dtype=dtype)

    #This is the causal mask for the 0th eagle step
    m = torch.finfo(dtype).min
    attention_mask_0 = torch.tensor([[0, m, m, m], #  input tok 10-> predicting token 20
                                     [0, 0, m, m], #  20 -> 30
                                     [0, 0, 0, m], #  30 -> 40
                                     [0, 0, 0, 0]] #  40 -> tok after 40

                                    , dtype=dtype).view(1, 1, 4, 4)

    # 2nd eagle step
    eagle_input_h_1, eagle_input_ids_1, attention_mask_1, position_ids_1 = dummy_model._concat_eagle_inputs(
        eagle_input_ids_0,
        cat_aux_hidden_states,
        attention_mask_0,
        position_ids_0,
        dummy_eagle_output_hidden_states,
    )

    assert eagle_input_ids_1.equal(torch.tensor([[10, 20, 30, 40, 10, 20, 30, 40]], dtype=torch.long))
    assert position_ids_1.equal(torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long))

    assert attention_mask_1.equal(torch.tensor([[0, m, m, m,  m, m, m, m], # (x) output discarded
                                                [0, 0, m, m,  m, m, m, m], # (x)
                                                [0, 0, 0, m,  m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m], # (x) input tok 10-> predicting token 20
                                                [0, m, m, m,  m, 0, m, m], # 20 -> 30
                                                [0, 0, m, m,  m, m, 0, m], # 30 -> 40
                                                [0, 0, 0, 0,  m, m, m, m], # (x) 40 -> tok after 40
                                                ], dtype=dtype).view(1, 1, 8, 8))

    # 3rd eagle step
    eagle_input_hidden_states_2, eagle_input_ids_2, attention_mask_2, position_ids_2 = dummy_model._concat_eagle_inputs(
        eagle_input_ids_0,
        cat_aux_hidden_states,
        attention_mask_0,
        position_ids_0,
        torch.cat([dummy_eagle_output_hidden_states, dummy_eagle_output_hidden_states], dim=1),
    )
    assert eagle_input_ids_2.equal(torch.tensor([[10, 20, 30, 40,  10, 20, 30, 40,  10, 20, 30, 40]], dtype=torch.long))
    assert position_ids_2.equal(torch.tensor([[0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3]], dtype=torch.long))

    assert attention_mask_2.equal(torch.tensor([[0, m, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, 0, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, m, m, m,  m, 0, m, m,  m, m, m, m], # (x)
                                                [0, 0, m, m,  m, m, 0, m,  m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)10 -> 20
                                                [m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)20 -> 30
                                                [0, m, m, m,  m, 0, m, m,  m, m, 0, m], # 30 -> 40
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m], # (x) 40 -> tok after 40

                                                ], dtype=dtype).view(1, 1, 12, 12))

    # 4th eagle step
    eagle_input_hidden_states_3, eagle_input_ids_3, attention_mask_3, position_ids_3 = dummy_model._concat_eagle_inputs(
        eagle_input_ids_0,
        cat_aux_hidden_states,
        attention_mask_0,
        position_ids_0,
        torch.cat([dummy_eagle_output_hidden_states, dummy_eagle_output_hidden_states,
                   dummy_eagle_output_hidden_states],dim=1),
    )

    assert eagle_input_ids_3.equal(torch.tensor([[10, 20, 30, 40,  10, 20, 30, 40,
                                                  10, 20, 30, 40,  10, 20, 30, 40]], dtype=torch.long))
    assert position_ids_3.equal(torch.tensor([[0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3]], dtype=torch.long))

    assert attention_mask_3.equal(torch.tensor([[0, m, m, m,  m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, m, m,  m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, 0, m,  m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m,  m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [0, m, m, m,  m, 0, m, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [0, 0, m, m,  m, m, 0, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [0, m, m, m,  m, 0, m, m,  m, m, 0, m,   m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)

                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)10 -> 20
                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)20 -> 30
                                                [m, m, m, m,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)
                                                [0, 0, 0, 0,  m, m, m, m,  m, m, m, m,   m, m, m, m], # (x)

                                                ], dtype=dtype).view(1, 1, 16, 16))
