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

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

transformers = pytest.importorskip("transformers")

mto.enable_huggingface_checkpointing()


def test_medusa_model_convert_save_and_restore(tmpdir):
    llama_config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=4,
    )
    model = transformers.LlamaForCausalLM(llama_config)

    config = {
        "medusa_num_heads": 2,
        "medusa_num_layers": 1,
    }
    mtsp.convert(model, mode=[("medusa", config)])
    assert isinstance(model, mtsp.plugins.HFMedusaModel)

    model.save_pretrained(tmpdir + "/modelopt_model")
    assert os.path.exists(tmpdir + "/modelopt_model/modelopt_state.pth")

    res_model = transformers.AutoModelForCausalLM.from_pretrained(tmpdir + "/modelopt_model")
    assert isinstance(res_model, mtsp.plugins.HFMedusaModel)

    modelopt_state = mto.modelopt_state(model)
    res_modelopt_state = mto.modelopt_state(res_model)
    assert modelopt_state == res_modelopt_state
