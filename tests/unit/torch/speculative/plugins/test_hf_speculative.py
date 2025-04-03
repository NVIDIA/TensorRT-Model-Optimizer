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

from _test_utils.torch_model.transformers_models import (
    create_tiny_llama_dir,
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
