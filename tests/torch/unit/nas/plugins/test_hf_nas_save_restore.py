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
    create_tiny_bert_dir,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering

import modelopt.torch.nas as mtn


def test_pruned_transformers_save_restore(tmp_path):
    tiny_bert_dir = create_tiny_bert_dir(tmp_path)
    model_ref = BertForQuestionAnswering.from_pretrained(tiny_bert_dir)

    # Export a random subnet (proxy for search / prune)
    model_ref = mtn.convert(model_ref, "fastnas")
    mtn.sample(model_ref)
    model_ref = mtn.export(model_ref)

    model_ref.save_pretrained(tiny_bert_dir / "modelopt_model")
    assert os.path.exists(tiny_bert_dir / "modelopt_model/modelopt_state.pth")

    # Restore pruned model + weights
    model_test = AutoModelForQuestionAnswering.from_pretrained(tiny_bert_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)
