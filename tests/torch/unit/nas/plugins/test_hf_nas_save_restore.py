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

import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto

tf = pytest.importorskip("transformers")


def test_pruned_transformers_save_restore(tmpdir):
    def create_base_model() -> str:
        base_model_path = tmpdir + "/base_model"
        model = tf.AutoModelForQuestionAnswering.from_config(
            tf.BertConfig(
                vocab_size=64,
                hidden_size=8,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=16,
                max_position_embeddings=32,
            )
        )
        model.save_pretrained(base_model_path)
        return base_model_path

    mto.enable_huggingface_checkpointing()

    model_ref = tf.BertForQuestionAnswering.from_pretrained(create_base_model())

    # Export a random subnet (proxy for search / prune)
    model_ref = mtn.convert(model_ref, "fastnas")
    mtn.sample(model_ref)
    model_ref = mtn.export(model_ref)

    model_ref.save_pretrained(tmpdir + "/modelopt_model")
    assert os.path.exists(tmpdir + "/modelopt_model/modelopt_state.pth")

    # Restore pruned model + weights
    model_test = tf.BertForQuestionAnswering.from_pretrained(tmpdir + "/modelopt_model")

    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for module in model_test.modules():
        if hasattr(module, "_is_hf_initialized"):
            delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs)
    output_test = model_test(**inputs)
    assert torch.allclose(output_ref.start_logits, output_test.start_logits)
    assert torch.allclose(output_ref.end_logits, output_test.end_logits)
