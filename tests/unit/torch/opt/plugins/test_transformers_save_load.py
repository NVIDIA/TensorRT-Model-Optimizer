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

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from _test_utils.torch_model.transformers_models import (
    create_tiny_llama_dir,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM

import modelopt.torch.quantization as mtq


@pytest.mark.parametrize(
    "model_cls,fold_weight", [(LlamaForCausalLM, True), (AutoModelForCausalLM, False)]
)
def test_transformers_save_restore(tmp_path, model_cls, fold_weight):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    model_ref = model_cls.from_pretrained(tiny_llama_dir)
    mtq.quantize(model_ref, mtq.INT8_DEFAULT_CFG, lambda model: model(**model.dummy_inputs))
    if fold_weight:
        mtq.fold_weight(model_ref)
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)


@pytest.mark.parametrize("model_cls", [LlamaForCausalLM, AutoModelForCausalLM])
def test_transformers_load_with_multi_thread(tmp_path, model_cls):
    """Multi-threaded test for save/restore functionality"""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    workers = 2
    exceptions = []

    def worker_func(worker_id):
        try:
            _ = model_cls.from_pretrained(tiny_llama_dir)
        except Exception as e:
            traceback.print_exc()
            return e

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_func, i) for i in range(workers)]

        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                exceptions.append(result)

    assert len(exceptions) == 0, "Parallel model loading tests failed, check error log"
