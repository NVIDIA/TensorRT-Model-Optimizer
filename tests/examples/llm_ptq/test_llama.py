# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.examples.run_command import run_llm_ptq_command

# TODO: Enable export Affine NVFP4 KV cache tests when supported by TRTLLM
# TODO: sparsegpt test is disable due to some bug when sparsifying models with Pytorch 2.4.0a0+3bcc3cddb5.nv24.7


# Use actual TinyLlama-1.1B for nightly tests
@pytest.fixture(scope="module")
def llama_path(tiny_llama_path):
    fast_tests = os.getenv("MODELOPT_FAST_TESTS", "true").lower() == "true"
    if fast_tests:
        return tiny_llama_path
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.parametrize(
    "quant,export_fmt,sparsity",
    [
        ("fp16", "tensorrt_llm", None),
        ("bf16", "tensorrt_llm", None),
        ("int8_sq", "tensorrt_llm", None),
        # ("int8_sq", "tensorrt_llm", "sparsegpt"),
        ("int4_awq", "tensorrt_llm", None),
        ("int4_awq", "hf", None),
        ("nvfp4", "tensorrt_llm", None),
        ("nvfp4", "hf", None),
        ("nvfp4_awq", "tensorrt_llm", None),
        ("nvfp4_awq", "hf", None),
    ],
)
def test_llama(llama_path, quant, export_fmt, sparsity):
    run_llm_ptq_command(model=llama_path, quant=quant, export_fmt=export_fmt, sparsity=sparsity)


@pytest.mark.parametrize(
    "quant,export_fmt",
    [
        ("int4_awq,nvfp4,fp8,w4a8_awq", "tensorrt_llm"),
        ("int4_awq,nvfp4,fp8", "hf"),
    ],
)
def test_llama_autoquant(llama_path, quant, export_fmt):
    run_llm_ptq_command(
        model=llama_path, quant=quant, export_fmt=export_fmt, calib_batch_size=4, effective_bits=6.4
    )


@pytest.mark.parametrize(
    "quant,export_fmt,kv_cache_quant",
    [
        ("nvfp4_awq", "tensorrt_llm", "nvfp4"),
        ("nvfp4_awq", "hf", "nvfp4"),
        # ("nvfp4_awq", "tensorrt_llm", "nvfp4_affine"),
        # ("nvfp4_awq", "hf", "nvfp4_affine"),
    ],
)
def test_llama_kv_cache(llama_path, quant, export_fmt, kv_cache_quant):
    run_llm_ptq_command(
        model=llama_path, quant=quant, export_fmt=export_fmt, kv_cache_quant=kv_cache_quant
    )


@pytest.mark.parametrize(
    "quant,export_fmt,kv_cache_quant",
    [
        ("int4_awq,nvfp4,fp8,w4a8_awq", "tensorrt_llm", "nvfp4"),
        ("int4_awq,nvfp4,fp8,w4a8_awq", "hf", "nvfp4"),
        # ("int4_awq,nvfp4,fp8,w4a8_awq", "tensorrt_llm", "nvfp4_affine"),
        # ("int4_awq,nvfp4,fp8,w4a8_awq", "hf", "nvfp4_affine"),
    ],
)
def test_llama_autoquant_kv_cache(llama_path, quant, export_fmt, kv_cache_quant):
    run_llm_ptq_command(
        model=llama_path,
        quant=quant,
        export_fmt=export_fmt,
        calib_batch_size=4,
        effective_bits=6.4,
        kv_cache_quant=kv_cache_quant,
    )


@pytest.mark.parametrize(
    "quant,export_fmt,sparsity,kv_cache_quant",
    [
        ("fp8", "tensorrt_llm", None, None),
        ("fp8", "tensorrt_llm", None, "none"),  # disable kv cache quantization
        # ("fp8", "tensorrt_llm", "sparsegpt", None),
        ("fp8", "hf", None, None),
        ("w4a8_awq", "tensorrt_llm", None, None),
    ],
)
def test_llama_sm89(require_sm89, llama_path, quant, export_fmt, sparsity, kv_cache_quant):
    run_llm_ptq_command(
        model=llama_path,
        quant=quant,
        export_fmt=export_fmt,
        sparsity=sparsity,
        kv_cache_quant=kv_cache_quant,
    )


@pytest.mark.parametrize(
    "quant,tasks,sparsity,tp,pp",
    [
        # TP
        ("fp16", "build", None, 2, 1),
        # ("fp16", "build", "sparsegpt", 1),
        ("nvfp4", "build", None, 2, 1),
        ("fp16", "benchmark", None, 2, 1),
        # ("fp16", "benchmark", "sparsegpt", 2, 1),
        # PP
        # ("nvfp4", "build", None, 1, 2),
        # ("fp16", "build", None, 1, 2),
        # ("fp16", "build", "sparsegpt", 1, 2),
    ],
)
def test_llama_multi_gpu(require_2_gpus, llama_path, quant, tasks, sparsity, tp, pp):
    run_llm_ptq_command(model=llama_path, quant=quant, tasks=tasks, sparsity=sparsity, tp=tp, pp=pp)
