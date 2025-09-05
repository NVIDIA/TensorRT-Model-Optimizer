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
from _test_utils.model import BART_PATH, MIXTRAL_PATH, T5_PATH, TINY_LLAMA_PATH, WHISPER_PATH
from _test_utils.ptq_utils import PTQCommand, WithRequirements


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp16"),
        PTQCommand(quant="fp8", min_sm=89),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_bart(command):
    command.run(BART_PATH)


class TestT5(WithRequirements):
    requirements = [("transformers", "4.48.0")]

    @pytest.mark.parametrize(
        "command",
        [
            PTQCommand(quant="fp16"),
            PTQCommand(quant="fp8", min_sm=89),
        ],
        ids=PTQCommand.param_str,
    )
    def test_ptq_t5(self, command):
        command.run(T5_PATH)


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp16"),
        PTQCommand(quant="fp8", min_sm=89),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_mixtral(command):
    command.run(MIXTRAL_PATH)


class TestWhisper(WithRequirements):
    requirements = [
        ("librosa", None),
        ("soundfile", None),
    ]

    @pytest.mark.parametrize(
        "command",
        [
            # Auto-batch-size computation seems to take >10mins for Whisper hence using a fixed batch size
            PTQCommand(quant="fp16", calib_batch_size=16),
            PTQCommand(quant="fp8", calib_batch_size=16, min_sm=89),
        ],
        ids=PTQCommand.param_str,
    )
    def test_ptq_whisper(self, command):
        command.run(WHISPER_PATH)


@pytest.fixture(scope="module")
def llama_path(tiny_llama_path):
    fast_tests = os.getenv("MODELOPT_FAST_TESTS", "true").lower() == "true"
    if fast_tests:
        return tiny_llama_path
    return TINY_LLAMA_PATH


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp16"),
        PTQCommand(quant="bf16"),
        PTQCommand(quant="int8_sq"),
        # ("int8_sq", "tensorrt_llm", "sparsegpt"),
        PTQCommand(quant="int4_awq"),
        PTQCommand(quant="nvfp4"),
        PTQCommand(quant="nvfp4_awq"),
        #
        # autoquant
        PTQCommand(
            quant="int4_awq,nvfp4,fp8,w4a8_awq",
            calib_batch_size=4,
            auto_quantize_bits=6.4,
        ),
        #
        # kv_cache
        PTQCommand(quant="nvfp4_awq", kv_cache_quant="nvfp4"),
        # ("nvfp4_awq", "tensorrt_llm", "nvfp4_affine"),
        # ("nvfp4_awq", "hf", "nvfp4_affine"),
        #
        # autoquant_kv_cache
        PTQCommand(
            quant="int4_awq,nvfp4,fp8,w4a8_awq",
            kv_cache_quant="nvfp4",
            calib_batch_size=4,
            auto_quantize_bits=6.4,
        ),
        # ("int4_awq,nvfp4,fp8,w4a8_awq", "tensorrt_llm", "nvfp4_affine"),
        # ("int4_awq,nvfp4,fp8,w4a8_awq", "hf", "nvfp4_affine"),
        #
        # sm89
        PTQCommand(quant="fp8", min_sm=89),
        PTQCommand(quant="fp8", kv_cache_quant="none", min_sm=89),
        # ("fp8", "tensorrt_llm", "sparsegpt", None),
        PTQCommand(quant="w4a8_awq", min_sm=89),
        #
        # multi_gpu
        # TP
        PTQCommand(quant="fp16", tp=2, pp=1, min_gpu=2),
        # ("fp16", "build", "sparsegpt", 1),
        PTQCommand(quant="nvfp4", tp=2, pp=1, min_gpu=2),
        PTQCommand(quant="fp16", tasks="benchmark", tp=2, pp=1, min_gpu=2),
        # ("fp16", "benchmark", "sparsegpt", 2, 1),
        # PP
        # ("nvfp4", "build", None, 1, 2),
        # ("fp16", "build", None, 1, 2),
        # ("fp16", "build", "sparsegpt", 1, 2),
    ],
    ids=PTQCommand.param_str,
)
def test_ptq_llama(command, llama_path):
    command.run(llama_path)
