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
from _test_utils.examples.run_command import run_onnx_llm_export_command


@pytest.mark.parametrize(
    ("torch_dir", "dtype", "lm_head", "output_dir", "calib_size"),
    [
        ("Qwen/Qwen2-0.5B-Instruct", "fp16", "fp16", "/tmp/qwen2-0.5b-instruct-fp16", "1"),
        ("Qwen/Qwen2-0.5B-Instruct", "fp8", "fp16", "/tmp/qwen2-0.5b-instruct-fp8", "1"),
        ("Qwen/Qwen2-0.5B-Instruct", "int4_awq", "fp16", "/tmp/qwen2-0.5b-instruct-int4_awq", "1"),
        ("Qwen/Qwen2-0.5B-Instruct", "nvfp4", "fp16", "/tmp/qwen2-0.5b-instruct-nvfp4", "1"),
    ],
)
def test_llm_export_onnx(torch_dir, dtype, lm_head, output_dir, calib_size):
    run_onnx_llm_export_command(
        torch_dir=torch_dir,
        dtype=dtype,
        lm_head=lm_head,
        output_dir=output_dir,
        calib_size=calib_size,
    )
