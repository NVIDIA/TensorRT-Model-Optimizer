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

import subprocess

from _test_utils.examples.run_command import run_llm_ptq_command


def test_llama_eval_fp8(require_sm89, tiny_llama_path):
    try:
        run_llm_ptq_command(
            model=tiny_llama_path,
            quant="fp8",
            tasks="mmlu,lm_eval,simple_eval,benchmark",
            calib=64,
            lm_eval_tasks="hellaswag,gsm8k",
            simple_eval_tasks="humaneval",
            lm_eval_limit=0.1,
        )
    finally:
        # Force kill llm-serve if it's still running
        subprocess.run(["pkill", "-f", "llm-serve"], check=False)
