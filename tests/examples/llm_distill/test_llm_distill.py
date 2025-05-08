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


from _test_utils.examples.run_command import run_example_command


# fmt: off
def test_llama_distill(tiny_llama_path, tmp_path):
    run_example_command(
        [
            "accelerate", "launch", "--multi_gpu", "--mixed_precision", "bf16", "main.py",
            "--teacher_name_or_path", tiny_llama_path,
            "--student_name_or_path", tiny_llama_path,
            "--output_dir", tmp_path,
            "--logging_steps", "5",
            "--max_steps", "10",
            "--max_seq_length", "1024",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "8",
            "--gradient_checkpointing", "True",
            "--fsdp", "full_shard auto_wrap",
            "--fsdp_transformer_layer_cls_to_wrap", "LlamaDecoderLayer",
        ],
        "llm_distill",
    )
