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
def test_llama_eagle(tiny_llama_path, num_gpus, daring_anteater_path, tmp_path):
    run_example_command(
        [
            "./launch.sh",
            "--model", tiny_llama_path,
            "--data", daring_anteater_path,
            "--num_epochs", "0.005",
            "--lr", "1e-5",
            "--save_steps", "50",
            "--do_eval", "False",
            "--num_gpu", str(num_gpus),
            "--mode", "eagle",
            "--output_dir", tmp_path / "eagle-tinyllama",
            "--eagle_num_layers", "1",
        ],
        "speculative_decoding",
    )
