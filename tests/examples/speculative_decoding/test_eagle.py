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

import json

from _test_utils.examples.run_command import run_example_command


# fmt: off
def test_llama_eagle3(tiny_llama_path, num_gpus, tiny_daring_anteater_path, tmp_path):
    # Create an ultra-tiny EAGLE config for testing to reduce memory usage
    tiny_eagle_config = {
        "max_position_embeddings": 128,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
    }

    # Write the tiny config to a temporary file
    config_file = tmp_path / "tiny_eagle_config.json"
    with open(config_file, "w") as f:
        json.dump(tiny_eagle_config, f)

    run_example_command(
        [
            "./launch_train.sh",
            "--model", tiny_llama_path,
            "--data", tiny_daring_anteater_path,
            "--num_epochs", "1",
            "--lr", "1e-5",
            "--num_gpu", str(num_gpus),
            "--mode", "eagle3",
            "--eagle_config", str(config_file),
            "--output_dir", tmp_path / "eagle-tinyllama",
            "--training_seq_len", "128", # Match max_position_embeddings
        ],
        "speculative_decoding",
    )
