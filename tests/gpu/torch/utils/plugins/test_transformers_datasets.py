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

import pytest
from transformers import AutoTokenizer

from modelopt.torch.utils.plugins import LanguageDataset


@pytest.mark.parametrize(
    ("model_name", "dataset_name", "subset", "split"),
    [
        (
            "nvidia/Llama-3.1-8B-Instruct-FP8",
            "HuggingFaceTB/smoltalk2",
            "SFT",
            "smoltalk_smollm3_everyday_conversations_no_think",
        ),
        ("nvidia/Llama-3.1-8B-Instruct-FP8", "nvidia/Daring-Anteater", None, "train"),
        (
            "nvidia/Llama-3.1-8B-Instruct-FP8",
            "nvidia/Nemotron-Pretraining-Dataset-sample",
            "Nemotron-CC-High-Quality",
            "train",
        ),
        (
            "Qwen/Qwen3-0.6B",
            "HuggingFaceTB/smoltalk2",
            "SFT",
            "smoltalk_smollm3_everyday_conversations_no_think",
        ),
        ("Qwen/Qwen3-0.6B", "HuggingFaceTB/smoltalk2", "SFT", "s1k_1.1_think"),
        ("Qwen/Qwen3-0.6B", "nvidia/Daring-Anteater", None, "train"),
        (
            "Qwen/Qwen3-0.6B",
            "nvidia/Nemotron-Pretraining-Dataset-sample",
            "Nemotron-CC-High-Quality",
            "train",
        ),
    ],
)
def test_transformers_language_dataset(model_name, dataset_name, subset, split):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = LanguageDataset(
        tokenizer,
        dataset_name,
        subset,
        split=split,
    )

    dataset[0]
