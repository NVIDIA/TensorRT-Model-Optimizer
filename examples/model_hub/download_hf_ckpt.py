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

from huggingface_hub import login, snapshot_download

login(token="<YOUR-HF-TOKEN>", add_to_git_credential=True)
model_names = [
    "Llama-3.1-8B-Instruct-FP8",
    "Llama-3.1-70B-Instruct-FP8",
    "Llama-3.1-405B-Instruct-FP8",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "quantized_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)
