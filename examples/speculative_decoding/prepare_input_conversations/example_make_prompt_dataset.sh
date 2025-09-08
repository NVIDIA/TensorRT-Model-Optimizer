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

# Example script to prepare a dataset of prompts for generation
# Lines in this script can be uncommented to include specific datasets/splits in the prompt dataset.

python3 make_prompts_for_gen/add_sharegpt.py --output-split eval --output-file data/mtbench_prompts_dataset.json
# python3 make_prompts_for_gen/add_ultrachat.py --ultrachat-split train_sft --output-split train
# python3 make_prompts_for_gen/add_ultrachat.py --ultrachat-split train_gen --output-split train
# python3 make_prompts_for_gen/add_ultrachat.py --ultrachat-split test_sft --output-split mix_test
# python3 make_prompts_for_gen/add_ultrachat.py --ultrachat-split test_gen --output-split mix_test
python3 make_prompts_for_gen/add_mtbench.py --output-split train --output-file data/mtbench_prompts_dataset.json
# python3 make_prompts_for_gen/add_mtbench.py --output-split eval --output-file data/mtbench_prompts_dataset.json
