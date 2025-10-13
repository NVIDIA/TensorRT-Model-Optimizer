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

_LOCAL_ROOT = os.getenv("MODELOPT_LOCAL_MODEL_ROOT")


def _select_path(remote_id: str, local_id: str) -> str:
    if _LOCAL_ROOT:
        return f"{_LOCAL_ROOT}/{local_id}"
    return remote_id


BART_PATH = _select_path(
    remote_id="facebook/bart-large-cnn",
    local_id="bart-large-cnn",
)

T5_PATH = _select_path(
    remote_id="google-t5/t5-small",
    local_id="t5-small",
)

MIXTRAL_PATH = _select_path(
    remote_id="AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE",
    local_id="TinyLlama-1.1B-Chat-v1.0-x2-MoE",
)

WHISPER_PATH = _select_path(
    remote_id="openai/whisper-tiny",
    local_id="whisper-tiny",
)

TINY_LLAMA_PATH = _select_path(
    remote_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_id="TinyLlama-1.1B-Chat-v1.0",
)

SXDL_PATH = _select_path(
    remote_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_id="stable-diffusion-xl-base-1.0",
)

PIXART_PATH = _select_path(
    remote_id="PixArt-alpha/PixArt-XL-2-1024-MS",
    local_id="PixArt-XL-2-1024-MS",
)

LLAVA_PATH = _select_path(
    remote_id="llava-hf/llava-1.5-7b-hf",
    local_id="llava-1.5-7b-hf",
)

QWEN_VL_PATH = _select_path(
    remote_id="Qwen/Qwen2-VL-2B-Instruct",
    local_id="Qwen2-VL-2B-Instruct",
)

LLAMA30_8B_INST_PATH = _select_path(
    remote_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_id="Meta-Llama-3-8B-Instruct",
)

LLAMA31_8B_PATH = _select_path(
    remote_id="meta-llama/Llama-3.1-8B",
    local_id="Llama-3.1-8B",
)

LLAMA32_3B_PATH = _select_path(
    remote_id="meta-llama/Llama-3.2-3B",
    local_id="Llama-3.2-3B",
)

QWEN2_7B_INST_PATH = _select_path(
    remote_id="Qwen/Qwen2-7B-Instruct",
    local_id="Qwen2-7B-Instruct",
)

QWEN25_0_5B_INST_PATH = _select_path(
    remote_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_id="Qwen2.5-0.5B-instruct",
)

QWEN25_3B_INST_PATH = _select_path(
    remote_id="Qwen/Qwen2.5-3B-Instruct",
    local_id="Qwen2.5-3B-Instruct",
)

QWEN25_7B_INST_PATH = _select_path(
    remote_id="Qwen/Qwen2.5-7B-Instruct",
    local_id="Qwen2.5-7B-Instruct",
)

# Diffusers
FLUX_SCHNELL_PATH = _select_path(
    remote_id="hf-internal-testing/tiny-flux-pipe",
    local_id="black-forest-labs/FLUX.1-schnell",
)

SDXL_1_0_PATH = _select_path(
    remote_id="hf-internal-testing/tiny-sdxl-pipe",
    local_id="stabilityai/stable-diffusion-xl-base-1.0",
)

SD3_PATH = _select_path(
    remote_id="hf-internal-testing/tiny-sd3-pipe",
    local_id="stabilityai/stable-diffusion-3-medium-diffusers",
)
