#!/bin/bash
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

set -e
set -o pipefail

LLAMA_PATH=$1

if [ -z "$LLAMA_PATH" ]; then
    echo "Usage: $0 <path_to_save_tiny_llama>"
    exit 1
fi

if [ -d "$LLAMA_PATH" ]; then
    echo "TinyLlama already exists at $LLAMA_PATH"
    exit 0
fi

python -c "\
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=128,
    )
).to(torch.bfloat16)
model.save_pretrained('$LLAMA_PATH')
tokenizer.save_pretrained('$LLAMA_PATH')
num_params = sum(p.numel() for p in model.parameters())
print(f'TinyLlama with {num_params/1e6:.1f}M parameters saved to $LLAMA_PATH')
"
