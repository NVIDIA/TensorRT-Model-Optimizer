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

"""Tests for rotation preprocessing."""

import torch

# from _test_utils.torch_model.transformers_models import get_tiny_llama
from transformers import LlamaConfig, LlamaForCausalLM

from modelopt.torch.quantization.rotation import apply_rotation  # apply_per_head_rotation,


def test_overall_rotation():
    """Test overall rotation application."""
    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "head_dim": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    # kwargs.update(**config_kwargs)
    tiny_llama = LlamaForCausalLM(LlamaConfig(**kwargs))
    # tiny_llama = get_tiny_llama()
    config = "modelopt/torch/quantization/rotation/configs/test_r1.yaml"
    device = "cuda"
    print("Device:", device)
    print("Tiny Llama:", tiny_llama)

    tiny_llama.to(device=device)
    print("moved to cuda")
    input_ids = torch.randint(0, tiny_llama.config.vocab_size, (1, 10)).to(device)
    output = tiny_llama(input_ids)[0]
    apply_rotation(tiny_llama, config)
    output_after = tiny_llama(input_ids)[0]
    print(
        f"Output max diff: {torch.abs(output - output_after).max().item()}, max rel diff: {torch.abs(output - output_after).max().item() / output.max().item()}"
    )
    assert torch.allclose(output, output_after, atol=1e-2, rtol=1e-2)


test_overall_rotation()
