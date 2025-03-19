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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch_model.transformers_models import get_tiny_llama
from packaging.version import Version

import modelopt.torch.quantization as mtq

transformers = pytest.importorskip("transformers")


class MatmulAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        return torch.matmul(a, v)

    def get_input(self):
        return torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)


class BMMAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        a = torch.softmax(torch.bmm(q, k.transpose(-2, -1)), dim=-1)
        return torch.bmm(a, v)

    def get_input(self):
        return torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)


class BinMatmulAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v

    def get_input(self):
        return torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)


class SDPAAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        return F.scaled_dot_product_attention(q, k, v)

    def get_input(self):
        return torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)


@pytest.mark.skipif(
    Version(transformers.__version__) >= Version("4.48.0"),
    reason="Legacy K/V cache quantization requires transformers < 4.48.0",
)
@pytest.mark.parametrize(
    "attn_cls", [MatmulAttention, BMMAttention, BinMatmulAttention, SDPAAttention]
)
def test_kv_quant(attn_cls):
    model_test = get_tiny_llama()

    mtq.plugins.register_attention_for_kv_quant(attn_cls)
    mtq.replace_quant_module(model_test)
    for module in model_test.modules():
        if isinstance(module, attn_cls):
            assert hasattr(module, "k_bmm_quantizer")
            assert hasattr(module, "v_bmm_quantizer")

    input_ids = torch.randint(0, 1, (1, 4))
    model_test(input_ids)

    mtq.unregister(attn_cls)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.48.0"),
    reason="K/V cache quantization for attention_interface requires transformers >= 4.48.0",
)
def test_kv_quant_hf():
    model_test = get_tiny_llama()
    attn_cls = model_test.model.layers[0].self_attn.__class__

    mtq.replace_quant_module(model_test)
    for module in model_test.modules():
        if isinstance(module, attn_cls):
            assert hasattr(module, "k_bmm_quantizer")
            assert hasattr(module, "v_bmm_quantizer")

    input_ids = torch.randint(0, 1, (1, 4))
    model_test(input_ids)
