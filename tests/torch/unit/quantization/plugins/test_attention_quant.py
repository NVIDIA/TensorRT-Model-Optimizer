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

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.attention import register_attention_for_kv_quant

transformers = pytest.importorskip("transformers")


class MatmulAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        return torch.matmul(a, v)


class BMMAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        a = torch.softmax(torch.bmm(q, k.transpose(-2, -1)), dim=-1)
        return torch.bmm(a, v)


class BinMatmulAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v


class SDPAAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        return F.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize(
    "attn_cls", [MatmulAttention, BMMAttention, BinMatmulAttention, SDPAAttention]
)
def test_convert_conv1d(attn_cls):
    register_attention_for_kv_quant(attn_cls)

    model_test = nn.Sequential(
        attn_cls(),
    )

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(module, attn_cls):
            assert hasattr(module, "k_bmm_quantizer")
            assert hasattr(module, "v_bmm_quantizer")

    model_test((q, k, v))

    mtq.unregister(attn_cls)
