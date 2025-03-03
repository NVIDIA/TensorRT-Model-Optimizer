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

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.conversion import (
    set_quantizer_by_cfg,
    set_quantizer_by_cfg_context,
)
from modelopt.torch.quantization.nn.functional import normalized_hadamard_transform
from modelopt.torch.quantization.plugins.attention import register_attention_for_kv_quant


class SDPAAttention(nn.Module):
    def forward(self, x):
        q, k, v = x
        return nn.functional.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize(
    "dim",
    [8, 16],
)
def test_hadamard_transform(dim):
    x = torch.rand(4, dim).cuda()
    xxt = x @ x.T
    x_h = normalized_hadamard_transform(x)
    xxt_h = x_h @ x_h.T
    assert torch.allclose(xxt_h, xxt, atol=1e-5)


def test_kv_rotate():
    register_attention_for_kv_quant(SDPAAttention)
    model = nn.Sequential(
        SDPAAttention(),
    )
    mtq.replace_quant_module(model)

    q = torch.randn(1, 4, 8).cuda()
    k = torch.randn(1, 4, 8).cuda()
    v = torch.randn(1, 4, 8).cuda()
    set_quantizer_by_cfg(model, {"*": {"enable": False}})
    output_ref = model((q, k, v))
    with set_quantizer_by_cfg_context(
        model,
        {
            "*[qk]_bmm_quantizer": {
                "rotate": True,
            },
        },
    ):
        output_test = model((q, k, v))
    assert torch.allclose(output_ref, output_test, atol=1e-5)

    set_quantizer_by_cfg(
        model,
        {
            "*q_bmm_quantizer": {
                "enable": False,
                "rotate": False,
            },
            "*k_bmm_quantizer": {
                "num_bits": 4,
                "axis": -1,
                "enable": True,
                "rotate": False,
            },
        },
    )
    output_ref1 = model((q, k, v))
    set_quantizer_by_cfg(
        model,
        {
            "*[qk]_bmm_quantizer": {
                "rotate": True,
            },
        },
    )
    output_test1 = model((q, k, v))
    torch.not_equal(output_ref1, output_test1)

    mtq.unregister(SDPAAttention)
