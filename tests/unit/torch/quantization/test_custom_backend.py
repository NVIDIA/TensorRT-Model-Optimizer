# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for registering and using a custom quantization backend via model quantize().

This test uses a dummy backend and does NOT rely on PSX formats. It quantizes a test
model with a custom config that enables only the output quantizer to ensure the backend
is invoked and the model output is shifted by a known offset.
"""

import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import register_quant_backend, unregister_quant_backend


def test_custom_backend_via_quantize():
    # Define and register a simple dummy backend that adds a constant to inputs
    def dummy_backend(inputs: torch.Tensor, tq) -> torch.Tensor:
        extra = getattr(tq, "backend_extra_args", None) or {}
        offset = extra.get("offset", 1.0)
        return inputs + offset

    register_quant_backend("dummy_backend", dummy_backend)

    model = torch.nn.Linear(16, 16, bias=False)

    cfg = {
        "quant_cfg": {
            "*weight_quantizer": {
                "enable": True,
                "num_bits": 8,
                "axis": None,
                "backend": "dummy_backend",
                "backend_extra_args": {"offset": 2.5},
            },
            "default": {"enable": False},
        },
        "algorithm": "max",
    }

    inputs = torch.randn(1, 16)

    def forward_loop(m):
        m(inputs)

    mtq.quantize(model, cfg, forward_loop=forward_loop)
    output_test = model(inputs)

    assert torch.allclose(output_test, inputs @ (model.weight.T + 2.5))

    # Unregister the backend to avoid impacting other tests
    unregister_quant_backend("dummy_backend")
