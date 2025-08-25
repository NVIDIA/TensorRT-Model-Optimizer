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
import torch
from _test_utils.torch_quantization.models import OneLayerLinear, SimpleLinear
from _test_utils.torch_quantization.quantize_common import compute_backward_grad

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.fp8_per_tensor_gemm import Fp8PerTensorLinear
from modelopt.torch.quantization.backends.nvfp4_gemm import Nvfp4Linear
from modelopt.torch.quantization.backends.utils import fp4_compatible, fp8_compatible


@pytest.mark.parametrize(
    ("model", "config", "gemm_forward"),
    [
        pytest.param(
            OneLayerLinear(in_features=64, out_features=32),
            mtq.NVFP4_DEFAULT_CFG,
            Nvfp4Linear.apply,
            marks=[
                pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU"),
            ],
        ),
        pytest.param(
            SimpleLinear(),
            mtq.FP8_DEFAULT_CFG,
            Fp8PerTensorLinear.apply,
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU"),
            ],
        ),
    ],
)
def test_gemm(model, config, gemm_forward):
    model = model.to(torch.float16).cuda()
    calib_data = [model.get_input().to(torch.float16).cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    module = model.net[0]
    input_tensor = calib_data[0].clone()
    expected = torch.nn.functional.linear(input_tensor, module.weight, bias=None)

    # Test without bias
    result_no_bias = gemm_forward(module, input_tensor, module.weight)
    diff = (result_no_bias - expected).abs()
    assert torch.allclose(result_no_bias, expected, atol=0.5, rtol=0.1), (
        f"Test without bias failed: {diff.amax()}\n"
        f"{result_no_bias[diff > 0.1]} != {expected[diff > 0.1]}"
    )

    # Generate a random bias for testing
    bias = torch.randn(module.weight.shape[0], device="cuda", dtype=torch.float16)
    expected_with_bias = torch.nn.functional.linear(input_tensor, module.weight, bias=bias)

    # Test 1: Bias as keyword argument (kwargs)
    result_with_bias_kwargs = gemm_forward(module, input_tensor, module.weight, bias=bias)
    diff = (result_with_bias_kwargs - expected_with_bias).abs()
    assert torch.allclose(result_with_bias_kwargs, expected_with_bias, atol=0.5, rtol=0.1), (
        f"Bias as kwargs failed: {diff.amax()}\n"
        f"{result_with_bias_kwargs[diff > 0.1]} != {expected_with_bias[diff > 0.1]}"
    )

    # Test 2: Bias as positional argument (args)
    result_with_bias_args = gemm_forward(module, input_tensor, module.weight, bias)
    diff = (result_with_bias_args - expected_with_bias).abs()
    assert torch.allclose(result_with_bias_args, expected_with_bias, atol=0.5, rtol=0.1), (
        f"Bias as args failed: {diff.amax()}\n"
        f"{result_with_bias_args[diff > 0.1]} != {expected_with_bias[diff > 0.1]}"
    )

    # Verify both methods produce the same result
    assert torch.equal(result_with_bias_kwargs, result_with_bias_args), (
        "Args and kwargs methods produced different results"
    )


@pytest.mark.parametrize(
    ("model", "config"),
    [
        pytest.param(
            SimpleLinear(),
            mtq.NVFP4_DEFAULT_CFG,
            marks=[
                pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU"),
            ],
        ),
        pytest.param(
            SimpleLinear(),
            mtq.FP8_DEFAULT_CFG,
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU"),
            ],
        ),
    ],
)
def test_compressed_backward_to_input(model, config):
    model = model.to(torch.float16).cuda()
    input_tensor = model.get_input().to(torch.float16).cuda()
    input_tensor.requires_grad = True
    _, bias_grads = compute_backward_grad(model, input_tensor, config=config, quantize=True)
    input_grad = input_tensor.grad
    input_tensor.grad = None
    weight_grads_quantized, bias_grads_quantized = compute_backward_grad(
        model, input_tensor, config=config, compress=True
    )
    input_grad_quantized = input_tensor.grad
    for weight_grad in weight_grads_quantized:
        assert weight_grad is None

    for bias_grad, bias_grad_quantized in zip(bias_grads, bias_grads_quantized):
        diff = (bias_grad - bias_grad_quantized).abs()
        assert torch.allclose(bias_grad, bias_grad_quantized, atol=0.5), (
            f"bias grad mismatch: {bias_grad[diff > 0.5]} != {bias_grad_quantized[diff > 0.5]}"
        )

    diff = (input_grad - input_grad_quantized).abs()
    assert torch.allclose(input_grad, input_grad_quantized, atol=0.2), (
        f"input grad mismatch: {diff.amax()}\n"
        f"{input_grad[diff > 0.2]} != {input_grad_quantized[diff > 0.2]}"
    )
