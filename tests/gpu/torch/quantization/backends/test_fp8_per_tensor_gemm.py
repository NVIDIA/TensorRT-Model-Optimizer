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
from modelopt.torch.quantization.backends import gemm_registry
from modelopt.torch.quantization.backends.utils import fp8_compatible


@pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU")
@pytest.mark.parametrize("model_cls", [SimpleLinear])
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG])
def test_fp8_per_tensor_gemm(model_cls, config):
    """Test for fp8_per_tensor_gemm function with hardware-friendly dimensions."""
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    # Take the first module in the net
    module = model.net[0]
    input_tensor = calib_data[0].clone()
    expected = torch.nn.functional.linear(input_tensor, module.weight, bias=None)

    # Find the matching GEMM implementation
    gemm_forward = gemm_registry.find_match(module, input_tensor, [], {})
    assert gemm_forward is not None

    # Test without bias
    result_no_bias = gemm_forward(module, input_tensor, module.weight)
    assert torch.allclose(result_no_bias, expected, rtol=0.1, atol=0.1)

    # Generate a random bias for testing
    bias = torch.randn(module.weight.shape[0], device="cuda")
    expected_with_bias = torch.nn.functional.linear(input_tensor, module.weight, bias=bias)

    # Test 1: Bias as keyword argument (kwargs)
    args = []
    kwargs = {"bias": bias}
    result_with_bias_kwargs = gemm_forward(module, input_tensor, module.weight, *args, **kwargs)
    assert torch.allclose(result_with_bias_kwargs, expected_with_bias, rtol=0.1, atol=0.1), (
        "Bias as kwargs failed"
    )

    # Test 2: Bias as positional argument (args)
    args = [bias]
    kwargs = {}
    result_with_bias_args = gemm_forward(module, input_tensor, module.weight, *args, **kwargs)
    assert torch.allclose(result_with_bias_args, expected_with_bias, rtol=0.1, atol=0.1), (
        "Bias as args failed"
    )

    # Verify both methods produce the same result
    assert torch.allclose(result_with_bias_kwargs, result_with_bias_args, rtol=1e-5, atol=1e-5), (
        "Args and kwargs methods produced different results"
    )


@pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU")
@pytest.mark.parametrize(
    "model",
    [
        OneLayerLinear(in_features=64, out_features=32),
        OneLayerLinear(in_features=64, out_features=32, bias=False),
    ],
)
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG])
def test_backward(model, config):
    """Test backward

    with one single linear layer, the grad of no quant and real quant are the same.
    """
    model = model.to(torch.float16).cuda()
    input_tensor = model.get_input().to(torch.float16).cuda()
    weight_grads, bias_grads = compute_backward_grad(model, input_tensor)
    weight_grads_nvfp4, bias_grads_nvfp4 = compute_backward_grad(
        model, input_tensor, config=config, quantize=True, enable_real_quant=True
    )
    for bias_grad, bias_grad_nvfp4 in zip(bias_grads, bias_grads_nvfp4):
        mask = bias_grad != bias_grad_nvfp4
        assert (bias_grad is None and bias_grad_nvfp4 is None) or torch.equal(
            bias_grad, bias_grad_nvfp4
        ), f"bias grad mismatch: {bias_grad[mask]} != {bias_grad_nvfp4[mask]}"
    for weight_grad, weight_grad_nvfp4 in zip(weight_grads, weight_grads_nvfp4):
        mask = weight_grad != weight_grad_nvfp4
        assert torch.equal(weight_grad, weight_grad_nvfp4), (
            f"weight grad mismatch: {weight_grad[mask]} != {weight_grad_nvfp4[mask]}"
        )


@pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU")
@pytest.mark.parametrize("model", [SimpleLinear()])
@pytest.mark.parametrize("config", [mtq.FP8_DEFAULT_CFG])
def test_compressed_backward(model, config):
    model = model.to(torch.float16).cuda()
    input_tensor = model.get_input().to(torch.float16).cuda()
    input_tensor.requires_grad = True
    _, bias_grads = compute_backward_grad(model, input_tensor, config=config, quantize=True)
    input_grad = input_tensor.grad
    input_tensor.requires_grad = False
    weight_grads_nvfp4, bias_grads_nvfp4 = compute_backward_grad(
        model, input_tensor, config=config, compress=True
    )
    input_grad_nvfp4 = input_tensor.grad
    for weight_grad in weight_grads_nvfp4:
        assert weight_grad is None

    for bias_grad, bias_grad_nvfp4 in zip(bias_grads, bias_grads_nvfp4):
        diff = (bias_grad - bias_grad_nvfp4).abs()
        assert torch.allclose(bias_grad, bias_grad_nvfp4, atol=0.5), (
            f"bias grad mismatch: {bias_grad[diff > 0.5]} != {bias_grad_nvfp4[diff > 0.5]}"
        )

    diff = (input_grad - input_grad_nvfp4).abs()
    assert torch.allclose(input_grad, input_grad_nvfp4, atol=0.1), (
        f"input grad mismatch: {diff.amax()}\n"
        f"{input_grad[diff > 0.1]} != {input_grad_nvfp4[diff > 0.1]}"
    )
