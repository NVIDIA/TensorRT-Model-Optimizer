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
from modelopt.torch.quantization.backends.gemm_registry import enable_real_quant_gemm
from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.qtensor import NVFP4QTensor


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("shape", [(128, 64), (3, 16)])
def test_nvfp4_quantization(shape):
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
        cutlass_fp4_scale_to_modelopt_fp4_scale,
    )

    block_sizes = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    weight = torch.randn(shape).to(torch.float16).cuda()

    weight_fp4_trtllm, wsf_trtllm, wsf2_trtllm = NVFP4QTensor.quantize(weight, block_sizes[-1])
    weight_fp4, wsf, wsf2 = NVFP4QTensor.quantize(weight, block_sizes[-1], try_tensorrt=False)
    mask = weight_fp4_trtllm._quantized_data != weight_fp4._quantized_data
    diff = (
        weight_fp4_trtllm._quantized_data.to(torch.int16)
        - weight_fp4._quantized_data.to(torch.int16)
    ).abs()
    # two quantized_data are combined in an unin8.
    # 1, 16, and 17 mean the number in uint4 is only different by 1.
    assert (mask.sum((0, 1)) <= 30) and (set(diff[mask].unique().tolist()) <= {1, 16, 17}), (
        f"quantized data mismatch:\n"
        f"{weight_fp4_trtllm._quantized_data[mask]} != {weight_fp4._quantized_data[mask]}\n"
        f"{diff[mask]}"
    )
    wsf_converted = cutlass_fp4_scale_to_modelopt_fp4_scale(wsf_trtllm, weight.shape[-2:])
    mask = wsf_converted != wsf
    assert torch.equal(wsf_converted, wsf), (
        f"wsf mismatch: {wsf_converted.float()[mask]} != {wsf.float()[mask]}"
    )
    assert torch.equal(wsf2_trtllm, wsf2), f"wsf2 mismatch: {wsf2_trtllm} != {wsf2}"

    weight_dequantized_trtllm = weight_fp4_trtllm.dequantize(
        scale=wsf_trtllm, double_scale=wsf2_trtllm, block_sizes=block_sizes
    )
    weight_dequantized = weight_fp4.dequantize(
        scale=wsf, double_scale=wsf2, block_sizes=block_sizes
    )
    diff = (weight_dequantized_trtllm - weight_dequantized).abs()
    mask = diff > 0.1
    assert (mask.sum((0, 1)) <= 10) and (diff.amax() < 0.5), (
        f"dequantized data mismatch:\n"
        f"{weight_dequantized_trtllm[mask]} != {weight_dequantized[mask]}\n"
        f"{(weight_dequantized_trtllm[mask] - weight_dequantized[mask]).abs().amax()}"
    )
    diff = (weight_dequantized_trtllm - weight).abs()
    assert torch.allclose(weight_dequantized_trtllm, weight, atol=0.5, rtol=0.1), (
        f"dequantized data mismatch: {diff.amax()}\n"
        f"{weight_dequantized_trtllm[diff > 0.4]} != {weight[diff > 0.4]}\n"
    )


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("model", [OneLayerLinear(in_features=64, out_features=32)])
@pytest.mark.parametrize("config", [mtq.NVFP4_DEFAULT_CFG])
def test_nvfp4_gemm(model, config):
    model = model.to(torch.float16).cuda()
    calib_data = [model.get_input(2).to(torch.float16).cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    module = model.net[0]
    input_tensor = calib_data[0].clone()
    expected = torch.nn.functional.linear(input_tensor, module.weight, bias=None)

    # Find the matching GEMM implementation
    gemm_forward = gemm_registry.find_match(module, input_tensor, [], {})
    assert gemm_forward is not None

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


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("model", [OneLayerLinear(in_features=64, out_features=32)])
@pytest.mark.parametrize("config", [mtq.NVFP4_DEFAULT_CFG])
def test_forward(model, config):
    model = model.to(torch.float16).cuda()
    calib_data = [model.get_input(2).to(torch.float16).cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    input_tensor = calib_data[0].clone()
    output_fake = model(input_tensor)

    enable_real_quant_gemm(model)
    output_real = model(input_tensor)

    diff = (output_fake - output_real).abs()
    assert torch.allclose(output_fake, output_real, atol=0.2), (
        f"Difference between real and fake quantization: {diff.amax()}, {diff.mean()}, {diff.std()}"
    )

    mtq.compress(model)
    output_compressed_real = model(input_tensor)
    diff = (output_real - output_compressed_real).abs()
    # The difference is due to the scale factor.
    # The reciprocal of reciprocal is not the same as the original scale factor.
    assert torch.allclose(output_real, output_compressed_real, atol=0.1), (
        "Difference between real and compressed real quantization: "
        f"{diff.amax()}, {diff.mean()}, {diff.std()}"
    )


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize(
    "model",
    [
        OneLayerLinear(in_features=64, out_features=32),
        OneLayerLinear(in_features=64, out_features=32, bias=False),
    ],
)
@pytest.mark.parametrize("config", [mtq.NVFP4_DEFAULT_CFG])
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


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("model", [SimpleLinear()])
@pytest.mark.parametrize("config", [mtq.NVFP4_DEFAULT_CFG])
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
