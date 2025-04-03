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

"""Tests of tensor quantization function and module"""

import pytest
import torch
from _test_utils.torch_quantization.quant_utils import quant
from _test_utils.torch_quantization.tensor_quant_common import (
    FakeAffineTensorQuantTester,
    FakeTensorQuantTester,
    TensorQuantTester,
)

import modelopt.torch.quantization.triton as triton_kernel
import modelopt.torch.quantization.utils as quant_utils
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.extensions import get_cuda_ext, get_cuda_ext_fp8, get_cuda_ext_mx
from modelopt.torch.quantization.tensor_quant import mx_format_map


class TestTensorQuantCuda(TensorQuantTester):
    device = "cuda"


class TestFakeTensorQuantCuda(FakeTensorQuantTester):
    device = "cuda"

    def test_non_current_gpu(self, need_2_gpus):
        device = torch.cuda.device_count() - 1
        assert torch.cuda.current_device() != device
        x = torch.randn(3, 4).cuda(device)
        quant_x = tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)))
        quant_x_ref = quant(x, torch.max(torch.abs(x)), fake=True)
        assert torch.allclose(quant_x, quant_x_ref)


class TestFakeAffineTensorQuantCuda(FakeAffineTensorQuantTester):
    device = "cuda"


class TestCudaExt:
    @pytest.mark.parametrize("num_bits", [3, 4, 5, 7, 8, 11])
    @pytest.mark.parametrize("unsigned", [True, False])
    def test_cuda_ext_num_bits(self, num_bits, unsigned):
        x = torch.randn(31).cuda()

        if unsigned:
            x = x.abs()
        assert torch.allclose(
            get_cuda_ext().fake_tensor_quant(x, torch.max(torch.abs(x)), num_bits, unsigned),
            tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)), num_bits, unsigned),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_ext_dtype(self, dtype):
        # Test fp16 and bf16
        x = torch.randn(31).cuda().to(dtype)
        cuda_ext_out = (
            get_cuda_ext().fake_tensor_quant(x, torch.max(torch.abs(x))).to(torch.float32)
        )
        pytorch_out = tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x))).to(torch.float32)
        assert torch.allclose(cuda_ext_out, pytorch_out, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_bits", [3, 4, 5, 7, 8, 11])
    @pytest.mark.parametrize("unsigned", [True, False])
    def test_cuda_ext_with_axis(self, dtype, num_bits, unsigned):
        x = torch.randn(3, 4, 5, 6).cuda().to(dtype)

        # amax along axis 1
        amax_torch = torch.tensor([0.8, 0.9, 0.7, 0.6]).cuda()

        if unsigned:
            x = x.abs()
        cuda_ext_out = (
            get_cuda_ext()
            .fake_tensor_quant_with_axis(x, amax_torch, 1, num_bits, unsigned)
            .to(torch.float32)
        )
        pytorch_out = tensor_quant.fake_tensor_quant(
            x, amax_torch.view(1, -1, 1, 1), num_bits, unsigned
        ).to(torch.float32)
        assert torch.allclose(cuda_ext_out, pytorch_out, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_ext_inplace(self, dtype):
        torch.manual_seed(1234)
        x = torch.randn(31).cuda().to(dtype)
        quant_x_ref = quant(x, torch.max(x.abs()), fake=True)
        get_cuda_ext().fake_tensor_quant_(x, torch.max(torch.abs(x)))
        if dtype == torch.bfloat16:
            assert torch.allclose(x, quant_x_ref, atol=1e-1)
        elif dtype == torch.float16:
            assert torch.allclose(x, quant_x_ref, atol=1e-3)
        else:
            assert torch.allclose(x, quant_x_ref)

    def test_cuda_ext_tiny_amax(self):
        x = torch.rand(2, 3, 4).cuda()
        amax = torch.tensor([1.0, 1.0e-26, 1.0]).cuda().unsqueeze(-1).unsqueeze(1)
        quant_x = get_cuda_ext().fake_tensor_quant_with_axis(x, amax, axis=1)
        assert quant_x[:, 1, :].sum() == 0


class TestScaledE4M3:
    x = [
        [-2.0000, -1.8000, -1.6000, -1.4000, -1.2000],
        [-1.0000, -0.8000, -0.6000, -0.4000, -0.2000],
        [-0.0000, 0.2000, 0.4000, 0.6000, 0.8000],
        [1.0000, 1.2000, 1.4000, 1.6000, 1.8000],
    ]

    xq_unscaled = [
        [-2.0000, -1.7500, -1.6250, -1.3750, -1.2500],
        [-1.0000, -0.8125, -0.6250, -0.4062, -0.2031],
        [0.0000, 0.2031, 0.4062, 0.6250, 0.8125],
        [1.0000, 1.2500, 1.3750, 1.6250, 1.7500],
    ]

    xq_scaled = [
        [-2.0000, -1.8571, -1.5714, -1.4286, -1.1429],
        [-1.0000, -0.7857, -0.5714, -0.3929, -0.1964],
        [0.0000, 0.1964, 0.3929, 0.5714, 0.7857],
        [1.0000, 1.1429, 1.4286, 1.5714, 1.8571],
    ]

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_e4m3_no_scale(self, device):
        x = torch.tensor(TestScaledE4M3.x).to(device)
        xq_ref = torch.tensor(TestScaledE4M3.xq_unscaled).to(device)
        e4m3_x = tensor_quant.scaled_e4m3(x, None, 4, 3)
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_with_amax(self, device):
        x = torch.tensor(TestScaledE4M3.x).to(device).unsqueeze(-1)
        xq_ref = torch.tensor(TestScaledE4M3.xq_scaled).to(device).unsqueeze(-1)

        amax = quant_utils.reduce_amax(x, axis=None, keepdims=True)

        e4m3_x = tensor_quant.scaled_e4m3(x, amax, 4, 3)

        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    def test_e4m3_incontiguous(self):
        x = torch.tensor(TestScaledE4M3.x).cuda().transpose(1, 0)
        xq_ref = torch.tensor(TestScaledE4M3.xq_unscaled).cuda().transpose(1, 0)
        assert not x.is_contiguous()
        e4m3_x = tensor_quant.scaled_e4m3(x, None, 4, 3)
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_backward(self, device):
        x = torch.randn(3, 7, requires_grad=True).to(device)
        labels = torch.randint(6, (3,)).type(torch.LongTensor).to(device)
        quant_x = tensor_quant.scaled_e4m3(x, None, 4, 3)
        x.retain_grad()
        quant_x.retain_grad()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(quant_x, labels)
        loss.backward()
        assert torch.allclose(quant_x.grad, x.grad)

    def test_non_current_gpu(self, need_2_gpus):
        device = torch.cuda.device_count() - 1
        assert torch.cuda.current_device() != device
        x = torch.randn(3, 4).cuda()
        quant_x_ref = tensor_quant.scaled_e4m3(x, x.amax(), 4, 3)
        x = x.cuda(device)
        quant_x = tensor_quant.scaled_e4m3(x, x.amax(), 4, 3)
        assert torch.allclose(quant_x, quant_x_ref.cuda(device))

    def test_fused_e4m3_kernel(self):
        cuda_ext_fp8 = get_cuda_ext_fp8()
        x = torch.tensor(TestScaledE4M3.x).cuda()
        xq_ref = torch.tensor(TestScaledE4M3.xq_scaled).cuda()
        amax = torch.ones(1, x.shape[-1]).cuda() * x.abs().amax()
        e4m3_x = cuda_ext_fp8.fused_fake_e4m3fy(x, amax.float(), 1.0 / (1 << 24))
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    def test_e4m3_kernel_non_last_axis(self):
        x = torch.tensor(TestScaledE4M3.x).cuda()
        xq_ref = torch.tensor(TestScaledE4M3.xq_scaled).cuda()
        amax = torch.ones(x.shape[0], 1).cuda() * x.abs().amax()
        e4m3_x = tensor_quant.scaled_e4m3(x, amax, 4, 3)
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)


class Testfp4:
    @pytest.mark.skipif(get_cuda_ext_mx() is None, reason="cuda_ext_mx is not available")
    @pytest.mark.parametrize(
        "set_torch_dtype", [torch.float, torch.float16, torch.bfloat16], indirect=True
    )
    @pytest.mark.parametrize("block_size", [8, 16, 32])
    def test_cuda_ext_fp4(self, set_torch_dtype, block_size):
        cuda_ext_mx = get_cuda_ext_mx()
        # Test with e2m1 table values
        sign = torch.randint(0, 2, (1, 8)).cuda() * 2 - 1

        def _get_test_inputs_outputs(test_in, test_out):
            return torch.concat((test_in,) * (block_size // 8), dim=-1), torch.concat(
                (test_out,) * (block_size // 8), dim=-1
            )

        def _test_fp4_kernel(test_in, test_out):
            inputs, expected_outputs = _get_test_inputs_outputs(test_in, test_out)
            quantized_outputs = cuda_ext_mx.fused_amax_convert(
                inputs,
                16,
                getattr(cuda_ext_mx.Types, mx_format_map[(2, 1)]),
                getattr(cuda_ext_mx.Types, mx_format_map[(4, 3)]),
                inputs.abs().amax(),
            )
            assert torch.allclose(quantized_outputs, expected_outputs)
            if triton_kernel.IS_AVAILABLE:
                quantized_outputs_triton = triton_kernel.fp4_fake_quant_block(
                    inputs, inputs.abs().amax().item()
                )
                assert torch.allclose(quantized_outputs_triton, expected_outputs)

        test_in = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        test_out = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)

        # Test with e2m1 boundary values. The even indexes are rounded down and odd indexes are rounded up.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda() * sign
        test_out = torch.tensor([[0.0, 1, 1, 2, 2, 4, 4, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)

        # Test slightly below the e2m1 boundary values.
        # Numbers should be quantized down to the corresponding e2m1 value.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda()
        test_in[:, :-1] -= 0.1
        test_in *= sign
        test_out = torch.tensor([[0.0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)

        # Test slightly above the e2m1 boundary values.
        # Numbers should be quantized up to the corresponding e2m1 value.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda()
        test_in[:, :-1] += 0.1
        test_in *= sign
        test_out = torch.tensor([[0.5, 1, 1.5, 2, 3, 4, 6, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)
