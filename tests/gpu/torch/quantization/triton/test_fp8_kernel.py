import pytest
import torch
from _test_utils.torch_misc import set_seed

from modelopt.torch.quantization.triton.fp8_kernel import fp8_gemm


@pytest.fixture(autouse=True)
def setup_seed():
    """Set seed before each test function."""
    set_seed()


@pytest.mark.parametrize(
    "M,N,K,dtype,with_bias",
    [
        (16, 16, 16, torch.float16, False),
        (32, 32, 32, torch.bfloat16, False),
        (16, 32, 48, torch.float16, False),
        (48, 32, 16, torch.bfloat16, False),
        (16, 16, 16, torch.float16, True),
        (32, 32, 32, torch.bfloat16, True),
    ],
)
def test_fp8_gemm_basic(M, N, K, dtype, with_bias):
    # Create random input matrices
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")
    # amax for scaling (simulate FP8 quantization)
    a_amax = a.abs().max()
    b_amax = b.abs().max()

    # Reference: simulate quantization/dequantization and matmul
    # Quantize to FP8 (simulate with clamping and scaling)
    a_scale = a_amax.to(torch.float32) / 448.0
    b_scale = b_amax.to(torch.float32) / 448.0
    a_fp8 = torch.clamp((a.to(torch.float32) / a_scale), -448.0, 448.0)
    b_fp8 = torch.clamp((b.to(torch.float32) / b_scale), -448.0, 448.0)

    bias = None if not with_bias else torch.randn(N, dtype=dtype, device="cuda")

    # Run fp8_gemm
    actual = fp8_gemm(a, b_fp8.to(torch.float8_e4m3fn), a_amax, b_amax, bias=bias).to(torch.float32)

    ref = torch._scaled_mm(
        a_fp8.to(torch.float8_e4m3fn),
        b_fp8.to(torch.float8_e4m3fn).T,
        a_scale,
        b_scale,
        use_fast_accum=True,
        out_dtype=a.dtype,
        bias=bias,
    ).to(torch.float32)

    # Compare
    assert actual.shape == (M, N)
    # Allow some tolerance due to quantization error
    torch.testing.assert_close(actual, ref, atol=1e-1, rtol=1e-1)
