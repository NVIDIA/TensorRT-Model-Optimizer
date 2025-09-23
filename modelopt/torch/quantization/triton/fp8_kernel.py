import torch
import triton
import triton.language as tl
from triton import Config

fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_amax_ptr,
    b_amax_ptr,
    bias_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        bias_ptr (tl.tensor): Pointer to the bias tensor.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_amax = tl.load(a_amax_ptr)
    b_amax = tl.load(b_amax_ptr)

    a_scale = a_amax / 448.0
    b_scale = b_amax / 448.0

    output_dtype = c_ptr.dtype.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        a_fp8 = tl.clamp((a / a_scale), -448.0, 448.0).to(tl.float8e4nv)
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a_fp8, b_fp8).to(tl.bfloat16)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = (accumulator * a_scale * b_scale).to(output_dtype)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c = c + bias
    # offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(
    a: torch.Tensor,
    b_fp8: torch.Tensor,
    a_amax: torch.Tensor,
    b_amax: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        a_amax (torch.Tensor): The amax for the first input matrix, must be a scalar.
        b_amax (torch.Tensor): The amax for the second input matrix, must be a scalar.
        bias (torch.Tensor | None): The bias tensor, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b_fp8.is_contiguous(), "Input tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b_fp8.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))
    fp8_gemm_kernel[grid](a, b_fp8, c, a_amax, b_amax, bias, M, N, K)

    return c
