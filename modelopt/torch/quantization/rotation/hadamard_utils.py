# Copied from
import math
import os
from functools import cache

import scipy.linalg as linalg

try:
    import fast_hadamard_transform
except ImportError:
    print("fast_hadamard_transform not found. It's currently not needed for pre-fuse rotation.")
    fast_hadamard_transform = None
import torch

HADK_WEIGHTS = os.path.join(os.path.dirname(__file__), "hadK.pth")


@cache
def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)
        K = 172
        hadK = torch.load(HADK_WEIGHTS)[172]
        hadK = hadK.T if transpose else hadK
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)
        K = 156
        hadK = torch.load(HADK_WEIGHTS)[156]
        hadK = hadK.T if transpose else hadK
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)
        K = 140
        hadK = torch.load(HADK_WEIGHTS)[140]
        hadK = hadK.T if transpose else hadK
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)
        K = 108
        hadK = torch.load(HADK_WEIGHTS)[108]
        hadK = hadK.T if transpose else hadK
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)
        K = 60
        hadK = torch.load(HADK_WEIGHTS)[60]
        hadK = hadK.T if transpose else hadK
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)
        K = 52
        hadK = torch.load(HADK_WEIGHTS)[52]
        hadK = hadK.T if transpose else hadK
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = torch.load(HADK_WEIGHTS)[36]
        hadK = hadK.T if transpose else hadK
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        K = 28
        hadK = torch.load(HADK_WEIGHTS)[28]
        hadK = hadK.T if transpose else hadK
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = torch.load(HADK_WEIGHTS)[40]
        hadK = hadK.T if transpose else hadK
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = torch.load(HADK_WEIGHTS)[20]
        hadK = hadK.T if transpose else hadK
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = torch.load(HADK_WEIGHTS)[12]
        hadK = hadK.T if transpose else hadK
    else:
        assert is_pow2(n)
        K = 1
    if hadK is not None:
        assert hadK.shape[0] == K

    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


# See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
def random_hadamard_matrix(size, device):
    """Generate a random hadamard matrix."""
    Q = torch.randint(0, 2, (size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def random_base_hadamard_matrix(size, device):
    """Generate a random base hadamard matrix by Sylvester's construction."""
    assert is_pow2(size), "Size must be a power of 2 for hadamard matrix!"
    return torch.tensor(linalg.hadamard(size), dtype=torch.float64).to(device) / torch.sqrt(size)


# def matmul_hadU_cuda(X, hadK, K):
#     n = X.shape[-1]
#     if K == 1:
#         return fast_hadamard_transform.hadamard_transform(
#             X.contiguous(), 1.0 / torch.tensor(n).sqrt()
#         )
#     # if transpose:
#     #     hadK = hadK.T.contiguous()
#     input = X.view(-1, K, n // K)
#     input = fast_hadamard_transform.hadamard_transform(
#         input.contiguous(), 1.0 / torch.tensor(n).sqrt()
#     )
#     input = hadK.to(input.device).to(input.dtype) @ input
#     return input.reshape(X.shape)


def matmul_hadU_cuda(X, hadK, transpose=False):
    """Transpose is dummy here for support matmul_diag."""
    n = X.shape[-1]
    if hadK is None:
        assert is_pow2(n), "Input dimension must be a power of 2!"
        return fast_hadamard_transform.hadamard_transform(
            X.contiguous(),
            1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
        )
    # if transpose:
    #     hadK = hadK.T.contiguous()
    K = hadK.shape[0]
    input = X.view(*X.shape[:-1], K, n // K)
    input = fast_hadamard_transform.hadamard_transform(
        input.contiguous(),
        1.0 / math.sqrt(n),  # torch.tensor(n).sqrt()
    )
    # input = hadK.to(input.device).to(input.dtype) @ input
    input = hadK @ input
    return input.contiguous().view(X.shape)


def matmul_hadUt_cuda(X, hadK, K):
    return matmul_hadU_cuda(X, hadK, K, transpose=True)


def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    # Apply Hadamard to the last had_dim chunks of the weights
    elif output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = (
            fast_hadamard_transform.hadamard_transform(
                W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim),
                scale=1 / math.sqrt(had_dim),
            )
            .reshape(transposed_shape)
            .t()
        )
    else:
        raise NotImplementedError("Not implemented (or tested) yet!")
        n = W_.shape[1]
        W_ = hadamard_transform(
            W_.reshape(-1, n // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
        ).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
