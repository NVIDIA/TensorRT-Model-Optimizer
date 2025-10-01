#!/usr/bin/env python
"""Flash Attention with Softmax Skip Optimization

This module implements a Triton kernel for Flash Attention v2 with dynamic block skipping
based on softmax thresholds. Blocks with maximum attention scores below the threshold
are skipped to improve performance on sparse attention patterns.

Based on the original Flash Attention implementation with skip optimization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0, 1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor


def get_nvidia_autotune_configs():
    """Get autotune configurations optimized for NVIDIA GPUs.

    Returns:
        Tuple of (configs, keys) where:
        - configs: List of Triton Config objects
        - keys: List of parameter names that affect kernel compilation
    """
    configs = [
        # Large tiles for better memory coalescing on modern GPUs
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
        # Smaller tiles for different workload characteristics
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": True},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=8,
        ),
    ]

    # Keys for autotuning - parameters that affect kernel compilation
    # Note: softmax_skip_thresh and ENABLE_SKIP_STATS are added separately at kernel decorator
    keys = ["IS_CAUSAL", "dropout_p", "BLOCK_DMODEL", "USE_FP8"]

    return configs, keys


def check_args(
    q,
    k,
    v,
    o,
    varlen=True,
    max_seqlens=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    assert q.dim() == k.dim() and q.dim() == v.dim()
    if varlen:
        assert q.dim() == 3
        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
    else:
        assert q.dim() == 4
        batch, nheads_q, seqlen_q, head_size = q.shape
        _, nheads_k, seqlen_k, _ = k.shape
        assert max_seqlens > 0
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    # TODO: Change assert if we support qkl f8 and v f16
    assert q.dtype == k.dtype and q.dtype == v.dtype
    assert head_size <= 256
    assert o.shape == q.shape
    assert (nheads_q % nheads_k) == 0


# Get FP8 info for NVIDIA GPUs
if hasattr(torch, "float8_e4m3fn"):
    float8_info = torch.finfo(torch.float8_e4m3fn)
else:
    # Use FP16 as fallback if FP8 is not available
    float8_info = torch.finfo(torch.float16)


@triton.jit
def _attn_fwd_inner_with_skip(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    actual_seqlen_k,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    bias_ptr,
    softmax_skip_thresh,  # Skip threshold
    skip_counter_ptr,  # Optional skip statistics
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    USE_FP8: tl.constexpr,
    ENABLE_SKIP_STATS: tl.constexpr,
    qk_scale,
    p_descale,
):
    """Inner loop with softmax skip optimization.

    This function processes K/V blocks and skips computation for blocks
    where the maximum attention score is below the threshold.
    """
    # Pre-compute log2 of threshold for efficiency
    log2_thresh = tl.math.log2(softmax_skip_thresh)

    # Statistics tracking
    total_blocks = 0
    skipped_blocks = 0

    # Loop over K, V blocks and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        total_blocks += 1

        # Load K block
        k = load_fn(
            K_block_ptr,
            PADDED_HEAD,
            MASK_STEPS and (n_extra_tokens != 0),
            "zero",
        )

        # Compute QK dot product
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Apply masks if needed
        if MASK_STEPS:
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Compute QK scores
        qk += tl.dot(q, k)

        if USE_FP8:
            qk *= qk_scale

        # Add bias if provided
        if bias_ptr is not None:
            bias = load_fn(bias_ptr, False, MASK_STEPS and (n_extra_tokens != 0), "zero")
            qk += bias * 1.44269504089  # log2(e) for base-2 exp later

        # Find block maximum for skip decision
        block_max_qk = tl.max(qk, axis=1)
        block_max_overall = tl.max(block_max_qk)

        # Update running maximum
        m_ij = tl.maximum(m_i, block_max_overall)

        # Compute qk relative to new maximum
        qk_adjusted = qk - m_ij[:, None]

        # Check if block should be skipped (in log2 space for efficiency)
        block_max_log2 = tl.math.log2(tl.max(tl.math.exp2(qk_adjusted)))
        should_compute = block_max_log2 > log2_thresh

        if should_compute:
            # Compute softmax
            p = tl.math.exp2(qk_adjusted)

            # Sum for normalization
            l_ij = tl.sum(p, 1)

            # Apply dropout if enabled
            if ENABLE_DROPOUT:
                philox_offset = (
                    batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
                )
                keep = dropout_mask(
                    philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k
                )
                if RETURN_ENCODED_SOFTMAX:
                    tl.store(
                        encoded_softmax_block_ptr,
                        tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty),
                    )
                p = tl.where(keep, p, 0.0)
            elif RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty))

            # Update output accumulator
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]

            # Load V and accumulate
            if not PRE_LOAD_V:
                v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
            else:
                v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")

            # Update l_i
            l_i = l_i * alpha + l_ij

            # Apply FP8 descaling if needed
            if USE_FP8:
                p *= p_descale

            # Accumulate attention * value
            acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        else:
            # Skip this block - only update running maximum
            # The accumulator and normalization factor remain unchanged
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha

            # Track skipped blocks
            skipped_blocks += 1

        # Update m_i for next iteration
        m_i = m_ij

        # Advance pointers
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))

    # Store statistics if requested
    if ENABLE_SKIP_STATS and skip_counter_ptr is not None:
        tl.atomic_add(skip_counter_ptr, skipped_blocks)
        tl.atomic_add(skip_counter_ptr + 1, total_blocks)

    return acc, l_i, m_i


# Get autotune configurations
autotune_configs, autotune_keys = get_nvidia_autotune_configs()


@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys + ["softmax_skip_thresh", "ENABLE_SKIP_STATS"],
)
@triton.jit
def attn_fwd_with_skip(
    Q,
    K,
    V,
    bias,
    sm_scale,
    q_scale,
    k_scale,
    v_scale,
    p_scale,
    p_descale,
    o_descale,
    L,
    Out,
    stride_qz: tl.int64,
    stride_qh: tl.int64,
    stride_qm: tl.int64,
    stride_qk: tl.int64,
    stride_kz: tl.int64,
    stride_kh: tl.int64,
    stride_kn: tl.int64,
    stride_kk: tl.int64,
    stride_vz: tl.int64,
    stride_vh: tl.int64,
    stride_vk: tl.int64,
    stride_vn: tl.int64,
    stride_oz: tl.int64,
    stride_oh: tl.int64,
    stride_om: tl.int64,
    stride_on: tl.int64,
    stride_bz: tl.int64,
    stride_bh: tl.int64,
    stride_bm: tl.int64,
    stride_bn: tl.int64,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    softmax_skip_thresh,  # NEW: Skip threshold
    skip_counter_ptr,  # NEW: Optional skip statistics
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_FP8_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    ENABLE_SKIP_STATS: tl.constexpr,  # NEW: Enable statistics collection
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    """Flash Attention kernel with softmax skip optimization.

    This variant skips computation for blocks where the maximum attention score
    is below the threshold, providing significant speedup for sparse patterns.
    """
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Check for early exit in causal case
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            return

    # Handle MQA/GQA
    GROUP_SIZE: tl.constexpr = HQ // HK
    off_h_k = off_h_q // GROUP_SIZE if GROUP_SIZE != 1 else off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_head = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

    # Setup pointers
    q_offset = off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    if BIAS_TYPE != 0:
        bias_ptr = tl.make_block_ptr(
            base=bias + off_h_q * stride_bh,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        bias_ptr = None

    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + (off_z * HQ + off_h_q) * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0

    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
            base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
            shape=(seqlen_q, seqlen_k),
            strides=(seqlen_k, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        encoded_softmax_block_ptr = 0

    # Initialize state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504089
    q = load_fn(Q_block_ptr, True, padded_head, "zero")

    if not USE_FP8:
        q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
        acc_scale = 1.0
    else:
        qk_scale *= q_scale * k_scale
        acc_scale = p_scale * v_scale

    # Determine block ranges
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    # Process full blocks (no masking needed)
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner_with_skip(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            seqlen_k,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            encoded_softmax_block_ptr,
            block_min,
            block_max,
            0,
            0,
            0,
            bias_ptr,
            softmax_skip_thresh,  # Skip threshold
            skip_counter_ptr,  # Skip counter
            False,  # IS_CAUSAL (no causal mask for full blocks)
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            False,  # MASK_STEPS
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            padded_head,
            USE_FP8,
            ENABLE_SKIP_STATS,
            qk_scale,
            p_descale,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()

    # Process masked blocks
    if masked_blocks > 0:
        offs_n_causal = offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL else 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks * BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks * BLOCK_N, 0))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, n_full_blocks * BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, n_full_blocks))

        acc, l_i, m_i = _attn_fwd_inner_with_skip(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            start_m,
            seqlen_k,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            encoded_softmax_block_ptr,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            bias_ptr,
            softmax_skip_thresh,  # Skip threshold
            skip_counter_ptr,  # Skip counter
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            True,  # MASK_STEPS
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            padded_head,
            USE_FP8,
            ENABLE_SKIP_STATS,
            qk_scale,
            p_descale,
        )

    # Epilogue
    if USE_FP8:
        acc *= acc_scale
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)

    # Handle NaN rows for causal case
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if USE_FP8_OUT:
        acc *= o_descale
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    acc = acc.to(Out.type.element_ty)

    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = tl.zeros((1,), tl.float32)
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # Write output
    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))


class _attention_with_skip(torch.autograd.Function):
    """Attention with softmax skip optimization."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        causal=False,
        sm_scale=1.0,
        bias=None,
        fp8_scales=None,
        fp8_out_scale=None,
        softmax_skip_thresh=1e-4,  # NEW: Skip threshold
        collect_skip_stats=False,  # NEW: Collect skip statistics
    ):
        if fp8_scales is not None:
            use_fp8 = True
            (q_scale, k_scale, v_scale, p_scale) = fp8_scales
            float8 = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16

            def check_and_convert(t, scale):
                if t.dtype != float8:
                    descale = 1.0 / scale
                    ts = (t * descale).clamp(min=float8_info.min, max=float8_info.max)
                    return ts.to(float8)
                else:
                    return t

            q = check_and_convert(q, q_scale)
            k = check_and_convert(k, k_scale)
            v = check_and_convert(v, v_scale)
        else:
            use_fp8 = False
            q_scale = k_scale = v_scale = p_scale = 1.0

        if o is None:
            o = torch.empty_like(q, dtype=v.dtype)

        check_args(q, k, v, o, varlen=True, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)

        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        batch = len(cu_seqlens_q) - 1
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))

        # Get closest power of 2 for head dimension
        unpadded_head_dims = {32, 64, 128, 256}
        if head_size not in unpadded_head_dims:
            padded_d_model = None
            for i in unpadded_head_dims:
                if i > head_size:
                    padded_d_model = i
                    break
            assert padded_d_model is not None
        else:
            padded_d_model = head_size

        grid = lambda META: (
            triton.cdiv(max_seqlens_q, META["BLOCK_M"]),
            nheads_q,
            batch,
        )

        encoded_softmax = None
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        if bias is not None:
            bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
        else:
            bias_strides = (0, 0, 0, 0)

        p_descale = 1.0 / p_scale
        o_descale = 1.0 / fp8_out_scale.item() if fp8_out_scale is not None else 1.0

        # Allocate skip counter if stats collection is enabled
        skip_counter = None
        if collect_skip_stats:
            # Store [skipped_blocks, total_blocks]
            skip_counter = torch.zeros(2, dtype=torch.int32, device=q.device)

        # Call the skip-optimized kernel
        attn_fwd_with_skip[grid](
            q,
            k,
            v,
            bias,
            sm_scale,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            p_descale,
            o_descale,
            None,
            o,  # L and Out
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            *bias_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p=0.0,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            softmax_skip_thresh=softmax_skip_thresh,
            skip_counter_ptr=skip_counter,
            HQ=nheads_q,
            HK=nheads_k,
            ACTUAL_BLOCK_DMODEL=head_size,
            MAX_SEQLENS_Q=max_seqlens_q,
            MAX_SEQLENS_K=max_seqlens_k,
            IS_CAUSAL=causal,
            VARLEN=True,
            BLOCK_DMODEL=padded_d_model,
            BIAS_TYPE=0 if bias is None else 1,
            ENABLE_DROPOUT=False,
            RETURN_ENCODED_SOFTMAX=False,
            USE_FP8=use_fp8,
            USE_FP8_OUT=fp8_out_scale is not None,
            ENABLE_SKIP_STATS=collect_skip_stats,
        )

        # Save context for backward pass (if needed)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = causal
        ctx.dropout_p = 0.0
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = False
        ctx.softmax_skip_thresh = softmax_skip_thresh

        # Return skip statistics if collected
        if collect_skip_stats and skip_counter is not None:
            skipped = skip_counter[0].item()
            total = skip_counter[1].item()
            skip_ratio = skipped / total if total > 0 else 0.0
            return o, encoded_softmax, skip_ratio

        return o, encoded_softmax


triton_attention_with_skip = _attention_with_skip.apply


def triton_flash_attention_with_skip(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_skip_thresh: float = 1e-4,
    causal: bool = True,
    sm_scale: float | None = None,
    bias: torch.Tensor | None = None,
    collect_skip_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    """Flash Attention with softmax skip optimization.

    Args:
        q: Query tensor (total_tokens, num_heads, head_dim)
        k: Key tensor (total_tokens, num_kv_heads, head_dim)
        v: Value tensor (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys
        softmax_skip_thresh: Threshold for skipping softmax computation
        causal: Whether to use causal masking
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
        bias: Optional attention bias
        collect_skip_stats: Whether to collect skip statistics

    Returns:
        Tuple of:
        - Output tensor
        - None (for compatibility)
        - Skip ratio (if collect_skip_stats is True)
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Use the skip-optimized attention
    result = triton_attention_with_skip(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        sm_scale,
        bias,
        None,
        None,  # fp8_scales, fp8_out_scale
        softmax_skip_thresh,
        collect_skip_stats,
    )

    if collect_skip_stats:
        return result[0], None, result[2]  # output, None, skip_ratio
    else:
        return result[0], None, None  # output, None, None
