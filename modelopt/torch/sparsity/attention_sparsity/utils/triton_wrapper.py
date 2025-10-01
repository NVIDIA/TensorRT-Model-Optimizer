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

"""Triton attention wrapper for HuggingFace integration."""

import torch

from ..kernel import triton_flash_attention_with_skip


def triton_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softmax_skip_thresh: float = 1e-4,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Triton Flash Attention wrapper for HuggingFace models.

    Converts HuggingFace tensor format to Triton varlen format, calls the Triton kernel,
    and converts back to HuggingFace format.

    Args:
        module: The attention module
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        attention_mask: Optional attention mask (ignored - using causal mask)
        dropout: Dropout probability (ignored)
        scaling: Scaling factor for attention scores
        softmax_skip_thresh: Threshold for skipping softmax computation
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Tuple of:
        - attention_output: Shape (batch, seq_len, num_heads, head_dim)
        - attention_weights: None (not computed in Flash Attention)
    """
    # Get dimensions
    batch_size = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    seq_len_q = query.shape[2]
    seq_len_k = key.shape[2]
    head_dim = query.shape[3]

    # Store original dtype
    original_dtype = query.dtype

    # Convert to float16 if needed (Triton kernel requires float16)
    if original_dtype != torch.float16:
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)

    # Convert from HuggingFace format to Triton format
    # HuggingFace: (batch, num_heads, seq_len, head_dim)
    # Triton: (total_tokens, num_heads, head_dim)

    # Reshape to (batch * seq_len, num_heads, head_dim)
    q_triton = query.transpose(1, 2).reshape(batch_size * seq_len_q, num_q_heads, head_dim)
    k_triton = key.transpose(1, 2).reshape(batch_size * seq_len_k, num_kv_heads, head_dim)
    v_triton = value.transpose(1, 2).reshape(batch_size * seq_len_k, num_kv_heads, head_dim)

    # Create cumulative sequence lengths
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device=query.device
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seq_len_k, seq_len_k, dtype=torch.int32, device=key.device
    )

    # Ensure tensors are contiguous
    q_triton = q_triton.contiguous()
    k_triton = k_triton.contiguous()
    v_triton = v_triton.contiguous()

    # Determine if causal masking should be applied
    is_causal = getattr(module, "is_causal", True) or (seq_len_q == seq_len_k)

    # Calculate scaling factor
    if scaling is None:
        scaling = 1.0 / (head_dim**0.5)

    # Call Triton kernel
    result = triton_flash_attention_with_skip(
        q_triton,
        k_triton,
        v_triton,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_len_q,
        seq_len_k,
        softmax_skip_thresh=softmax_skip_thresh,
        causal=is_causal,
        sm_scale=scaling,
        bias=None,
        collect_skip_stats=False,
    )

    # Extract output (result might be tuple if stats were collected)
    output = result[0] if isinstance(result, tuple) else result

    # Convert back to HuggingFace format
    # From (batch * seq_len, num_heads, head_dim) to (batch, seq_len, num_heads, head_dim)
    output = output.view(batch_size, seq_len_q, num_q_heads, head_dim)

    # Convert back to original dtype if needed
    if original_dtype != torch.float16:
        output = output.to(original_dtype)

    # Return in expected format: (batch, seq_len, num_heads, head_dim)
    return output, None
