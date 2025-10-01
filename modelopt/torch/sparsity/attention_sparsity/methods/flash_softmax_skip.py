"""Flash Attention-aware softmax skip method for sparse attention.

This module implements block-wise sparsity that aligns with Flash Attention's
processing pattern for optimal performance.
"""

import math

import numpy as np
import torch

from . import SparseAttentionMethod, register_sparse_method


@register_sparse_method("flash_softmax_skip")
class FlashSoftmaxSkipMethod(SparseAttentionMethod):
    """Flash Attention-aware softmax skip sparse attention method.

    This method implements block-wise sparsity that aligns with Flash Attention's
    tiling pattern. It applies threshold-based masking at the block level rather
    than element-wise, which is more efficient for Flash Attention implementations.

    The method supports:
    - Block-wise sparsity aligned with Flash Attention tiles
    - Correction factor tracking for accurate softmax computation
    - Phase-aware thresholds (different for prefill vs decode)
    - Statistics collection for sparsity analysis
    """

    def __init__(self, method_config: dict = None):
        """Initialize Flash softmax skip method.

        Args:
            method_config: Configuration dictionary containing:
                - threshold: Sparsity threshold (float or dict with prefill/decode)
                - br: Block row size (default: 128)
                - bc: Block column size (default: 128)
                - enable_correction_factor: Whether to track correction factor
                - collect_stats: Whether to collect statistics
                - phase: Force specific phase ('prefill' or 'decode', default: auto)
                - backend: Backend to use ('pytorch' or 'triton', default: 'pytorch')
        """
        config = method_config or {}

        # Extract configuration
        self.threshold_config = config.get("threshold", 1e-4)
        self.br = config.get("br", 128)
        self.bc = config.get("bc", 128)
        self.enable_correction_factor = config.get("enable_correction_factor", True)
        self.collect_stats = config.get("collect_stats", True)
        self.phase = config.get("phase", None)
        self.backend = config.get("backend", "pytorch")

        # Initialize current threshold
        if isinstance(self.threshold_config, dict):
            # Use default threshold initially, will be updated based on phase
            self.threshold = self.threshold_config.get(
                "default", self.threshold_config.get("prefill", 1e-4)
            )
        else:
            self.threshold = self.threshold_config

        # Statistics collection
        self.stats = {}

        # Calibration mode for collecting per-sample statistics
        self.calibration_mode = False
        self.stats_history = []  # List to store per-sample stats during calibration

    def _update_threshold(self, phase: str):
        """Update threshold based on phase.

        Args:
            phase: Current phase ('prefill' or 'decode')
        """
        if isinstance(self.threshold_config, dict):
            if phase in self.threshold_config:
                self.threshold = self.threshold_config[phase]
            elif "default" in self.threshold_config:
                self.threshold = self.threshold_config["default"]

    def set_calibration_mode(self, enabled: bool, reset_history: bool = True):
        """Enable or disable calibration mode for per-sample stats collection.

        Args:
            enabled: Whether to enable calibration mode
            reset_history: Whether to reset stats_history when enabling
        """
        self.calibration_mode = enabled
        self.collect_stats = enabled  # Ensure stats collection is on during calibration
        if enabled and reset_history:
            self.stats_history = []

    def get_calibration_stats(self) -> list[dict]:
        """Get accumulated calibration statistics.

        Returns:
            List of per-sample statistics dictionaries
        """
        return self.stats_history

    def _infer_phase(self, attention_scores: torch.Tensor) -> str:
        """Infer whether we're in prefill or decode phase based on tensor shape.

        Args:
            attention_scores: Attention score tensor with shape [batch, heads, seq_q, seq_k]

        Returns:
            'prefill' if seq_len > 1, 'decode' if seq_len == 1
        """
        # Attention scores are always 4D: [batch, heads, seq_q, seq_k]
        seq_q = attention_scores.shape[2]
        return "decode" if seq_q == 1 else "prefill"

    def _reshape_to_blocks(
        self, tensor: torch.Tensor, br: int, bc: int
    ) -> tuple[torch.Tensor, ...]:
        """Reshape tensor into blocks for Flash Attention processing.

        Args:
            tensor: Input tensor of shape [batch, heads, seq_q, seq_k]
            br: Block row size
            bc: Block column size

        Returns:
            Tuple of (blocked_tensor, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k)
        """
        batch_size, num_heads, seq_q, seq_k = tensor.shape

        # Calculate padding needed
        padded_seq_q = math.ceil(seq_q / br) * br
        padded_seq_k = math.ceil(seq_k / bc) * bc

        # Pad tensor if necessary
        if padded_seq_q != seq_q or padded_seq_k != seq_k:
            pad_q = padded_seq_q - seq_q
            pad_k = padded_seq_k - seq_k
            tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_q), value=float("-inf"))

        # Reshape to blocks
        num_block_rows = padded_seq_q // br
        num_block_cols = padded_seq_k // bc

        blocked = tensor.view(
            batch_size, num_heads, num_block_rows, br, num_block_cols, bc
        ).permute(0, 1, 2, 4, 3, 5)  # [batch, heads, block_rows, block_cols, br, bc]

        return blocked, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k

    def calc_correction_factor_and_P(
        self, attn_weights: torch.Tensor, phase: str
    ) -> tuple[torch.Tensor, dict]:
        """Calculate correction factor and apply block-wise sparsity.

        This is the core Flash Attention-aware sparsity logic that:
        1. Reshapes attention into blocks matching Flash Attention tiles
        2. Computes block-wise maximums and tracks cumulative maximums
        3. Applies threshold-based masking at the block level
        4. Tracks correction factor for accurate softmax computation

        Args:
            attn_weights: Attention weights tensor
            phase: "prefill" or "decode"

        Returns:
            Tuple of (sparse_mask, stats_dict)
        """
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape

        # Use log threshold for numerical stability
        log_threshold = np.log(self.threshold)

        if phase == "decode":
            # Decode: Process single query against key sequence
            blocked_attn, _, num_block_cols, _, padded_seq_k = self._reshape_to_blocks(
                attn_weights, 1, self.bc
            )

            # Compute block-wise maximum and cumulative maximum
            # Shape: [batch, heads, 1, num_block_cols]
            block_max = blocked_attn.max(dim=-1)[0].max(dim=-1)[0]
            block_max_cummax = block_max.cummax(dim=-1)[0]

            # Track when maximum changes (for correction factor)
            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = float(torch.sum(block_max_larger) / torch.numel(block_max_larger))

            # Normalize blocks relative to cumulative maximum
            # P represents how much each value differs from the running maximum
            P = blocked_attn - block_max_cummax.unsqueeze(-1).unsqueeze(-1)

            # Apply threshold: keep blocks where max value > threshold
            block_mask = P.max(dim=-1)[0].max(dim=-1)[0] > log_threshold

            # Expand block mask to element level
            element_mask = block_mask.unsqueeze(-1).unsqueeze(-1).expand_as(blocked_attn)

            # Reshape back to original dimensions
            element_mask = element_mask.permute(0, 1, 2, 4, 3, 5).contiguous()
            element_mask = element_mask.view(batch_size, num_heads, 1, padded_seq_k)

            # Remove padding
            element_mask = element_mask[:, :, :seq_q, :seq_k]

            # Collect statistics
            if self.collect_stats:
                total_blocks = num_block_cols
                kept_blocks = block_mask.sum().item() / (batch_size * num_heads)
                sparsity = 1 - (kept_blocks / total_blocks)
            else:
                sparsity = 0.0

        else:  # prefill
            # Prefill: Process multiple queries against key sequence
            blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
                self._reshape_to_blocks(attn_weights, self.br, self.bc)
            )

            # Compute block-wise statistics
            # For each block, compute maximum value
            block_max = blocked_attn.max(dim=-1)[0].max(dim=-1)[
                0
            ]  # [batch, heads, block_rows, block_cols]

            # For Flash Attention compatibility, track row-wise cumulative maximum
            # This simulates the online softmax computation in Flash Attention
            if num_block_cols > 1:
                block_max_cummax = block_max.cummax(dim=-1)[0]

                # Track correction factor (how often maximum changes)
                block_max_larger = torch.ones_like(block_max)
                block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
                correction_factor = float(
                    torch.sum(block_max_larger) / torch.numel(block_max_larger)
                )

                # Compute P (normalized difference from cumulative max)
                P = blocked_attn - block_max_cummax.unsqueeze(-1).unsqueeze(-1)
            else:
                # Single block column, no cumulative max needed
                correction_factor = 1.0
                P = blocked_attn - block_max.unsqueeze(-1).unsqueeze(-1)

            # Apply threshold at block level
            block_mask = P.max(dim=-1)[0].max(dim=-1)[0] > log_threshold

            # Expand to element level
            element_mask = block_mask.unsqueeze(-1).unsqueeze(-1).expand_as(blocked_attn)

            # Reshape back
            element_mask = element_mask.permute(0, 1, 2, 4, 3, 5).contiguous()
            element_mask = element_mask.view(batch_size, num_heads, padded_seq_q, padded_seq_k)

            # Remove padding
            element_mask = element_mask[:, :, :seq_q, :seq_k]

            # Statistics
            if self.collect_stats:
                total_blocks = num_block_rows * num_block_cols
                kept_blocks = block_mask.sum().item() / (batch_size * num_heads)
                sparsity = 1 - (kept_blocks / total_blocks)
            else:
                sparsity = 0.0

        # Store statistics
        stats = {
            "correction_factor": correction_factor if self.enable_correction_factor else 1.0,
            "sparsity": sparsity,
            "phase": phase,
            "total_blocks": num_block_cols
            if phase == "decode"
            else num_block_rows * num_block_cols,
            "sparse_blocks": int(
                sparsity
                * (num_block_cols if phase == "decode" else num_block_rows * num_block_cols)
            ),
            "sample_length": seq_k,
        }

        if self.collect_stats:
            self.stats = stats

        # During calibration, also append to history for per-sample tracking
        if self.calibration_mode:
            # Add sample length information from input shape
            sample_stats = stats.copy()
            self.stats_history.append(sample_stats)

        return element_mask, stats

    def apply_sparsity(
        self,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply Flash Attention-aware block-wise sparsity.

        Args:
            query: Query tensor (unused, for API compatibility)
            key: Key tensor (unused, for API compatibility)
            value: Value tensor (unused, for API compatibility)
            attention_scores: Attention scores tensor with shape [batch, heads, seq_q, seq_k]

        Returns:
            Tuple with potentially modified attention_scores
        """
        # Attention scores must be provided for sparse attention
        assert attention_scores is not None, "attention_scores must be provided for apply_sparsity"

        # Attention scores are always 4D: [batch, heads, seq_q, seq_k]
        assert len(attention_scores.shape) == 4, (
            f"Expected 4D attention scores, got shape {attention_scores.shape}"
        )

        # Infer phase from tensor shape
        phase = self._infer_phase(attention_scores)

        # Update threshold for the detected phase (skip during calibration to preserve set threshold)

        if not self.calibration_mode:
            self._update_threshold(phase)

        # Apply block-wise sparsity
        sparse_mask, stats = self.calc_correction_factor_and_P(attention_scores, phase)

        # Apply mask to create sparse scores
        sparse_scores = attention_scores.masked_fill(~sparse_mask, float("-inf"))

        return query, key, value, sparse_scores

    @property
    def name(self) -> str:
        """Method identifier."""
        return "flash_softmax_skip"
