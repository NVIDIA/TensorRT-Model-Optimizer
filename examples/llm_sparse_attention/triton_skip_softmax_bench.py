#!/usr/bin/env python3
"""
Benchmark script for Triton Flash Attention with Skip Softmax optimization.

Compares baseline (softmax_skip_thresh=0.0) vs optimized (softmax_skip_thresh>0) performance.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Import the Triton skip kernel
sys.path.insert(0, str(Path(__file__).parent.parent))
from modelopt.torch.sparsity.attention_sparsity.kernel.flash_attn_triton_skip import (
    triton_flash_attention_with_skip,
)

# Block sizes for Flash Attention
BLOCK_ROWS = 128
BLOCK_COLS = 128


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_size: int = 1
    seq_len: int = 4096
    num_qo_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    sm_scale: float | None = None
    skip_threshold: float = 1e-4
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16
    warmup_iters: int = 10
    benchmark_iters: int = 50
    data_path: str | None = None
    layer_idx: int = 8
    calculate_sparsity: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: dict[str, Any]
    baseline_time_ms: float
    baseline_std_ms: float
    skip_time_ms: float
    skip_std_ms: float
    speedup: float
    actual_skip_ratio: float
    predicted_block_sparsity: float
    predicted_element_sparsity: float


class SparsityAnalyzer:
    """Analyzer for attention sparsity metrics."""

    def __init__(self):
        pass

    def calculate_sparsity(self, q, k, config):
        """Calculate sparsity metrics for given Q and K tensors."""
        # Always use batch_size=1 for sparsity calculation to save memory
        actual_batch_size = config.batch_size
        batch_size = 1
        qo_len = q.shape[0] // actual_batch_size
        kv_len = k.shape[0] // actual_batch_size
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim_qk = q.shape[2]

        # Take only the first batch for sparsity analysis
        q = q[:qo_len]
        k = k[:kv_len]

        # Expand K heads if using GQA
        gqa_group_ratio = num_qo_heads // num_kv_heads
        if gqa_group_ratio > 1:
            k_expanded = torch.repeat_interleave(k, gqa_group_ratio, dim=1)
        else:
            k_expanded = k

        # Calculate Q @ K^T
        with torch.amp.autocast("cuda", enabled=False):
            logits = torch.einsum(
                "bmhd,bnhd->bhmn",
                q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
                k_expanded.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
            ) * (config.sm_scale if config.sm_scale else 1.0 / math.sqrt(head_dim_qk))

        # Apply causal mask if needed
        if config.causal:
            causal_mask = torch.triu(torch.ones(qo_len, kv_len, device=q.device), diagonal=1)
            logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 1, float("-inf"))

        # Calculate block-level sparsity
        block_sparsity = self._calculate_block_sparsity(logits, config.skip_threshold)

        # Calculate reference sparsity (element-wise)
        reference_sparsity = self._calculate_reference_sparsity(logits, config.skip_threshold)

        return {"reference_sparsity": reference_sparsity, "block_sparsity": block_sparsity}

    def _calculate_block_sparsity(self, logits, threshold):
        """Calculate block-level sparsity based on threshold."""
        batch_size, num_heads, query_len, key_len = logits.shape

        # Calculate number of blocks
        num_block_rows = (query_len + BLOCK_ROWS - 1) // BLOCK_ROWS
        num_block_cols = (key_len + BLOCK_COLS - 1) // BLOCK_COLS

        # Pad if necessary
        padded_query_len = num_block_rows * BLOCK_ROWS
        padded_key_len = num_block_cols * BLOCK_COLS

        if query_len != padded_query_len or key_len != padded_key_len:
            padding = (0, padded_key_len - key_len, 0, padded_query_len - query_len)
            logits_padded = torch.nn.functional.pad(logits, padding, value=float("-inf"))
        else:
            logits_padded = logits

        # Reshape into blocks
        blocked_logits = logits_padded.view(
            batch_size, num_heads, num_block_rows, BLOCK_ROWS, num_block_cols, BLOCK_COLS
        )

        # Calculate block max
        block_max = blocked_logits.max(dim=-1)[0].max(dim=-2)[0]

        # Calculate cumulative max for causal attention
        if num_block_cols > 1:
            block_max_cummax = block_max.cummax(dim=-1)[0]
        else:
            block_max_cummax = block_max

        # Calculate probability threshold
        log_threshold = math.log(threshold) if threshold > 0 else float("-inf")

        # For each block, check if it could have values above threshold after softmax
        # We use the block max as an approximation
        # A block is considered sparse if its max value - cumulative max < log_threshold
        block_relative_max = block_max - block_max_cummax

        # Blocks that are definitely below threshold
        sparse_blocks_mask = block_relative_max < log_threshold

        # Calculate sparsity
        total_blocks = torch.numel(sparse_blocks_mask)
        sparse_blocks = torch.sum(sparse_blocks_mask).item()
        block_sparsity = sparse_blocks / total_blocks if total_blocks > 0 else 0.0

        return block_sparsity

    def _calculate_reference_sparsity(self, logits, threshold):
        """Calculate element-wise reference sparsity."""
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(logits, dim=-1)

        # Count values below threshold
        sparse_elements = torch.sum(attention_weights < threshold).item()
        total_elements = torch.numel(attention_weights)

        reference_sparsity = sparse_elements / total_elements if total_elements > 0 else 0.0

        return reference_sparsity


class TensorConverter:
    """Convert between different tensor formats for Triton kernel."""

    @staticmethod
    def to_ragged_format(
        tensor: torch.Tensor, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Convert [batch, seq, heads, dim] to ragged format.

        Args:
            tensor: Shape [batch, seq, heads, dim] or [batch, heads, seq, dim]
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            - Ragged tensor: [total_tokens, heads, dim]
            - cu_seqlens: [batch_size + 1] cumulative offsets
            - max_seqlen: Maximum sequence length
        """
        # Detect format and transpose if needed
        if tensor.shape[1] == seq_len:  # [batch, seq, heads, dim]
            tensor = tensor.transpose(1, 2)  # -> [batch, heads, seq, dim]

        # Now tensor is [batch, heads, seq, dim]
        # Reshape to [batch * seq, heads, dim]
        batch, num_heads, seq, head_dim = tensor.shape
        ragged = tensor.transpose(1, 2).reshape(batch * seq, num_heads, head_dim)

        # Create cumulative sequence lengths
        cu_seqlens = (
            torch.arange(0, batch_size + 1, device=tensor.device, dtype=torch.int32) * seq_len
        )

        return ragged, cu_seqlens, seq_len


def load_or_generate_tensors(
    config: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load real attention tensors or generate synthetic ones.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of Q, K, V tensors in standard format [batch, heads, seq, dim]
    """
    q = None
    k = None
    v = None

    # Load attention data
    if config.data_path is not None:
        q_path = Path(config.data_path) / f"q_layer_{config.layer_idx}.pt"
        k_path = Path(config.data_path) / f"k_layer_{config.layer_idx}.pt"

        if q_path.exists() and k_path.exists():
            print(f"Loading attention data from {config.data_path}")
            q_full = torch.load(q_path, map_location="cuda")
            k_full = torch.load(k_path, map_location="cuda")

            print(f"  Original tensor shapes - Q: {q_full.shape}, K: {k_full.shape}")

            # Handle different tensor formats
            if len(q_full.shape) == 4:
                # Format: [batch, num_heads, seq_len, head_dim]
                batch_in_file = q_full.shape[0]
                seq_len_q = q_full.shape[2]
                seq_len_k = k_full.shape[2]

                # Convert to target dtype
                q_full = q_full.to(config.dtype)
                k_full = k_full.to(config.dtype)

                # Handle batch size - repeat if needed
                if config.batch_size > batch_in_file:
                    # Repeat along batch dimension to match config.batch_size
                    repeats = (config.batch_size + batch_in_file - 1) // batch_in_file
                    q_full = q_full.repeat(repeats, 1, 1, 1)[: config.batch_size]
                    k_full = k_full.repeat(repeats, 1, 1, 1)[: config.batch_size]
                else:
                    # Take only the needed batches
                    q_full = q_full[: config.batch_size]
                    k_full = k_full[: config.batch_size]

                # Handle sequence length (dimension 2)
                if seq_len_q < config.seq_len:
                    # Need to repeat/pad sequence length
                    repeats = (config.seq_len + seq_len_q - 1) // seq_len_q
                    q_full = q_full.repeat(1, 1, repeats, 1)[:, :, : config.seq_len]
                else:
                    q_full = q_full[:, :, : config.seq_len]

                if seq_len_k < config.seq_len:
                    repeats = (config.seq_len + seq_len_k - 1) // seq_len_k
                    k_full = k_full.repeat(1, 1, repeats, 1)[:, :, : config.seq_len]
                else:
                    k_full = k_full[:, :, : config.seq_len]

                # Handle head/dim mismatch
                q_full = q_full[:, : config.num_qo_heads, :, : config.head_dim]
                k_full = k_full[:, : config.num_kv_heads, :, : config.head_dim]

                print(f"  Final Q shape: {q_full.shape}, K shape: {k_full.shape}")

                # Already in [batch, num_heads, seq, head_dim] format
                q = q_full
                k = k_full
            else:
                raise ValueError(f"Unexpected tensor format: Q shape {q_full.shape}")
        else:
            print(f"Data not found at {config.data_path}, generating synthetic tensors")
            q = None
            k = None

    # Generate synthetic tensors if data not loaded
    if q is None:
        print("Generating synthetic tensors")
        q = torch.randn(
            config.batch_size,
            config.num_qo_heads,
            config.seq_len,
            config.head_dim,
            dtype=config.dtype,
            device="cuda",
        )
        k = torch.randn(
            config.batch_size,
            config.num_kv_heads,
            config.seq_len,
            config.head_dim,
            dtype=config.dtype,
            device="cuda",
        )

    # Generate V tensor (always synthetic for now)
    v = torch.randn(
        config.batch_size,
        config.num_kv_heads,
        config.seq_len,
        config.head_dim,
        dtype=config.dtype,
        device="cuda",
    )

    return q, k, v


def benchmark_triton_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: BenchmarkConfig,
    threshold: float,
    collect_stats: bool = False,
) -> tuple[float, float, float | None]:
    """
    Benchmark Triton kernel with given threshold.

    Args:
        q, k, v: Input tensors in ragged format
        cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths
        max_seqlen_q, max_seqlen_k: Maximum sequence lengths
        config: Benchmark configuration
        threshold: softmax_skip_thresh value (0.0 for baseline)
        collect_stats: Whether to collect skip statistics

    Returns:
        Tuple of (mean_time_ms, std_time_ms, skip_ratio)
    """
    # Setup scale factor
    sm_scale = config.sm_scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(config.head_dim)

    # Create timing measurements list
    timings = []

    # Warmup phase
    for _ in range(config.warmup_iters):
        output, _ = triton_flash_attention_with_skip(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_skip_thresh=threshold,
            causal=config.causal,
            sm_scale=sm_scale,
            bias=None,
            collect_skip_stats=False,
        )
        torch.cuda.synchronize()

    # Benchmark phase
    skip_ratio = None
    for _ in range(config.benchmark_iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = triton_flash_attention_with_skip(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_skip_thresh=threshold,
            causal=config.causal,
            sm_scale=sm_scale,
            bias=None,
            collect_skip_stats=collect_stats,
        )
        if collect_stats:
            output, _, skip_ratio = result
        else:
            output, _ = result
        end_event.record()
        torch.cuda.synchronize()

        timings.append(start_event.elapsed_time(end_event))

    # Calculate statistics
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    return mean_time, std_time, skip_ratio


def run_single_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    # Print configuration
    print("\nBenchmarking config:")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Heads: qo={config.num_qo_heads}, kv={config.num_kv_heads}")
    print(f"  Skip threshold: {config.skip_threshold}")

    # Load/generate tensors
    q, k, v = load_or_generate_tensors(config)

    # Convert to ragged format
    converter = TensorConverter()
    q_ragged, cu_seqlens_q, max_seqlen_q = converter.to_ragged_format(
        q, config.batch_size, config.seq_len
    )
    k_ragged, cu_seqlens_k, max_seqlen_k = converter.to_ragged_format(
        k, config.batch_size, config.seq_len
    )
    v_ragged, _, _ = converter.to_ragged_format(v, config.batch_size, config.seq_len)

    # Analyze sparsity (optional)
    if config.calculate_sparsity:
        print("  Analyzing attention sparsity...")
        analyzer = SparsityAnalyzer()
        # Convert ragged back to standard for analysis
        q_std = q_ragged.view(
            config.batch_size, config.seq_len, config.num_qo_heads, config.head_dim
        ).transpose(1, 2)
        k_std = k_ragged.view(
            config.batch_size, config.seq_len, config.num_kv_heads, config.head_dim
        ).transpose(1, 2)
        # Reshape for analyzer: expects [total_tokens, num_heads, head_dim]
        q_for_analysis = q_ragged
        k_for_analysis = k_ragged
        sparsity_metrics = analyzer.calculate_sparsity(q_for_analysis, k_for_analysis, config)
        element_sparsity = sparsity_metrics["reference_sparsity"]
        block_sparsity = sparsity_metrics["block_sparsity"]
        print(f"    Predicted element sparsity: {element_sparsity:.1%}")
        print(f"    Predicted block sparsity: {block_sparsity:.1%}")
    else:
        element_sparsity = -1.0
        block_sparsity = -1.0

    # Run baseline (threshold=0.0)
    print("  Running baseline (thresh=0.0)...")
    baseline_time, baseline_std, _ = benchmark_triton_kernel(
        q_ragged,
        k_ragged,
        v_ragged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        config,
        threshold=0.0,
        collect_stats=False,
    )
    print(f"    Baseline: {baseline_time:.3f} ± {baseline_std:.3f} ms")

    # Run optimized (threshold>0)
    print(f"  Running skip-optimized (thresh={config.skip_threshold})...")
    skip_time, skip_std, actual_skip_ratio = benchmark_triton_kernel(
        q_ragged,
        k_ragged,
        v_ragged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        config,
        threshold=config.skip_threshold,
        collect_stats=True,
    )
    print(f"    Skip-optimized: {skip_time:.3f} ± {skip_std:.3f} ms")
    if actual_skip_ratio is not None:
        print(f"    Actual skip ratio: {actual_skip_ratio:.1%}")

    # Calculate speedup
    speedup = baseline_time / skip_time
    print(f"  Speedup: {speedup:.3f}x")

    # Create and return result
    result = BenchmarkResult(
        config=asdict(config),
        baseline_time_ms=baseline_time,
        baseline_std_ms=baseline_std,
        skip_time_ms=skip_time,
        skip_std_ms=skip_std,
        speedup=speedup,
        actual_skip_ratio=actual_skip_ratio if actual_skip_ratio else 0.0,
        predicted_block_sparsity=block_sparsity,
        predicted_element_sparsity=element_sparsity,
    )
    return result


def run_benchmark_sweep(args) -> list[BenchmarkResult]:
    """Run a sweep of benchmark configurations."""
    # Initialize results list
    results = []

    # Build configuration list based on sweep mode
    configs_to_test = []

    if args.sweep_mode == "thresholds":
        # Sweep over different BLASST thresholds
        for threshold in args.thresholds:
            config = BenchmarkConfig(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_qo_heads=args.num_qo_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                sm_scale=1.0 / math.sqrt(args.head_dim),
                skip_threshold=threshold,
                causal=args.causal,
                dtype=getattr(torch, args.dtype),
                warmup_iters=args.warmup,
                benchmark_iters=args.iterations,
                data_path=args.data_path,
                layer_idx=args.layer_idx,
                calculate_sparsity=not args.skip_sparsity,
            )
            configs_to_test.append(config)

    elif args.sweep_mode == "sequence_lengths":
        # Sweep over different sequence lengths
        for seq_len in args.seq_lengths:
            for threshold in args.thresholds:
                config = BenchmarkConfig(
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    num_qo_heads=args.num_qo_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    sm_scale=1.0 / math.sqrt(args.head_dim),
                    skip_threshold=threshold,
                    causal=args.causal,
                    dtype=getattr(torch, args.dtype),
                    warmup_iters=args.warmup,
                    benchmark_iters=args.iterations,
                    data_path=args.data_path,
                    layer_idx=args.layer_idx,
                    calculate_sparsity=not args.skip_sparsity,
                )
                configs_to_test.append(config)

    # Execute benchmarks with progress bar
    print(f"Running {len(configs_to_test)} benchmark configurations...")
    for config in tqdm(configs_to_test, desc="Benchmarking"):
        try:
            result = run_single_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"Error in benchmark: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Return results
    return results


def generate_summary(results: list[BenchmarkResult]):
    """Generate a summary of benchmark results."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Overall statistics
    speedups = [r.speedup for r in results]
    actual_skip_ratios = [r.actual_skip_ratio for r in results]
    pred_block_sparsities = [
        r.predicted_block_sparsity for r in results if r.predicted_block_sparsity >= 0
    ]

    print("\nOverall Statistics:")
    print(f"  Maximum Speedup: {np.max(speedups):.3f}x")
    print(f"  Minimum Speedup: {np.min(speedups):.3f}x")
    if pred_block_sparsities:
        print(f"  Average Predicted Block Sparsity: {np.mean(pred_block_sparsities):.1%}")
    if actual_skip_ratios:
        print(f"  Average Actual Skip Ratio: {np.mean(actual_skip_ratios):.1%}")

    # Results table with fixed-width formatting for better alignment
    print("\n" + "=" * 100)
    if results:
        first_config = results[0].config
        print(
            f"RESULTS TABLE (batch={first_config['batch_size']}, seq={first_config['seq_len']}, heads=[{first_config['num_qo_heads']},{first_config['num_kv_heads']}])"
        )
    else:
        print("RESULTS TABLE")
    print("=" * 100)

    # Print header
    print(
        f"{'Threshold':<12} {'Pred Block%':<12} {'Actual Skip%':<14} {'Baseline(ms)':<16} {'Skip(ms)':<16} {'Speedup':<8}"
    )
    print("-" * 100)

    # Print data rows
    for result in sorted(results, key=lambda x: x.config["skip_threshold"]):
        threshold = f"{result.config['skip_threshold']:.1e}"
        pred_blk = (
            f"{result.predicted_block_sparsity:.1%}"
            if result.predicted_block_sparsity >= 0
            else "N/A"
        )
        act_skip = f"{result.actual_skip_ratio:.1%}" if result.actual_skip_ratio >= 0 else "N/A"
        baseline = f"{result.baseline_time_ms:.3f}±{result.baseline_std_ms:.3f}"
        skip = f"{result.skip_time_ms:.3f}±{result.skip_std_ms:.3f}"
        speedup = f"{result.speedup:.3f}x"

        print(
            f"{threshold:<12} {pred_blk:<12} {act_skip:<14} {baseline:<16} {skip:<16} {speedup:<8}"
        )

    print("=" * 100)


def save_results_to_csv(results: list[BenchmarkResult], csv_path: str):
    """Save benchmark results to CSV file."""
    if not results:
        return

    # Get config info from first result
    first_config = results[0].config

    with open(csv_path, "w", newline="") as csvfile:
        # Write metadata as comments
        csvfile.write("# Benchmark Results\n")
        csvfile.write(f"# Batch Size: {first_config['batch_size']}\n")
        csvfile.write(f"# Sequence Length: {first_config['seq_len']}\n")
        csvfile.write(f"# Query Heads: {first_config['num_qo_heads']}\n")
        csvfile.write(f"# KV Heads: {first_config['num_kv_heads']}\n")
        csvfile.write(f"# Head Dimension: {first_config['head_dim']}\n")
        csvfile.write(f"# Data Type: {str(first_config['dtype']).replace('torch.', '')}\n")
        csvfile.write("#\n")

        # Write CSV data
        fieldnames = [
            "threshold",
            "predicted_block_sparsity",
            "predicted_element_sparsity",
            "actual_skip_ratio",
            "baseline_ms",
            "baseline_std_ms",
            "skip_ms",
            "skip_std_ms",
            "speedup",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in sorted(results, key=lambda x: x.config["skip_threshold"]):
            writer.writerow(
                {
                    "threshold": result.config["skip_threshold"],
                    "predicted_block_sparsity": (
                        result.predicted_block_sparsity
                        if result.predicted_block_sparsity >= 0
                        else ""
                    ),
                    "predicted_element_sparsity": (
                        result.predicted_element_sparsity
                        if result.predicted_element_sparsity >= 0
                        else ""
                    ),
                    "actual_skip_ratio": result.actual_skip_ratio
                    if result.actual_skip_ratio >= 0
                    else "",
                    "baseline_ms": result.baseline_time_ms,
                    "baseline_std_ms": result.baseline_std_ms,
                    "skip_ms": result.skip_time_ms,
                    "skip_std_ms": result.skip_std_ms,
                    "speedup": result.speedup,
                }
            )

    print(f"CSV results saved to: {csv_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark for Triton Flash Attention Skip Softmax optimization"
    )

    # Benchmark mode
    parser.add_argument(
        "--sweep-mode",
        type=str,
        default="thresholds",
        choices=["thresholds", "sequence_lengths"],
        help="Benchmark sweep mode",
    )

    # Sequence length parameters
    parser.add_argument(
        "--seq-len", type=int, default=4096, help="Sequence length (for single config)"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to test in sequence_lengths mode",
    )

    # Model configuration
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-qo-heads", type=int, default=32, help="Number of query/output heads")
    parser.add_argument("--num-kv-heads", type=int, default=32, help="Number of key/value heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")

    # Skip softmax parameters
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1],
        help="Skip softmax thresholds to test",
    )

    # Attention configuration
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Data type"
    )
    parser.add_argument(
        "--skip-sparsity",
        action="store_true",
        help="Skip sparsity calculation (useful for large tensors to avoid OOM)",
    )

    # Data parameters
    parser.add_argument("--data-path", type=str, default=None, help="Path to attention data")
    parser.add_argument("--layer-idx", type=int, default=8, help="Layer index for real data")

    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")

    # Output
    parser.add_argument(
        "--output", type=str, help="Output file base name (will generate .json and .csv files)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def check_environment() -> bool:
    """Check if the environment supports the benchmark."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return False

    # Check Triton import
    try:
        import triton

        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Error: Triton not installed")
        return False

    # Check kernel import
    try:
        from modelopt.torch.sparsity.attention_sparsity.kernel.flash_attn_triton_skip import (
            triton_flash_attention_with_skip,
        )
    except ImportError as e:
        print(f"Error: Cannot import Triton skip kernel: {e}")
        return False

    # Print device info
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")

    # Return success
    return True


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Print banner
    print("\n" + "=" * 100)
    print("TRITON FLASH ATTENTION SKIP SOFTMAX BENCHMARK")
    print("=" * 100)

    # Print configuration
    print("\nBenchmark Configuration:")
    print(f"  Sweep Mode: {args.sweep_mode}")
    print(f"  Skip Thresholds: {args.thresholds}")
    print(f"  Warmup Iterations: {args.warmup}")
    print(f"  Benchmark Iterations: {args.iterations}")
    print(f"  Data Type: {args.dtype}")

    if args.sweep_mode == "sequence_lengths":
        print(f"  Sequence Lengths: {args.seq_lengths}")
    else:
        print(f"  Sequence Length: {args.seq_len}")

    print("=" * 100)

    try:
        # Run benchmarks
        results = run_benchmark_sweep(args)

        # Generate summary
        generate_summary(results)

        # Save results if requested
        if args.output:
            # Generate base filename without extension
            output_path = Path(args.output)
            base_name = output_path.stem
            parent_dir = output_path.parent

            # Save JSON file
            json_path = parent_dir / f"{base_name}.json"
            output_data = []
            for r in results:
                r_dict = asdict(r)
                # Convert dtype to string for JSON serialization
                if "dtype" in r_dict["config"]:
                    r_dict["config"]["dtype"] = str(r_dict["config"]["dtype"]).replace("torch.", "")
                output_data.append(r_dict)

            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nJSON results saved to: {json_path}")

            # Save CSV file
            csv_path = parent_dir / f"{base_name}.csv"
            save_results_to_csv(results, str(csv_path))

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
