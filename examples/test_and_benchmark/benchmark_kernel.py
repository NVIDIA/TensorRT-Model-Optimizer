#!/usr/bin/env python
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Triton FP4 fake quantization kernels."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from modelopt.torch.quantization.triton import fp4_kernel as fp4
from modelopt.torch.quantization.triton import fp4_kernel_new as fp4_new


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    amax: torch.Tensor,
    repetitions: int,
) -> float:
    # Warm-up
    for _ in range(5):
        fn(x, amax)
    _synchronize_cuda()

    times: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn(x, amax)
        _synchronize_cuda()
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times)


def run_benchmark(
    shapes: Sequence[tuple[int, int]],
    dtypes: Sequence[torch.dtype],
    block_size: int,
    tile_rows: int,
    tile_cols: int,
    repetitions: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = "cuda"

    for shape in shapes:
        m_dim, n_dim = shape
        print(f"\nShape: {m_dim}x{n_dim}")
        for dtype in dtypes:
            print(f"  dtype: {dtype}")

            torch.manual_seed(0)
            x = torch.randn(m_dim, n_dim, device=device, dtype=dtype)
            amax = x.abs().amax()

            def _old_kernel(inp: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
                return fp4.fp4_fake_quant_block(inp, a, block_size=block_size, tile_size=128)

            def _new_kernel(inp: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
                return fp4_new.fp4_fake_quant_block_v2(
                    inp,
                    a,
                    block_size=block_size,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                )

            out_old = _old_kernel(x, amax)
            out_new = _new_kernel(x, amax)
            max_diff = (out_old - out_new).abs().max().item()
            print(f"    max abs diff: {max_diff:.3e}")

            old_time = _benchmark(_old_kernel, x, amax, repetitions)
            new_time = _benchmark(_new_kernel, x, amax, repetitions)
            speedup = old_time / new_time if new_time > 0 else float("inf")

            print(f"    old kernel: {old_time * 1e6:.2f} µs")
            print(f"    new kernel: {new_time * 1e6:.2f} µs")
            print(f"    speedup: {speedup:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP4 Triton kernels")
    parser.add_argument(
        "--shapes",
        type=str,
        default="512x512,1024x1024,4096x4096,8192x8192,8192x12288,12288x12288,32x4096,1024x4096,32x5000,128x8200",
        help="Comma-separated list of MxN shapes (includes common LLM shapes by default)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float32,bfloat16,float16",
        help="Comma-separated list of dtypes",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=50)
    return parser.parse_args()


def _parse_shapes(arg: str) -> list[tuple[int, int]]:
    shapes = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        m_str, n_str = item.lower().split("x")
        shapes.append((int(m_str), int(n_str)))
    return shapes


def _parse_dtypes(arg: str) -> list[torch.dtype]:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtypes = []
    for item in arg.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported dtype: {item}")
        dtypes.append(mapping[key])
    return dtypes


def main() -> None:
    args = parse_args()
    shapes = _parse_shapes(args.shapes)
    dtypes = _parse_dtypes(args.dtypes)
    run_benchmark(
        shapes=shapes,
        dtypes=dtypes,
        block_size=args.block_size,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        repetitions=args.repetitions,
    )


if __name__ == "__main__":
    main()
