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

"""Utility functions for performance measurement."""

from contextlib import ContextDecorator

import torch

from . import distributed as dist
from .logging import print_rank_0

__all__ = ["clear_cuda_cache", "get_cuda_memory_stats", "report_memory", "Timer"]


def clear_cuda_cache():
    """Clear the CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_memory_stats(device=None):
    """Get memory usage of specified GPU in Bytes."""
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
    }


def report_memory(name="", rank=0, device=None):
    """Simple GPU memory report."""
    memory_stats = get_cuda_memory_stats(device)
    string = name + " memory (MB)"
    for k, v in memory_stats.items():
        string += f" | {k}: {v / 2**20: .2e}"

    if dist.is_initialized():
        string = f"[Rank {dist.rank()}] " + string
        if dist.rank() == rank:
            print(string, flush=True)
    else:
        print(string, flush=True)


class Timer(ContextDecorator):
    """A Timer that can be used as a decorator as well."""

    def __init__(self, name=""):
        """Initialize Timer."""
        super().__init__()
        self.name = name
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)
        self.estimated_time = 0
        self.start()

    def start(self):
        """Start the timer."""
        self._start_event.record()
        return self

    def stop(self) -> float:
        """End the timer."""
        self._stop_event.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        self.estimated_time = self._start_event.elapsed_time(self._stop_event)
        return self.estimated_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        print_rank_0(f"{self.name} took {self.estimated_time:.3e} ms")
