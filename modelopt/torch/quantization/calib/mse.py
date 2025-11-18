# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Calibrator that returns the MSE amax of all collected tensors."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .. import utils as quant_utils
from .calibrator import _Calibrator

__all__ = ["MseCalibrator"]


class MseCalibrator(_Calibrator):
    """Per-tensor and per-channel MSE amax search that minimizes error between x and quantized x."""

    def __init__(
        self,
        amax: torch.Tensor,
        axis: int | tuple | list | None = None,
        num_steps: int = 10,
        start_multiplier: float = 0.25,
        stop_multiplier: float = 4.0,
        quant_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize MSE calibrator.

        Args:
            amax: Initial amax value (required).
            axis: Quantization axis. None means per-tensor quantization.
            num_steps: Number of amax candidates to try.
            start_multiplier: Starting multiplier for amax search.
            stop_multiplier: Ending multiplier for amax search.
            quant_func: Function that quantizes input tensor given an amax value.
                       Should have signature: quant_func(x, amax) -> quantized_x.
            error_func: Function to compute error between x and xq.
                       Default is F.mse_loss(x, xq, reduction='none').
        """
        super().__init__(num_bits=None, axis=axis, unsigned=None)
        self._initial_amax = amax
        self._num_steps = num_steps
        self._start_multiplier = start_multiplier
        self._stop_multiplier = stop_multiplier
        self._quant_func = quant_func
        self._error_func = error_func
        self._losses_sum = [None] * num_steps
        self._losses_count = [0] * num_steps
        self._candidate_amaxs = [None] * num_steps

        self._amax = None

    @torch.no_grad()
    def collect(self, x: torch.Tensor):
        """Collect input tensor statistics and compute losses for MSE calibration.

        Args:
            x: Input tensor.
        """
        if self._quant_func is None:
            raise RuntimeError(
                "Quantization function not set. Msecalibrator requires a quant_func to be provided."
            )

        x = x.detach().to(dtype=torch.float32)

        device = x.device
        multipliers = torch.linspace(
            self._start_multiplier, self._stop_multiplier, steps=self._num_steps, device=device
        )

        # Get reduce axis for per-channel quantization
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)

        for step, multiplier in enumerate(multipliers):
            candidate_amax = self._initial_amax * multiplier
            xq = self._quant_func(x, candidate_amax)

            if self._error_func is not None:
                error = self._error_func(x, xq)
            else:
                error = F.mse_loss(x, xq, reduction="none")

            if reduce_axis is None:
                loss = torch.sum(error)
            else:
                loss = quant_utils.reduce_sum(error, axis=reduce_axis, keepdims=False)

            if self._candidate_amaxs[step] is None:
                self._candidate_amaxs[step] = candidate_amax

            if self._losses_sum[step] is None:
                self._losses_sum[step] = loss.clone()
            else:
                self._losses_sum[step] += loss
            self._losses_count[step] += 1

    def reset(self):
        """Reset the stored losses and amax value."""
        self._losses_sum = [None] * self._num_steps
        self._losses_count = [0] * self._num_steps
        self._candidate_amaxs = [None] * self._num_steps
        self._amax = None

    @torch.no_grad()
    def compute_amax(self, verbose: bool = False):
        """Return the amax value that minimizes quantization error.

        Args:
            verbose: If True, print the ratio of best_amax to initial_amax.
        """
        if not any(loss_sum is not None for loss_sum in self._losses_sum):
            return None

        # Check if this is per-tensor or per-channel based on the first loss
        first_loss_sum = None
        for loss_sum in self._losses_sum:
            if loss_sum is not None:
                first_loss_sum = loss_sum
                break

        if first_loss_sum is None:
            return None

        if first_loss_sum.ndim == 0:
            avg_losses = []
            for step in range(self._num_steps):
                if self._losses_sum[step] is not None and self._losses_count[step] > 0:
                    avg_loss = self._losses_sum[step] / self._losses_count[step]
                    avg_losses.append(avg_loss)
                else:
                    avg_losses.append(torch.tensor(float("inf"), device=first_loss_sum.device))

            avg_losses = torch.stack(avg_losses)
            best_step = torch.argmin(avg_losses).item()
            self._amax = self._candidate_amaxs[best_step]

            if verbose:
                ratio = (self._amax / self._initial_amax).item()
                print(f"MSE Calibrator: best_amax/initial_amax ratio = {ratio:.4f}")

        else:
            # Per-channel case: loss is a tensor with shape (num_channels,)
            # Compute average losses for each step: [num_steps, num_channels]
            avg_losses_per_step = []
            for step in range(self._num_steps):
                if self._losses_sum[step] is not None and self._losses_count[step] > 0:
                    avg_loss = self._losses_sum[step] / self._losses_count[step]
                    avg_losses_per_step.append(avg_loss)
                else:
                    # No data for this step, use inf
                    avg_losses_per_step.append(torch.full_like(first_loss_sum, float("inf")))

            # Stack to get [num_steps, num_channels]
            avg_losses_per_step = torch.stack(avg_losses_per_step)

            best_steps = torch.argmin(avg_losses_per_step, dim=0)
            # Stack candidate amaxs: [num_steps, num_channels]
            candidate_amaxs = torch.stack(self._candidate_amaxs)

            num_channels = best_steps.shape[0]
            self._amax = candidate_amaxs[
                best_steps, torch.arange(num_channels, device=best_steps.device)
            ]
            if self._amax is not None:
                self._amax = self._amax.reshape(self._initial_amax.shape)

            if verbose:
                ratio = self._amax / self._initial_amax
                print(
                    f"MSE Calibrator: best_amax/initial_amax ratio - "
                    f"mean: {ratio.mean().item():.4f}, "
                    f"min: {ratio.min().item():.4f}, "
                    f"max: {ratio.max().item():.4f}"
                )

        return self._amax
