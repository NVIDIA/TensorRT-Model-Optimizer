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
"""Forward hooks for activation-based importance estimation in NAS plugins."""

from abc import ABC, abstractmethod

import torch
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from torch import nn


class ForwardHook(ABC):
    """Base class for PyTorch forward hooks.

    This follows the PyTorch forward hook API where the second
    parameter is 'args' (a tuple of positional arguments passed to forward()).

    Usage:
        hook = MyHook()
        module.register_forward_hook(hook)
    """

    @abstractmethod
    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Forward hook that is called after the module's forward pass.

        Args:
            module: The module this hook is registered on
            args: Tuple of positional arguments passed to module.forward()
            output: The output from module.forward()

        Returns:
            None (does not modify the output)
        """
        ...


class L2NormHook(ForwardHook):
    """Hook for accumulating activation statistics for importance estimation.

    Activations are computed as mean over seq_len and then squared and summed over batch_size.
    In the accumulate() method we take the square root of the sum to get the L2 norm.

    Args:
        max_size: Optional maximum expected size to validate against (skips if mismatch).
                Useful for skipping non-max subnets during profiling.
    """

    def __init__(self, max_size: int | None = None):
        """Initialize the L2NormHook."""
        self.max_size = max_size
        self._activations: torch.Tensor | None = None

    def __call__(
        self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Accumulate activation statistics from the forward pass."""
        # Gather input [seq_len, batch_size, hidden_size] over all TP regions
        # NOTE: This is not used at the moment since we restrict to TP=1
        input_tensor = gather_from_tensor_model_parallel_region(args[0]).detach()

        # Dont aggregate activations from non-max subnets (e.g. from profiling)
        if self.max_size is not None and input_tensor.shape[-1] != self.max_size:
            return

        input_tensor = input_tensor.to(torch.float32)  # use full precision to avoid overflow
        activations = input_tensor.abs().mean(dim=0)  # [batch_size, hidden_size]
        activations = activations.pow(2).sum(dim=0)  # [hidden_size]

        if self._activations is None:
            self._activations = activations
        else:
            self._activations += activations

    def accumulate(self) -> torch.Tensor:
        """Return the accumulated L2 norm of activations.

        Returns:
            Tensor of accumulated scores, one per channel

        Raises:
            AssertionError: If no activations have been collected yet
        """
        assert self._activations is not None, "No activations collected for importance estimation."
        # Convert squared sum to L2 norm
        return self._activations.pow(0.5)
