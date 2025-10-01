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

"""Registry and base class for sparse attention methods."""

from abc import ABC, abstractmethod

import torch


class SparseAttentionMethod(ABC):
    """Base class for sparse attention methods.

    All sparse attention methods should inherit from this class and
    implement the required abstract methods.
    """

    @abstractmethod
    def apply_sparsity(
        self,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attention_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Apply sparsity to attention computation.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_scores: Pre-computed attention scores

        Returns:
            Tuple of (query, key, value, attention_scores) with sparsity applied
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name identifier."""


# Method Registry
_SPARSE_ATTENTION_METHODS: dict[str, type[SparseAttentionMethod]] = {}


def register_sparse_method(name: str):
    """Decorator to register sparse attention methods.

    Args:
        name: Method name to register

    Example:
        @register_sparse_method("my_method")
        class MyMethod(SparseAttentionMethod):
            ...
    """

    def decorator(cls: type[SparseAttentionMethod]):
        if name in _SPARSE_ATTENTION_METHODS:
            print(f"Warning: Overriding existing sparse attention method: {name}")
        _SPARSE_ATTENTION_METHODS[name] = cls
        return cls

    return decorator


def get_sparse_method(name: str) -> type[SparseAttentionMethod]:
    """Get sparse attention method by name.

    Args:
        name: Method name to retrieve

    Returns:
        Method class

    Raises:
        ValueError: If method name is not registered
    """
    if name not in _SPARSE_ATTENTION_METHODS:
        available = list(_SPARSE_ATTENTION_METHODS.keys())
        raise ValueError(f"Unknown sparse attention method: {name}. Available methods: {available}")
    return _SPARSE_ATTENTION_METHODS[name]
