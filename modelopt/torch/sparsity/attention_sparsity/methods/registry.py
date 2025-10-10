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
    """Base class for sparse attention methods."""

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


# Method Registry with versioning support
_SPARSE_ATTENTION_METHODS: dict[str, dict[str, type[SparseAttentionMethod]]] = {}


def register_sparse_method(name: str, version: str = "v1"):
    """Decorator to register sparse attention methods with version support.

    Args:
        name: Method name to register
        version: Version string (default: "v1")

    Example:
        @register_sparse_method("my_method", version="v3")
        class MyMethodV3(SparseAttentionMethod):
            ...
    """

    def decorator(cls: type[SparseAttentionMethod]):
        if name not in _SPARSE_ATTENTION_METHODS:
            _SPARSE_ATTENTION_METHODS[name] = {}

        if version in _SPARSE_ATTENTION_METHODS[name]:
            import warnings

            warnings.warn(
                f"Overriding existing sparse attention method: {name}@{version}",
                RuntimeWarning,
                stacklevel=2,
            )

        _SPARSE_ATTENTION_METHODS[name][version] = cls
        return cls

    return decorator


def get_sparse_method(name: str, version: str | None = None) -> type[SparseAttentionMethod]:
    """Get sparse attention method by name and optional version.

    Args:
        name: Method name to retrieve
        version: Optional version string. If None, uses latest version.

    Returns:
        Method class

    Raises:
        ValueError: If method name or version is not registered

    Example:
        >>> get_sparse_method("flash_softmax_skip")  # Latest version
        >>> get_sparse_method("flash_softmax_skip", "v1")  # Specific version
    """
    if name not in _SPARSE_ATTENTION_METHODS:
        available = list(_SPARSE_ATTENTION_METHODS.keys())
        raise ValueError(f"Unknown sparse attention method: {name}. Available: {available}")

    method_versions = _SPARSE_ATTENTION_METHODS[name]

    if not version:
        version = sorted(method_versions.keys())[-1]

    if version not in method_versions:
        available_versions = list(method_versions.keys())
        raise ValueError(
            f"Unknown version {version} for method {name}. Available: {available_versions}"
        )

    return method_versions[version]
