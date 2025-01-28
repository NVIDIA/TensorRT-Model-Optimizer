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

"""Utilities for optimization."""

from typing import Generator, Optional

import torch.nn as nn

from modelopt.torch.utils import unwrap_model

from .dynamic import DynamicModule, DynamicSpace
from .hparam import Hparam

__all__ = [
    "get_hparam",
    "is_configurable",
    "is_dynamic",
    "named_dynamic_modules",
    "named_hparams",
    "search_space_size",
]


class _DynamicSpaceUnwrapped(DynamicSpace):
    """A wrapper for the DynamicSpace class to handle unwrapping of model wrappers like DDP.

    This is useful to ensure that configurations are valid for both vanilla models and wrapped
    models, see :meth:`unwrap_models<modelopt.torch.utils.network.unwrap_model>` for supported wrappers.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(unwrap_model(model))


def is_configurable(model: nn.Module) -> bool:
    """Check if the model is configurable."""
    return _DynamicSpaceUnwrapped(model).is_configurable()


def is_dynamic(model: nn.Module) -> bool:
    """Check if the model is dynamic."""
    return _DynamicSpaceUnwrapped(model).is_dynamic()


def named_hparams(
    model: nn.Module, configurable: Optional[bool] = None
) -> Generator[tuple[str, Hparam], None, None]:
    """Recursively yield the name and instance of *all* hparams."""
    yield from _DynamicSpaceUnwrapped(model).named_hparams(configurable)


def named_dynamic_modules(model: nn.Module) -> Generator[tuple[str, DynamicModule], None, None]:
    """Recursively yield the name and instance of *all* dynamic modules."""
    yield from _DynamicSpaceUnwrapped(model).named_dynamic_modules()


def get_hparam(model: nn.Module, name: str) -> Hparam:
    """Get the hparam with the given name."""
    return _DynamicSpaceUnwrapped(model).get_hparam(name)


def search_space_size(model: nn.Module) -> int:
    """Return the size of the search space."""
    return _DynamicSpaceUnwrapped(model).size()
