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

"""Utility functions for PyTorch tensors."""

from collections import abc

import numpy as np
import torch

__all__ = ["numpy_to_torch", "torch_detach", "torch_to", "torch_to_numpy"]


def torch_to(data, *args, **kwargs):
    """Try to recursively move the data to the specified args/kwargs."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_to(val, *args, **kwargs) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_to(val, *args, **kwargs) for k, val in data.items()}
    return data


def torch_detach(data):
    """Try to recursively detach the data from the computation graph."""
    if isinstance(data, torch.Tensor):
        return torch.detach(data)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_detach(val) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_detach(val) for k, val in data.items()}
    return data


def torch_to_numpy(inputs: list[torch.Tensor]) -> list[np.ndarray]:
    """Convert torch tensors to numpy arrays."""
    return [t.detach().cpu().numpy() for t in inputs]


def numpy_to_torch(np_outputs: list[np.ndarray]) -> list[torch.Tensor]:
    """Convert numpy arrays to torch tensors."""
    return [torch.from_numpy(arr) for arr in np_outputs]
