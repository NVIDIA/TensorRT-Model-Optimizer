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

"""Nvidia TensorRT Model Optimizer (modelopt)."""

import sys as _sys
import warnings as _warnings
from importlib.metadata import version as _version

__version__ = _version("nvidia-modelopt")


try:
    # Import from local source if available
    from . import core

    __core_version__ = __version__
except ImportError:
    # Import from nvidia-modelopt-core wheel installation
    import modelopt_core as core  # type: ignore[no-redef]

    _sys.modules["modelopt.core"] = core
    __core_version__ = _version("nvidia-modelopt-core")

# Versions need to be the same for compatibility
if __version__.split(".")[:2] != __core_version__.split(".")[:2]:
    _warnings.warn(
        f"Version mismatch between nvidia-modelopt ({__version__}) and nvidia-modelopt-core"
        f" ({__core_version__}). Please ensure both versions are the same for compatibility."
    )
