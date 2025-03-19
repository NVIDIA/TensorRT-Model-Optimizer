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

import shutil

import pytest
import torch
from packaging.version import Version

from modelopt.onnx.quantization.ort_utils import _check_for_libcudnn, _check_for_tensorrt


def skip_if_no_tensorrt():
    try:
        _check_for_tensorrt()
    except (AssertionError, ImportError) as e:
        pytest.skip(f"{e}", allow_module_level=True)


def skip_if_no_trtexec():
    if not shutil.which("trtexec"):
        pytest.skip("trtexec cmdline tool is not available", allow_module_level=True)


def skip_if_no_libcudnn():
    try:
        _check_for_libcudnn()
    except FileNotFoundError as e:
        pytest.skip(f"{e}!", allow_module_level=True)


def skip_if_no_megatron(apex_or_te_required: bool = False):
    try:
        import megatron  # noqa: F401
    except ImportError:
        pytest.skip("megatron not available", allow_module_level=True)

    try:
        import apex  # noqa: F401

        HAS_APEX = True  # noqa: N806
    except ImportError:
        HAS_APEX = False  # noqa: N806

    try:
        import transformer_engine  # noqa: F401

        HAS_TE = True  # noqa: N806
    except ImportError:
        HAS_TE = False  # noqa: N806

    if apex_or_te_required and not HAS_APEX and not HAS_TE:
        pytest.skip("Apex or TE required for Megatron test", allow_module_level=True)


def skip_if_mcore_dist_ckpt_is_not_supported():
    from megatron.core import __version__ as mcore_version

    if Version(mcore_version) <= Version("0.8") and Version(torch.__version__) >= Version("2.4"):
        pytest.skip("Megatron Core 0.9+ is required for dist checkpointing with torch < 2.4")
