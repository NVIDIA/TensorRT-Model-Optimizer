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


def skip_if_no_tensorrt():
    from modelopt.onnx.quantization.ort_utils import _check_for_tensorrt

    try:
        _check_for_tensorrt()
    except (AssertionError, ImportError) as e:
        pytest.skip(f"{e}", allow_module_level=True)


def skip_if_no_trtexec():
    if not shutil.which("trtexec"):
        pytest.skip("trtexec cmdline tool is not available", allow_module_level=True)


def skip_if_no_libcudnn():
    from modelopt.onnx.quantization.ort_utils import _check_for_libcudnn

    try:
        _check_for_libcudnn()
    except FileNotFoundError as e:
        pytest.skip(f"{e}!", allow_module_level=True)


def skip_if_no_megatron(apex_or_te_required: bool = False, mamba_required: bool = False):
    try:
        import megatron  # noqa: F401
    except ImportError:
        pytest.skip("megatron not available", allow_module_level=True)

    try:
        import apex  # noqa: F401

        has_apex = True
    except ImportError:
        has_apex = False

    try:
        import transformer_engine  # noqa: F401

        has_te = True
    except ImportError:
        has_te = False

    try:
        import mamba_ssm  # noqa: F401

        has_mamba = True
    except ImportError:
        has_mamba = False

    if apex_or_te_required and not has_apex and not has_te:
        pytest.skip("Apex or TE required for Megatron test", allow_module_level=True)

    if mamba_required and not has_mamba:
        pytest.skip("Mamba required for Megatron test", allow_module_level=True)
