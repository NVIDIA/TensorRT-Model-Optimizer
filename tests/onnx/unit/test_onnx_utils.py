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

import os

import pytest

from modelopt.onnx.utils import save_onnx_bytes_to_dir, validate_onnx


@pytest.mark.parametrize(
    "onnx_bytes",
    [None, b"", b"invalid onnx"],
)
def test_validate_onnx(onnx_bytes):
    assert not validate_onnx(onnx_bytes)


def test_save_onnx(tmp_path):
    save_onnx_bytes_to_dir(b"test_onnx_bytes", tmp_path, "test")
    assert os.path.exists(os.path.join(tmp_path, "test.onnx"))
