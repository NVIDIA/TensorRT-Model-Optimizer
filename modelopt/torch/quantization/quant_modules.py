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

"""Deprecated. Placeholder module for throwing deprecated error."""

from contextlib import contextmanager

from modelopt.torch.utils import DeprecatedError


def initialize(*args, **kwargs):
    """Deprecated. This API is no longer supported.

    Use :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`
    instead to quantize the model.
    """
    raise DeprecatedError(
        "This API is no longer supported. Use `modelopt.torch.quantization.model_quant.quantize()`"
        " API instead to quantize the model."
    )


def deactivate():
    """Deprecated. This API is no longer supported."""
    raise DeprecatedError(
        "This API is no longer supported. Use "
        "`modelopt.torch.quantization.model_quant.disable_quantizer()` API instead "
        "to disable quantization."
    )


@contextmanager
def enable_onnx_export():
    """Deprecated. You no longer need to use this context manager while exporting to ONNX."""
    raise DeprecatedError(
        "You no longer need to use this context manager while exporting to ONNX! please call"
        " `torch.onnx.export` directly."
    )
