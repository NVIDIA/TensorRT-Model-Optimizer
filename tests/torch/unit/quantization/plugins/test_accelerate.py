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

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantLinearConvBase

try:
    from accelerate.hooks import ModelHook, add_hook_to_module
except ImportError:
    pytest.skip("accelerate not available", allow_module_level=True)


def test_linear_with_accelerate_monkey_patched_forward():
    module_test = nn.Linear(16, 16)
    add_hook_to_module(module_test, ModelHook())

    mtq.replace_quant_module(module_test)
    assert module_test._old_forward.__func__ == QuantLinearConvBase.forward

    module_test.input_quantizer.enable_calib()
    module_test.weight_quantizer.enable_calib()

    module_ref = nn.Linear(16, 16)
    mtq.replace_quant_module(module_ref)

    module_ref.load_state_dict(module_test.state_dict())

    x = torch.randn(1, 16)
    out1 = module_test(x)
    out2 = module_ref(x)
    assert torch.allclose(out1, out2)

    module_test.input_quantizer.load_calib_amax()
    module_test.weight_quantizer.load_calib_amax()

    assert module_test.input_quantizer.amax is not None
    assert module_test.weight_quantizer.amax is not None
