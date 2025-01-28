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

"""High-level tests for real weight-only quantization."""

import pytest
from _test_utils.torch_quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear
from _test_utils.torch_quantization.quant_utils import get_model_size
from _test_utils.torch_quantization.quantize_common import save_restore_test

import modelopt.torch.quantization as mtq


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.NF4_REAL_QUANT_CFG,
        mtq.INT4_AWQ_REAL_QUANT_CFG,
    ],
)
def test_real_quantize(model_cls, config):
    """Test quantize function can run without problems."""
    # update config to fit test cases
    if config == mtq.NF4_REAL_QUANT_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {
            -1: 16,
            "scale_bits": 8,
            "scale_block_sizes": {-1: 16},
        }
    if config == mtq.INT4_AWQ_REAL_QUANT_CFG:
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 16}

    # PTQ
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    fake_quant_mem = get_model_size(model)
    mtq.quantize(model, config, forward_loop)
    real_quant_mem = get_model_size(model)

    # check memory usage
    assert fake_quant_mem > real_quant_mem, "Memory after real quantization is not reduced."

    # test forward
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model):
        for batch in calib_data:
            model(batch)

    # make sure we can run the real quantized model
    forward_loop(model)


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.NF4_REAL_QUANT_CFG,
        mtq.INT4_AWQ_REAL_QUANT_CFG,
    ],
)
def test_save_restore(model_cls, config):
    # update config to fit test cases
    if config == mtq.NF4_REAL_QUANT_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {
            -1: 16,
            "scale_bits": 8,
            "scale_block_sizes": {-1: 16},
        }
    if config == mtq.INT4_AWQ_REAL_QUANT_CFG:
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 16}
    save_restore_test(model_cls, "cuda", config)
