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

import fnmatch

import pytest
from _test_utils.torch_quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear
from _test_utils.torch_quantization.quant_utils import get_model_size
from _test_utils.torch_quantization.quantize_common import save_restore_test

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.qtensor import QTensorWrapper


@pytest.mark.parametrize("model_cls", [SimpleLinear, SimpleConv, SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT4_AWQ_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    ],
)
def test_real_quantize(model_cls, config):
    """Test quantize function can run without problems."""
    # update config to fit test cases
    if config == mtq.INT4_AWQ_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {
            -1: 16,
            "scale_bits": 8,
            "scale_block_sizes": {-1: 16},
        }

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
    mtq.compress(model)
    real_quant_mem = get_model_size(model)

    # check memory usage
    if config != mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG:  # FP8 block may pad the weights
        assert fake_quant_mem > real_quant_mem, "Memory after real quantization is not reduced."

    # test forward
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model):
        for batch in calib_data:
            model(batch)

    # make sure we can run the real quantized model
    forward_loop(model)


@pytest.mark.parametrize("model_cls", [SimpleConvLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT4_AWQ_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    ],
)
def test_save_restore(model_cls, config):
    # update config to fit test cases
    if config == mtq.INT4_AWQ_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {
            -1: 16,
            "scale_bits": 8,
            "scale_block_sizes": {-1: 16},
        }

    save_restore_test(model_cls, "cuda", config, compress=True)


@pytest.mark.parametrize("model_cls", [SimpleConvLinear])
@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.INT4_AWQ_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    ],
)
@pytest.mark.parametrize(
    "compress_config",
    [
        {
            "*net.0*": False,
            "*.conv1*": False,
            "*.fc1*": False,
            "default": True,
        },  # Skip first layers
    ],
)
def test_compress_config(model_cls, quant_config, compress_config):
    """Test model compression with different configurations.

    Args:
        model_cls: Model class to test
        quant_config: Quantization configuration
        compress_config: Compression pattern configuration
    """
    model = model_cls().cuda()

    # Store original weights and get expected skip list
    original_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            original_weights[name] = module.weight.data.clone()

    # Quantize and compress the model
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, quant_config, forward_loop)
    mtq.compress(model, compress_config=compress_config)

    # Verify compression based on config
    for name, module in model.named_modules():
        if not hasattr(module, "weight_quantizer") or not hasattr(module, "weight"):
            continue

        # Check if layer should be skipped based on patterns
        should_compress = compress_config["default"] if "default" in compress_config else False
        for pattern in compress_config:
            if pattern == "default":
                continue
            if fnmatch.fnmatch(name, pattern):
                should_compress = compress_config[pattern]
        if should_compress:
            assert isinstance(module.weight, QTensorWrapper)
        else:
            assert not isinstance(module.weight, QTensorWrapper)

    # Verify model functionality
    input_tensor = model_cls.get_input().cuda()
    output = model(input_tensor)
    assert output.shape[0] == input_tensor.shape[0], "Model forward pass failed"
