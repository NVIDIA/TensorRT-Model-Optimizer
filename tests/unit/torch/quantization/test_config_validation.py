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

"""Test of quantization config validations."""

from contextlib import nullcontext

import pytest

from modelopt.torch.quantization.config import (
    FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    FP8_DEFAULT_CFG,
    FP8_PER_CHANNEL_PER_TOKEN_CFG,
    INT4_AWQ_CFG,
    NVFP4_DEFAULT_CFG,
    W4A8_AWQ_BETA_CFG,
    QuantizeConfig,
    need_calibration,
)


@pytest.mark.parametrize("q_rotate", [True, False])
@pytest.mark.parametrize("k_rotate", [True, False])
def test_qk_rotation(q_rotate, k_rotate):
    config = {
        "quant_cfg": {
            "*q_bmm_quantizer": {
                "rotate": q_rotate,
            },
            "*k_bmm_quantizer": {
                "rotate": k_rotate,
            },
        }
    }

    with nullcontext() if q_rotate == k_rotate else pytest.raises(Exception):
        QuantizeConfig.model_validate(config)


def test_need_calibration():
    assert need_calibration(FP8_DEFAULT_CFG)
    assert not need_calibration(FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert not need_calibration(FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert need_calibration(INT4_AWQ_CFG)
    assert need_calibration(W4A8_AWQ_BETA_CFG)
    assert need_calibration(NVFP4_DEFAULT_CFG)
