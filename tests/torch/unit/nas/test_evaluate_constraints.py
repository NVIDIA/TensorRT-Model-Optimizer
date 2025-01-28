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
from torchprofile import profile_macs
from torchvision.models import mobilenet_v2, resnet18

from modelopt.torch.nas._algorithms import ConstraintsFunc
from modelopt.torch.utils import get_module_device, param_num, remove_bn

try:
    from _test_utils.torch_deploy.runtime import FAKE_DEPLOYMENT, fake_latency

    SKIP_LATENCY_TEST = False
except ImportError:
    SKIP_LATENCY_TEST = True


@pytest.mark.parametrize("model_cls", [mobilenet_v2, resnet18])
def test_evaluate_constraints(model_cls):
    model = model_cls().eval()

    constraints = {"flops": float("inf"), "params": float("inf")}
    dummy_input = torch.randn(1, 3, 224, 224)
    cf = ConstraintsFunc(model, constraints=constraints, dummy_input=dummy_input)
    _, actual_results = cf()

    remove_bn(model)
    args_profile = (torch.zeros(dummy_input.shape, device=get_module_device(model)),)
    expected_results = {
        # NOTE: using param_num here instead of param_num_from_forward to check
        # correctness of the function.
        "params": param_num(model, unit=1.0),
        "flops": profile_macs(model, args=args_profile) / 1.0,
    }

    assert actual_results == expected_results


@pytest.mark.skipif(SKIP_LATENCY_TEST, reason="modelopt.torch._deploy dependencies not available")
def test_evaluate_latency_constraints():
    model = resnet18().eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    with fake_latency(100):
        cf = ConstraintsFunc(
            model,
            constraints={"latency": 200},
            dummy_input=dummy_input,
            deployment=FAKE_DEPLOYMENT,
        )
        is_satisfied, _vals = cf()
        assert is_satisfied is True

        cf = ConstraintsFunc(
            model, constraints={"latency": 50}, dummy_input=dummy_input, deployment=FAKE_DEPLOYMENT
        )
        is_satisfied, _vals = cf()
        assert is_satisfied is False


@pytest.mark.parametrize("model_cls", [resnet18])
def test_percent_limits(model_cls):
    model = model_cls().eval()

    constraints = {"flops": "80%", "params": "40%"}
    dummy_input = torch.randn(1, 3, 224, 224)
    cf = ConstraintsFunc(model, constraints=constraints, dummy_input=dummy_input)

    remove_bn(model)
    args_profile = (torch.zeros(dummy_input.shape, device=get_module_device(model)),)
    max_flops = profile_macs(model, args=args_profile)
    max_params = param_num(model, unit=1.0)
    expected_results = {
        # NOTE: using trainable_param_num here instead of trainable_param_num_from_forward to check
        # correctness of the function.
        "flops": (0, max_flops * 0.8),
        "params": (0, max_params * 0.4),
    }
    for k in cf.limits:
        assert expected_results[k] == pytest.approx(cf.limits[k])

    # check latency
    if SKIP_LATENCY_TEST:
        return
    max_latency = 200
    with fake_latency(max_latency):
        constraints = {"flops": "80%", "params": "40%", "latency": "20%"}
        cf = ConstraintsFunc(
            model, constraints=constraints, dummy_input=dummy_input, deployment=FAKE_DEPLOYMENT
        )

        assert cf.limits["latency"] == pytest.approx((0, max_latency * 0.2))
