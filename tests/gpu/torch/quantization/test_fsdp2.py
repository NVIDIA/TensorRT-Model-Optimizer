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

"""Test of quantization with FSDP2."""

import copy

import pytest
import torch
import torch.nn as nn
from _test_utils.torch_dist.dist_utils import (
    get_device_counts,
    spawn_multiprocess_job,
    synchronize_state_dict,
)
from packaging.version import Version

if Version(torch.__version__) < Version("2.4"):
    pytest.skip("FSDP2 is only supported in torch >= 2.4", allow_module_level=True)

from torch.distributed._composable.fsdp.fully_shard import fully_shard

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.dynamic import _pytorch_managed


def _test_fsdp2_simple_linear(rank, size):
    model = nn.Linear(128, 128).cuda(rank)
    inputs = torch.randn(2, 2, 128).cuda(rank)

    synchronize_state_dict(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    manager = model._get_dm_attribute_manager()
    assert "weight" in manager.da_keys()
    assert model._get_dm_attribute_manager().get_da_value("weight") is _pytorch_managed

    out_ref = model(inputs)

    fsdp_model = fully_shard(model)
    assert "weight" in manager.da_keys()
    out_test = fsdp_model(inputs)

    assert torch.allclose(out_ref, out_test)


def _test_netsted_fsdp2_backward(rank, size):
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
        nn.Linear(128, 128),
    ).cuda()
    inputs = torch.randn(2, 2, 128).cuda()
    inputss = inputs.detach().clone()
    synchronize_state_dict(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))
    fsdp_model = copy.deepcopy(model)

    optimizer_ref = torch.optim.SGD(model.parameters(), lr=0.1)
    out_ref = model(inputs)
    out_ref.sum().backward()

    fully_shard(fsdp_model[0])
    fully_shard(fsdp_model[1])
    fsdp_model = fully_shard(fsdp_model)

    optimizer_test = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    out_test = fsdp_model(inputs)
    out_test.sum().backward()

    assert torch.allclose(out_ref, out_test)

    optimizer_ref.step()
    optimizer_ref.zero_grad()

    optimizer_test.step()
    optimizer_test.zero_grad()

    out_ref_1 = model(inputss)
    out_test_1 = fsdp_model(inputss)
    assert torch.allclose(out_ref_1, out_test_1, rtol=1e-4)


@pytest.mark.parametrize("device_count", get_device_counts())
def test_fsdp_simple_linear(device_count):
    spawn_multiprocess_job(size=device_count, job=_test_fsdp2_simple_linear, backend="nccl")


@pytest.mark.parametrize("device_count", get_device_counts())
def test_nested_fsdp2_backward(device_count):
    spawn_multiprocess_job(size=device_count, job=_test_netsted_fsdp2_backward, backend="nccl")
