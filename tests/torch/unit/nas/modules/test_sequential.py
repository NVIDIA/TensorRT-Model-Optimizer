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

from typing import OrderedDict

import pytest
import torch
from torch import nn
from torchvision.models.mobilenetv2 import InvertedResidual

from modelopt.torch.nas.registry import DMRegistry


class ModuleContainerWrapper:
    def __init__(self, modules: OrderedDict) -> None:
        self._modules = modules

    def items(self) -> any:
        return self._modules.items()


@pytest.mark.parametrize(
    "in_features, depth, desired_depth",
    [
        (2, 4, 2),
        (4, 3, 3),
        (3, 1, 1),
    ],
)
def test_dynamic_sequential(in_features, depth, desired_depth):
    layers = [nn.Linear(in_features + d, in_features + d + 1) for d in range(depth)]

    # initialize seq and dynamic_seq
    seq = nn.Sequential(*layers[:desired_depth])
    dynamic_seq = DMRegistry.convert(nn.Sequential(*layers))

    # set depth
    def _set_depth(d):
        hp = dynamic_seq.get_hparam("depth")
        dynamic_seq.depth = d
        assert hp.active == d, f"depth was not set correctly: {hp.active} != {d}"

    # check that we cannot set wrong depth, then set correct depth
    with pytest.raises(AssertionError):
        _set_depth(depth + 2)
    _set_depth(desired_depth)

    # check printing
    print(dynamic_seq)

    # check inference
    input = torch.rand(1, in_features)
    out = seq(input)
    out_dynamic = dynamic_seq(input)
    assert torch.allclose(out, out_dynamic)

    # check that state_dict of dynamic_seq always contains the full set of keys
    expected_keys = [f"{idx}.{name}" for idx in range(depth) for name in ["weight", "bias"]]
    assert expected_keys == list(dynamic_seq.state_dict().keys())

    # check that we can limit depth choices correctly
    _set_depth(desired_depth)
    dynamic_seq.modify(min_depth=desired_depth)
    hp = dynamic_seq.get_hparam("depth")
    assert hp.choices == list(range(desired_depth, depth + 1))

    # check export
    _set_depth(desired_depth)
    exported_seq = dynamic_seq.export()
    assert type(exported_seq) is nn.Sequential
    assert len(exported_seq) == desired_depth

    # check that outputs agree
    out_exported = exported_seq(input)
    assert torch.allclose(out, out_exported)


def test_contained_dynamic_module():
    in_features = 16
    expand_ratio = 2
    depth = 4
    desired_depth = 2

    layers = [InvertedResidual(in_features, in_features, 1, expand_ratio) for d in range(depth)]

    # initialize dynamic_seq
    dynamic_seq = DMRegistry.convert(nn.Sequential(*layers))
    input = torch.rand(1, in_features, 32, 32)

    dynamic_seq.depth = desired_depth
    out_dynamic = dynamic_seq(input)

    seq = dynamic_seq.export()
    out = seq(input)
    assert torch.allclose(out, out_dynamic)
