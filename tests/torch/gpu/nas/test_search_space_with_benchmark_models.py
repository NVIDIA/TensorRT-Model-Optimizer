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

import copy

import pytest
import torch
from _test_utils.torch_misc import compare_outputs, set_seed
from _test_utils.torch_model.benchmark_models import get_benchmark_models

from modelopt.torch.nas.search_space import generate_search_space
from modelopt.torch.utils import flatten_tree, zero_grad
from modelopt.torch.utils.random import _set_deterministic_seed

benchmarks = get_benchmark_models()


@pytest.mark.parametrize("on_gpu", [True])  # just run on GPU, but leave it here for easy debugging
@pytest.mark.parametrize("get_model_and_input", benchmarks.values(), ids=benchmarks.keys())
def test_benchmarks(get_model_and_input, on_gpu):
    # initialize model
    model, args, kwargs = get_model_and_input(on_gpu)

    # Ensure original model is correct
    with torch.no_grad():
        _ = model(*args, **kwargs)

    # set deterministic seed
    _set_deterministic_seed()

    for _ in range(5):
        # Test search space generation
        # NOTE: use deepcopy as we don't wanna test repeated search space gen & export on same model
        search_space = generate_search_space(copy.deepcopy(model))

        # Test sample process which contains uncertainty
        search_space.sample()

        # Test subnet forward
        out1 = search_space.model(*args, **kwargs)

        # Test subnet backward
        with torch.autograd.set_detect_anomaly(True):
            for t in flatten_tree(out1)[0]:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    torch.sum(t).backward()
            zero_grad(search_space.model)

        # Test model export
        subnet = search_space.export()
        out2 = subnet(*args, **kwargs)

        # Test output match
        compare_outputs(out1, out2, atol=1e-5, rtol=1e-5)


# NOTE: we run this test on CPU because of better floating point precision!
@pytest.mark.parametrize("on_gpu", [False])  # don't run on GPU but leave it here for easy debugging
@pytest.mark.parametrize("get_model_and_input", benchmarks.values(), ids=benchmarks.keys())
def test_dynamic_sorting(get_model_and_input, on_gpu):
    set_seed()
    # initialize model
    model, args, kwargs = get_model_and_input(on_gpu)

    # Get the output of the original model
    with torch.no_grad():
        y1 = model(*args, **kwargs)

    # generate search space and sort params
    search_space = generate_search_space(model)
    search_space.sort_parameters()

    # sanity check if the model functionality is preserved after sorting
    with torch.no_grad():
        y2 = model(*args, **kwargs)

    # check if the inference results after sorting is the same
    compare_outputs(y1, y2, rtol=1e-4, atol=1e-5)
