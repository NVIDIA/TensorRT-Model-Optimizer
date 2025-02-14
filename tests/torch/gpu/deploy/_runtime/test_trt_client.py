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
from _test_utils.import_helper import skip_if_no_trtexec
from _test_utils.torch_misc import compare_outputs
from _test_utils.torch_model.benchmark_models import _process_model_and_inputs, get_benchmark_models
from _test_utils.torch_model.deploy_models import (
    LeNet5,
    LeNet5Ooo,
    LeNet5TwoInputs,
    LeNet5TwoOutputs,
)

skip_if_no_trtexec()


from modelopt.torch._deploy._runtime.trt_client import TRTLocalClient
from modelopt.torch._deploy.compilation import compile

deployment = {
    "runtime": "TRT",
    "accelerator": "GPU",
    "version": "8.6",
    "precision": "fp16",
    "onnx_opset": "14",
}

benchmarks = get_benchmark_models()


@pytest.mark.parametrize(
    "model, model_inputs",
    [
        (LeNet5(), {"args": (LeNet5().gen_input(),), "kwargs": {}}),
        (
            LeNet5TwoInputs(),
            {
                "args": (LeNet5TwoInputs().gen_input(0), LeNet5TwoInputs().gen_input(1)),
                "kwargs": {},
            },
        ),
        (LeNet5TwoOutputs(), {"args": (LeNet5TwoOutputs().gen_input(),), "kwargs": {}}),
        (LeNet5Ooo(), {"args": (LeNet5Ooo().gen_input(0), LeNet5Ooo().gen_input(1)), "kwargs": {}}),
    ],
)
def test_compile_and_profile_lenet5(model, model_inputs):
    model, args, kwargs = _process_model_and_inputs(model, model_inputs, on_gpu=True)
    _compile_and_profile(model, args, kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("get_model_and_input", benchmarks.values(), ids=benchmarks.keys())
def test_compile_and_profile_benchmark_models(get_model_and_input):
    model, args, kwargs = get_model_and_input(on_gpu=True)
    _compile_and_profile(model, args, kwargs)


def _compile_and_profile(model, args, kwargs):
    device_model = compile(model, (*args, kwargs), deployment)
    assert isinstance(
        device_model.client, TRTLocalClient
    ), "device model client is not an instance of TRTLocalClient"
    latency, detailed_results = device_model.profile()
    device_model_outputs = device_model(*args, **kwargs)
    torch_model_ouptuts = model(*args, **kwargs)
    assert latency > 0.0
    assert detailed_results is not None
    assert device_model is not None
    compare_outputs(device_model_outputs, torch_model_ouptuts, rtol=1e-2, atol=1e-1)
