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

import json
import sys
from contextlib import nullcontext

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from _test_utils.torch_model.deploy_models import BaseDeployModel, get_deploy_models
from _test_utils.torch_model.vision_models import get_tiny_resnet_and_input
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from packaging.version import Version

from modelopt.onnx.utils import (
    get_batch_size_from_bytes,
    get_input_names_from_bytes,
    get_output_names_from_bytes,
    randomize_weights_onnx_bytes,
    remove_weights_data,
    validate_batch_size,
)
from modelopt.torch._deploy.utils import (
    OnnxBytes,
    flatten_tree,
    generate_onnx_input,
    get_onnx_bytes,
    get_onnx_bytes_and_metadata,
)
from modelopt.torch._deploy.utils.torch_onnx import _to_expected_onnx_type
from modelopt.torch.utils import standardize_model_args, unflatten_tree

deploy_benchmark_all = get_deploy_models()
deploy_benchmark_dynamo = get_deploy_models(dynamic_control_flow=False)


@pytest.mark.parametrize(
    "model", deploy_benchmark_dynamo.values(), ids=deploy_benchmark_dynamo.keys()
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12) and Version(torch.__version__) < Version("2.4"),
    reason="torch.compile is not supported for Python 3.12+ with torch < 2.4",
)
def test_onnx_dynamo_export(model: BaseDeployModel):
    # try it for all potential numeric types
    for active in range(model.get.num_choices):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, _ = get_onnx_bytes_and_metadata(model, args, dynamo_export=True)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

        if model.compile_fail:
            continue

        assert onnx_bytes != b""
        assert onnx.load_model_from_string(onnx_bytes)


@pytest.mark.parametrize("model", deploy_benchmark_all.values(), ids=deploy_benchmark_all.keys())
def test_onnx_export_and_inputs(model: BaseDeployModel):
    # try it for all potential numeric types
    for active in range(model.get.num_choices):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, metadata = get_onnx_bytes_and_metadata(model, args)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

        if model.compile_fail:
            continue

        assert onnx_bytes != b""
        assert onnx.load_model_from_string(onnx_bytes)

        # check correct naming assignment of ops by running a onnx inference session
        ort_session = ort.InferenceSession(onnx_bytes)
        ort_inputs = [inp.name for inp in ort_session.get_inputs()]

        print(ort_session)

        # NOTE: for dict inputs the order is determined by the order of the keys!
        # So if we change the order of the keys in the input this check might fail
        assert ort_inputs == model.onnx_input_names()

        # check correct naming assignments of outputs
        ort_outputs = [out.name for out in ort_session.get_outputs()]
        assert ort_outputs == model.onnx_output_names()

        # check correct output structure
        assert json.dumps(metadata["output_tree_spec"].spec) == json.dumps(model.output_spec())

        if not model.check_input_option(active):
            continue

        # run inference in ORT session with hand-generated input
        model.get.set_default_counter()
        out_ort_flat = ort_session.run(None, {k: np.asarray(model.get()) for k in ort_inputs})

        # run inference with torch model and flatten them
        out_torch = model(*standardize_model_args(model, args))
        out_torch_flat, out_torch_tree_spec = flatten_tree(out_torch)

        # making sure we have pytorch output type as expected ...
        out_torch_flat = [_to_expected_onnx_type(x) for x in out_torch_flat]

        print(out_ort_flat, out_torch_flat)

        # compare flat ORT and torch results
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort_flat)
        )

        # run inference with properly generated onnx inputs and fill data structure
        inputs_generated = generate_onnx_input(metadata, args)

        if model.invalid_device_input:
            continue

        inputs_generated = {k: v.cpu().numpy() for k, v in inputs_generated.items()}
        out_ort2 = unflatten_tree(ort_session.run(None, inputs_generated), out_torch_tree_spec)

        # now flatten both and compare
        out_ort2_flat, _ = flatten_tree(out_ort2)
        print(out_torch_flat, out_ort2_flat)
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort2_flat)
        )


class SingleArgModel(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.add(x, x) - x


class DoubleArgModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y) - x


@pytest.mark.parametrize(
    "model, n_args, batch_size",
    [
        (SingleArgModel(), 1, 1),
        (SingleArgModel(), 1, 2),
        (DoubleArgModel(), 2, 1),
        (DoubleArgModel(), 2, 2),
    ],
)
def test_get_and_validate_batch_size(model, n_args, batch_size):
    inputs = (torch.randn([batch_size, 3, 32, 32]),) * n_args
    onnx_bytes = get_onnx_bytes(model, inputs)

    assert validate_batch_size(onnx_bytes, batch_size)
    assert validate_batch_size(onnx_bytes, 3) is False

    assert batch_size == get_batch_size_from_bytes(onnx_bytes)


def make_onnx_model_for_matmul_op():
    input_left = np.array([1, 2])
    input_right = np.array([1, 3])
    output_shape = np.matmul(input_left, input_right).shape
    node = make_node("MatMul", ["X", "Y"], ["Z"], name="matmul")
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_left.shape),
            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_right.shape),
        ],
        [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = make_model(graph, producer_name="Omniengine Tester")
    return model.SerializeToString()


def test_input_names():
    model_bytes = make_onnx_model_for_matmul_op()
    input_names = get_input_names_from_bytes(model_bytes)
    assert input_names == ["X", "Y"]


def test_output_names():
    model_bytes = make_onnx_model_for_matmul_op()
    output_names = get_output_names_from_bytes(model_bytes)
    assert output_names == ["Z"]


def _get_avg_var_of_weights(model):
    inits = model.graph.initializer
    avg_var_dict = {}

    for init in inits:
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                np_tensor = np.frombuffer(init.raw_data, dtype=dtype)
                avg_var_dict[init.name + "_avg"] = np.average(np_tensor)
                avg_var_dict[init.name + "_var"] = np.var(np_tensor)

    return avg_var_dict


def test_random_onnx_weights():
    model, args, kwargs = get_tiny_resnet_and_input()
    assert not kwargs

    onnx_bytes = get_onnx_bytes(model, args)
    original_avg_var_dict = _get_avg_var_of_weights(onnx.load_from_string(onnx_bytes))
    original_model_size = len(onnx_bytes)

    onnx_bytes = remove_weights_data(onnx_bytes)
    # Removed model weights should be greater than 18 MB
    assert original_model_size - len(onnx_bytes) > 18e6

    # After assigning random weights, model size should be slightly greater than the the original
    # size due to some extra metadata
    onnx_bytes = randomize_weights_onnx_bytes(onnx_bytes)
    assert len(onnx_bytes) > original_model_size

    randomized_avg_var_dict = _get_avg_var_of_weights(onnx.load_from_string(onnx_bytes))
    for key, value in original_avg_var_dict.items():
        assert abs(value - randomized_avg_var_dict[key]) < 0.1


def test_reproducible_random_weights():
    model, args, kwargs = get_tiny_resnet_and_input()
    assert not kwargs

    original_onnx_bytes = get_onnx_bytes(model, args)
    onnx_bytes_wo_weights = remove_weights_data(original_onnx_bytes)

    # Check if the randomization produces the same weights
    onnx_bytes_1 = randomize_weights_onnx_bytes(onnx_bytes_wo_weights)
    onnx_bytes_2 = randomize_weights_onnx_bytes(onnx_bytes_wo_weights)
    assert onnx_bytes_1 == onnx_bytes_2
