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
import os
import sys
from types import ModuleType
from unittest import mock

import pytest

from modelopt.torch._deploy._runtime.tensorrt.constants import (
    DEFAULT_NUM_INFERENCE_PER_RUN,
    WARMUP_TIME_MS,
)

tensorrt = ModuleType("tensorrt")
tensorrt.Logger = mock.Mock(name="tensorrt_logger")  # type: ignore[attr-defined]
tensorrt.tensorrt = mock.Mock(name="tensorrt_tensorrt")  # type: ignore[attr-defined]
tensorrt.__version__ = "8.2"  # type: ignore[attr-defined]
sys.modules[tensorrt.__name__] = tensorrt

trex = ModuleType("trex")
trex.EnginePlan = mock.Mock(name="trex_EnginePlan")  # type: ignore[attr-defined]
trex.layer_type_formatter = mock.Mock(name="trex_layer_type_formatter")  # type: ignore[attr-defined]
trex.to_dot = mock.Mock(name="trex_to_dot")  # type: ignore[attr-defined]
trex.render_dot = mock.Mock(name="trex_render_dot")  # type: ignore[attr-defined]
sys.modules[trex.__name__] = trex

from modelopt.torch._deploy._runtime.tensorrt.engine_builder import (  # noqa
    build_engine,
    profile_engine,
)


@pytest.fixture
def setup_mock_tmp_path_and_engine_bytes(tmp_path):
    with (
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder._run_command"
        ) as mock_run_command,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder.TemporaryDirectory"
        ) as mock_temporary_directory,
        mock.patch(
            "modelopt.torch._deploy._runtime.tensorrt.engine_builder._draw_engine"
        ) as mock_draw_engine,
    ):
        mock_run_command.return_value = (0, "success")
        mock_temporary_directory.return_value.__enter__.return_value = tmp_path
        mock_onnx_bytes = mock.Mock()
        mock_onnx_bytes.model_name = "model"
        mock_draw_engine.return_value = b"svg"

        engine_bytes = b"engine_bytes"
        os.makedirs(os.path.join(tmp_path, "model"), exist_ok=True)
        with open(os.path.join(tmp_path, "model", "model.engine"), "wb") as f:
            f.write(engine_bytes)

        graph_json = {"name": "graph_json"}
        with open(os.path.join(tmp_path, "model", "model.engine.graph.json"), "w") as f:
            json.dump(graph_json, f)

        graph_svg = b"svg"
        with open(os.path.join(tmp_path, "model", "model.engine.graph.json.svg"), "wb") as f:
            f.write(graph_svg)

        expected_engine_bytes = b"\x99y0\xf2.\x99\xa3\xeeFe=\xb1\xa7\x85\xf7\xa5\xda\x1eW/\x1c\x12\xc7R\x1d\xb2t\x88\x0f\xb6b,engine_bytes"  # noqa: E501

        yield (tmp_path, mock_run_command, expected_engine_bytes, mock_onnx_bytes)


def test_build_engine(setup_mock_tmp_path_and_engine_bytes):
    tmp_path, mock_run_command, expected_engine_bytes, mock_onnx_bytes = (
        setup_mock_tmp_path_and_engine_bytes
    )

    engine_bytes, out, graph_svg = build_engine(
        onnx_bytes=mock_onnx_bytes, draw_engine=True, verbose=True
    )

    mock_run_command.assert_called_once_with(
        f"trtexec --onnx={tmp_path}/onnx/model.onnx"
        f" --saveEngine={tmp_path}/model/model.engine --skipInference"
        " --builderOptimizationLevel=3"
        f" --exportLayerInfo={tmp_path}/model/model.engine.graph.json"
        " --verbose"
    )

    assert engine_bytes == expected_engine_bytes
    assert out == b"success"
    assert graph_svg == b"svg"


def test_build_engine_dynamic_shapes(setup_mock_tmp_path_and_engine_bytes):
    tmp_path, mock_run_command, expected_engine_bytes, mock_onnx_bytes = (
        setup_mock_tmp_path_and_engine_bytes
    )

    dynamic_shapes = {
        "minShapes": {"input": [1, 3, 244, 244]},
        "optShapes": {"input": [16, 3, 244, 244]},
        "maxShapes": {"input": [32, 3, 244, 244]},
    }

    engine_bytes, out, graph_svg = build_engine(
        onnx_bytes=mock_onnx_bytes, draw_engine=True, dynamic_shapes=dynamic_shapes
    )

    mock_run_command.assert_called_once_with(
        f"trtexec --onnx={tmp_path}/onnx/model.onnx"
        " --minShapes=input:1x3x244x244 --optShapes=input:16x3x244x244"
        f" --maxShapes=input:32x3x244x244 --saveEngine={tmp_path}/model/model.engine --skipInference"
        " --builderOptimizationLevel=3"
        f" --exportLayerInfo={tmp_path}/model/model.engine.graph.json"
    )

    assert engine_bytes == expected_engine_bytes
    assert out == b"success"
    assert graph_svg == b"svg"


@pytest.mark.parametrize("kwargs", [{}, {"profiling_runs": 2}])
def test_profile_engine_runs(setup_mock_tmp_path_and_engine_bytes, kwargs):
    tmp_path, mock_run_command, _, _ = setup_mock_tmp_path_and_engine_bytes
    profiled_data, out = profile_engine(engine_bytes=b"engine", onnx_node_names=[], **kwargs)

    mock_run_command.assert_called_once_with(
        f"trtexec --loadEngine={tmp_path}/engine --warmUp={WARMUP_TIME_MS}"
        f" --avgRuns={DEFAULT_NUM_INFERENCE_PER_RUN}"
        f" --iterations={kwargs.get('profiling_runs', 1) * DEFAULT_NUM_INFERENCE_PER_RUN}"
    )

    assert profiled_data == {}
    assert out == b"success"
