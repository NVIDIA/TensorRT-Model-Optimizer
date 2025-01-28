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

import os
import sys

import onnx
import onnx_graphsurgeon as gs
import torch
from _test_utils.onnx_quantization.lib_test_models import SimpleMLP, export_as_onnx

from modelopt.onnx.quantization.quantize import quantize


def test_fused_q(tmpdir):
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    onnx_path = os.path.join(tmpdir, f"{sys._getframe().f_code.co_name}.onnx")
    onnx_path = export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    # Quantize the model and export with only DQ nodes for weights
    quantize(onnx_path, quantize_mode="fp8", dq_only=True)

    # Output model should be produced in the same tmpdir
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")

    # Check that quantized explicit model is generated
    assert os.path.isfile(output_onnx_path)

    # Load the output model and check QDQ node placements
    graph = gs.import_onnx(onnx.load(output_onnx_path))

    # Check that 3 Q nodes are fused, hence we have 3 less Q nodes than DQ nodes
    q_nodes = [n for n in graph.nodes if n.op == "QuantizeLinear"]
    dq_nodes = [n for n in graph.nodes if n.op == "DequantizeLinear"]

    assert len(q_nodes) == 3
    assert len(dq_nodes) == 6
