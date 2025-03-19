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

import onnx
import pytest
import torch
from _test_utils.torch_quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from modelopt.torch.quantization.utils import is_quantized_linear

dtype_to_onnx_type_name = {
    "BFloat16": "BFLOAT16",
    "Half": "FLOAT16",
}


def _get_initializer_type_by_name(model, initializer_name):
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return onnx.TensorProto.DataType.Name(initializer.data_type)
    return None


def _check_gemm_quantized(model, dtype):
    nodes = model.graph.node
    for node in nodes:
        if node.op_type == "Gemm":
            assert sum(1 for input_name in node.input if "DequantizeLinear" in input_name) == 2
        if node.op_type in ["DequantizeLinear", "TRT_FP4DynamicQuantize"]:
            scale_type = _get_initializer_type_by_name(model, node.input[1])
            assert not scale_type or scale_type == dtype_to_onnx_type_name[dtype]

            output_type = _get_initializer_type_by_name(model, node.output[0])
            assert not output_type or output_type == dtype_to_onnx_type_name[dtype]


@pytest.mark.skipif(
    not hasattr(onnx.TensorProto, "FLOAT4E2M1"), reason="onnx.TensorProto does not have FLOAT4E2M1"
)
@pytest.mark.parametrize(
    "dtype",
    ["Half", "BFloat16"],
)
def test_simple_linear(tmp_path, dtype: str):
    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    config = mtq.NVFP4_DEFAULT_CFG
    model = SimpleLinear().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]
    sample_input = model.get_input().cuda()
    if dtype == "BFloat16":
        model = model.to(torch.bfloat16)
        calib_data = [data.to(torch.bfloat16) for data in calib_data]
        sample_input = sample_input.to(torch.bfloat16)

    model = mtq.quantize(model, config, forward_loop=forward_loop)
    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
        if isinstance(module, torch.nn.Linear):
            module.input_quantizer._trt_high_precision_dtype = dtype
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"

    # Export the model to ONNX
    onnx_path = os.path.join(tmp_path, "simple_linear_qdq.onnx")
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        input_names=["x"],
        output_names=["output"],
        export_params=True,
        opset_version=17,
    )

    onnx_model = fp4qdq_to_2dq(onnx.load(onnx_path))
    _check_gemm_quantized(onnx_model, dtype)
