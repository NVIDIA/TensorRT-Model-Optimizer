# Adapted from: https://github.com/open-mmlab/mmcv/blob/5916fbd8789f9578cff4d325b1ffe0bc710b4534/mmcv/ops/multi_scale_deform_attn.py#L109
#
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import onnx
import onnxruntime as _ort
import torch
import torch.nn.functional as F
from onnxruntime_extensions import PyCustomOpDef, onnx_op
from onnxruntime_extensions import get_library_path as _get_library_path

from modelopt.onnx.quantization.quantize import quantize
from modelopt.onnx.utils import udpate_domain


# Note: This example uses the dino_fan_small model which can be downloaded from:
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_dino_coco/files?version=dino_fan_small_deployable_v1.0
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h_ * w_ for h_, w_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h_, w_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, h_, w_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


@onnx_op(
    op_type="MultiscaleDeformableAttnPlugin_TRT",
    inputs=[
        PyCustomOpDef.undefined,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.undefined,
        PyCustomOpDef.undefined,
    ],
    outputs=[PyCustomOpDef.dt_float],
    attrs={},
)
def multi_scale_deformable_attn_onnx(
    value, value_spatial_shapes, level_start_idx, sampling_locations, attention_weights
):
    return multi_scale_deformable_attn_pytorch(
        torch.Tensor(value),
        torch.Tensor(value_spatial_shapes).long(),
        torch.Tensor(sampling_locations),
        torch.Tensor(attention_weights),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        type=str,
        help=("Path to the dino fan small 544x960 ONNX model with MSDA plugin"),
        required=True,
    )
    args = parser.parse_args()

    output_onnx_path = args.onnx_path.replace(".onnx", ".quant.onnx")

    # Change the domain of the custom op to ai.onnx.contrib
    model = onnx.load(args.onnx_path)
    model = udpate_domain(model, "MultiscaleDeformableAttnPlugin_TRT", "ai.onnx.contrib")
    onnx.save(model, args.onnx_path)

    # Quantize the model in Int8
    quantize(
        args.onnx_path,
        op_types_to_quantize=[
            "MultiscaleDeformableAttnPlugin_TRT",
            "Conv",
            "LayerNormalization",
            "MatMul",
            "Add",
            "Mul",
            "BatchNormalization",
            "Clip",
            "AveragePool",
            "Gemm",
        ],
        output_path=output_onnx_path,
    )

    # Load the quantized model
    model = onnx.load(output_onnx_path)

    # Create an inference session for the quantized ONNX model
    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    sess = _ort.InferenceSession(model.SerializeToString(), so)

    # Test the quantized model with a random input
    inputs = torch.randn(1, 3, 544, 960)
    result = sess.run(None, {"inputs": inputs.numpy()})

    # Check if the results contain the expected number of outputs and print the shapes
    assert len(result) == 2
    print("Logits shape: ", result[0].shape)
    print("Boxes shape: ", result[1].shape)


if __name__ == "__main__":
    main()
