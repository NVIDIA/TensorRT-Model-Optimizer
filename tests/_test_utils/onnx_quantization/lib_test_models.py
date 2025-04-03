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

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision
from onnx import helper


class SimpleMLP(nn.Module):
    """Simple toy model."""

    def __init__(self, fi=16, f1=18, f2=20, fo=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fi, f1, bias=False),
            nn.ReLU(),
            nn.Linear(f1, f2, bias=False),
            nn.ReLU(),
            nn.Linear(f2, fo, bias=False),
        )

    def forward(self, x):
        for mod in self.net:
            x = mod(x)
        return x


def export_as_onnx(
    model,
    input_tensor,
    onnx_filename,
    input_names=["input"],
    output_names=["output"],
    opset=13,
    do_constant_folding=True,
):
    model.eval()
    torch.onnx.export(
        model,
        input_tensor,
        onnx_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=do_constant_folding,
    )

    return onnx_filename


def build_r1a_model():
    # Define your model inputs and outputs
    input_names = ["input_1"]
    output_names = ["resnet50/conv1_conv/BiasAdd:0"]
    input_shapes = [(1, 3, 224, 224)]
    output_shapes = [(1, 64, 112, 112)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_1", "resnet50/conv1_conv/Conv2D/ReadVariableOp:0"],
            outputs=["resnet50/conv1_conv/Conv2D:0"],
            name="resnet50/conv1_conv/Conv2D",
            dilations=[1, 1],
            strides=[2, 2],
            kernel_shape=[7, 7],
            group=1,
            auto_pad="NOTSET",
            pads=[3, 3, 3, 3],
        ),
        helper.make_node(
            op_type="Add",
            inputs=["resnet50/conv1_conv/Conv2D:0", "Reshape__7:0"],
            outputs=["resnet50/conv1_conv/BiasAdd:0"],
            name="resnet50/conv1_conv/BiasAdd",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="resnet50/conv1_conv/Conv2D/ReadVariableOp:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(64, 3, 7, 7),
            vals=np.random.uniform(low=0.5, high=1.0, size=64 * 3 * 7 * 7),
        ),
        helper.make_tensor(
            name="Reshape__7:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(64, 1, 1),
            vals=np.random.uniform(low=0.5, high=1.0, size=64 * 1 * 1),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "r1a", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 9  # TODO: remove manual ir_version change once ORT supports ir_version 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_resnet_block(feature_dim=16, num_blocks=1, final_conv=False, out_dim=None):
    """
    Construct a network block based on the BasicBlock used in ResNet 18 and 34.
    """
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(torchvision.models.resnet.BasicBlock(feature_dim, odim))
    input_tensor = torch.randn(1, feature_dim, 28, 28)
    return nn.Sequential(*feat_layers), input_tensor


def build_resnet_block_with_downsample(
    feature_dim=16, num_blocks=1, final_conv=False, out_dim=None
):
    """
    Construct a network block based on the BasicBlock used in ResNet 18 and 34.
    """
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        output_dim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        downsample = nn.Sequential(
            torchvision.models.resnet.conv3x3(feature_dim, output_dim), nn.BatchNorm2d(output_dim)
        )
        feat_layers.append(
            torchvision.models.resnet.BasicBlock(feature_dim, output_dim, downsample=downsample)
        )
    input_tensor = torch.randn(1, feature_dim, 28, 28)
    return nn.Sequential(*feat_layers), input_tensor


def find_init(onnx_model: onnx.onnx_ml_pb2.ModelProto, init_name: str) -> np.ndarray:
    ret = None
    for init in onnx_model.graph.initializer:
        if init.name == init_name:
            ret = init
            break

    assert ret is not None
    return onnx.numpy_helper.to_array(ret)


#
#                 |
#              conv_1
#             /   |
#         conv_2  |
#          /  |   |
#     conv_3  |   |
#        |    |   |
#       concatenation
#             |
#           conv_4
#             |
#
def build_conv_concat_model():
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["conv4_conv/Conv2D:0"]
    input_shapes = [(1, 128, 240, 64)]
    output_shapes = [(1, 256, 240, 64)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_0", "ReadVariableOp:0"],
            outputs=["conv1_conv/Conv2D:0"],
            name="conv1_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["conv1_conv/Conv2D:0", "ReadVariableOp:1"],
            outputs=["conv2_conv/Conv2D:0"],
            name="conv2_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["conv2_conv/Conv2D:0", "ReadVariableOp:2"],
            outputs=["conv3_conv/Conv2D:0"],
            name="conv3_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Concat",
            inputs=["conv1_conv/Conv2D:0", "conv2_conv/Conv2D:0", "conv3_conv/Conv2D:0"],
            outputs=["concat0:0"],
            name="concat0",
            axis=1,
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["concat0:0", "ReadVariableOp:3"],
            outputs=["conv4_conv/Conv2D:0"],
            name="conv4_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[1, 1],
            group=1,
            auto_pad="NOTSET",
            pads=[0, 0, 0, 0],
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="ReadVariableOp:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:2",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:3",
            data_type=onnx.TensorProto.FLOAT,
            dims=(256, 384, 1, 1),
            vals=np.random.uniform(low=0.5, high=1.0, size=256 * 384 * 1 * 1),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "r1a", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 9  # TODO: remove manual ir_version change once ORT supports ir_version 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred
