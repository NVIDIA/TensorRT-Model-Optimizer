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
from copy import deepcopy

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def add_groupnorm(graph):
    cnt = 0
    for node in graph.nodes:
        if (
            node.op == "Reshape"
            and node.o().op == "InstanceNormalization"
            and node.o().o().op == "Reshape"
            and node.o().o().o().op == "Mul"
            and node.o().o().o().o().op == "Add"
        ):
            last_node = node.o().o().o().o()

            instance_norm = node.o()
            epsilon = instance_norm.attrs["epsilon"]
            mul_node = node.o().o().o()
            add_node = node.o().o().o().o()

            gamma = np.ascontiguousarray(
                np.array(deepcopy(mul_node.inputs[1].values.tolist()), dtype=np.float32)
            )
            beta = np.ascontiguousarray(
                np.array(deepcopy(add_node.inputs[1].values.tolist()), dtype=np.float32)
            )

            with_swish = (
                True
                if node.o().o().o().o().o().op == "Sigmoid"
                and node.o().o().o().o().o().o().op == "Mul"
                else False
            )
            if with_swish:
                last_node = node.o().o().o().o().o().o()

            constant_gamma = gs.Constant("gamma_{}".format(cnt), gamma.reshape(-1))
            constant_beta = gs.Constant("beta_{}".format(cnt), beta.reshape(-1))
            x = node.inputs[0]
            group_norm_v = gs.Variable("group_norm_{}".format(cnt), np.dtype(np.float32), x.shape)
            group_norm = gs.Node(
                "GroupNorm",
                "GroupNorm_{}".format(cnt),
                attrs={"epsilon": epsilon, "bSwish": with_swish},
                inputs=[x, constant_gamma, constant_beta],
                outputs=[group_norm_v],
            )
            cnt += 1
            for n in graph.nodes:
                if last_node.outputs[0] in n.inputs:
                    index = n.inputs.index(last_node.outputs[0])
                    n.inputs[index] = group_norm.outputs[0]
            last_node.outputs = []
            graph.nodes.append(group_norm)

    graph.cleanup()
    print("find groupnorm: ", cnt)


def add_fp8conv(graph):
    cnt = 0
    for node in graph.nodes:
        if (
            node.op == "Conv"
            and len(node.inputs) > 0
            and len(node.inputs[0].inputs) > 0
            and node.i().op == "TRT_FP8DequantizeLinear"
            and node.i().i().op == "TRT_FP8QuantizeLinear"
        ):
            weight = node.inputs[1].inputs[0].i().inputs[0].values
            k = weight.shape[0]

            x_scale = node.i().i(1, 0).attrs["value"].values
            w_scale = node.i(1, 0).i(1, 0).attrs["value"].values

            stride = node.attrs["strides"]
            padding = node.attrs["pads"][:2]
            dilation = node.attrs["dilations"]

            weight_t = weight.transpose((0, 2, 3, 1))
            filter = np.ascontiguousarray(np.array(deepcopy(weight_t), dtype=np.float32))
            filter_constant = gs.Constant("filter_{}".format(cnt), filter)
            node.i(1, 0).i().inputs[0] = filter_constant

            raw_bias = np.ascontiguousarray(
                np.array(deepcopy(node.inputs[2].values), dtype=np.float32).reshape(1, 1, 1, -1)
            )
            bias = np.zeros(k, dtype=np.float32).reshape(-1).tolist()

            attrs = {
                "k": np.int32(k),
                "stride": np.int32(stride),
                "padding": np.int32(padding),
                "dilation": np.int32(dilation),
                "x_scale": 1.0 / x_scale,
                "w_scale": 1.0 / w_scale,
                "bias": bias,
            }

            dq_value = gs.Constant("fake_amax_{}".format(cnt), np.array([1.0], dtype=np.float32))
            node.i().inputs[1] = dq_value
            node.i(1, 0).inputs[1] = dq_value

            conv_output = node.outputs[0]
            fp8conv2d = gs.Node(
                "FP8Conv2D",
                "FP8Conv2D_{}".format(cnt),
                attrs=attrs,
                inputs=[node.inputs[0], node.inputs[1]],
            )
            graph.nodes.append(fp8conv2d)

            conv_kernel_output = gs.Variable(name="FP8Conv2D_Output_{}".format(cnt))
            fp8conv2d.outputs = [conv_kernel_output]

            bias_constant = gs.Constant("bias_{}".format(cnt), values=raw_bias)
            bias_add = gs.Node(op="Add", inputs=[conv_kernel_output, bias_constant])
            graph.nodes.append(bias_add)

            conv_output.inputs[0] = bias_add
            cnt += 1

    graph.cleanup()
    print("replaced conv count: ", cnt)


def modify_unsqueeze(graph):
    cnt = 0
    for node in graph.nodes:
        if (
            node.op == "Gemm"
            and node.o().op == "Unsqueeze"
            and node.o().o().op == "Unsqueeze"
            and node.o().o().o().op == "Add"
            and node.o().o().o().i().i().op == "FP8Conv2D"
        ):
            axes = gs.Constant("axes{}".format(cnt), np.array([1], dtype=np.int32))
            node.o().inputs[1] = axes
            node.o().o().inputs[1] = axes
            cnt += 1

    print("modify unsqueeze count: ", cnt)
    graph.cleanup()


def add_transpose_after_convin(graph):
    for node in graph.nodes:
        if node.name == "/conv_in/Conv":
            conv_out = node.outputs[0]
            transpose_in_v = gs.Variable("Transpose_convin_v")
            transpose_in = gs.Node(
                "Transpose",
                "Transpose_convin",
                attrs={"perm": [0, 2, 3, 1]},
                inputs=[conv_out],
                outputs=[transpose_in_v],
            )
            for n in graph.nodes:
                if node.outputs[0] in n.inputs:
                    index = n.inputs.index(node.outputs[0])
                    n.inputs[index] = transpose_in_v

    graph.nodes.append(transpose_in)
    graph.cleanup()


def get_new_indices(indices):
    new_indices = 0
    if indices == 0:
        new_indices = 0
    elif indices == 1:
        new_indices = 3
    elif indices == 2:
        new_indices = 1
    elif indices == 3:
        new_indices = 2
    return new_indices


def remove_transformer_transpose(graph):
    cnt = 0
    for node in graph.nodes:
        if (
            node.op == "FP8Conv2D"
            and node.o().op == "Add"
            and node.o().o().op == "Add"
            and node.o().o().o().op == "Div"
            and len(node.o().o().o().outputs[0].outputs) == 5
            and node.o().o().o().o(3, 0).op == "Add"
            and node.o().o().o().o(4, 0).op == "GroupNorm"
            and node.o().o().o().o(4, 0).o(1, 0).op == "Transpose"
        ):
            groupnorm = node.o().o().o().o(4, 0)
            transpose = groupnorm.o(1, 0)
            reshape = transpose.o()
            reshape.inputs[0] = transpose.inputs[0]

            add = node.o().o().o().o(3, 0)
            transpose = add.i(0, 0)
            if transpose.op != "Transpose":
                print("error")
            else:
                add.inputs[0] = transpose.inputs[0]
            cnt += 1

            div = node.o().o().o()
            for sub_node in div.outputs[0].outputs:
                if sub_node.op == "Shape":
                    gather = sub_node.o()
                    indices = gather.inputs[1].inputs[0].attrs["value"].values
                    new_indices = get_new_indices(indices)
                    indice_constant = gs.Constant(
                        "div_indices_{}_{}".format(cnt, new_indices),
                        np.array(new_indices, dtype=np.int32),
                    )
                    gather.inputs[1] = indice_constant

            groupnorm = node.o().o().o().o(4, 0)
            for sub_node in groupnorm.outputs[0].outputs:
                if sub_node.op == "Shape":
                    gather = sub_node.o()
                    indices = gather.inputs[1].inputs[0].attrs["value"].values
                    new_indices = get_new_indices(indices)
                    indice_constant = gs.Constant(
                        "groupnorm_indices_{}_{}".format(cnt, new_indices),
                        np.array(new_indices, dtype=np.int32),
                    )
                    gather.inputs[1] = indice_constant

    print("remove transformer transpose count: ", cnt)
    graph.cleanup()


def modify_shortcut_conv(graph):
    cnt = 0
    for node in graph.nodes:
        if node.op == "Conv" and "conv_shortcut" in node.name:
            conv_input = node.inputs[0]
            conv_output = node.outputs[0]

            transpose_in_v = gs.Variable("Transpose_in_v_{}".format(cnt))
            transpose_in = gs.Node(
                "Transpose",
                "Transpose_in_{}".format(cnt),
                attrs={"perm": [0, 3, 1, 2]},
                inputs=[conv_input],
                outputs=[transpose_in_v],
            )
            graph.nodes.append(transpose_in)
            node.inputs[0] = transpose_in_v

            transpose_out_v = gs.Variable("Transpose_out_v_{}".format(cnt))
            transpose_out_v.inputs = [node]
            transpose_out = gs.Node(
                "Transpose",
                "Transpose_out_{}".format(cnt),
                attrs={"perm": [0, 2, 3, 1]},
                inputs=[transpose_out_v],
            )
            graph.nodes.append(transpose_out)
            conv_output.inputs[0] = transpose_out
            cnt += 1
    print("add transpose count: ", cnt)


def add_transpose_before_convout(graph):
    for node in graph.nodes:
        if node.name == "/conv_out/Conv":
            conv_in = node.inputs[0]
            transpose_out_v = gs.Variable("Transpose_convout_v")
            transpose_out = gs.Node(
                "Transpose",
                "Transpose_convout",
                attrs={"perm": [0, 3, 1, 2]},
                inputs=[conv_in],
                outputs=[transpose_out_v],
            )
            node.inputs[0] = transpose_out_v

    graph.nodes.append(transpose_out)
    graph.cleanup()


def modify_resize(graph):
    cnt = 0
    for node in graph.nodes:
        if node.op == "Resize":
            resize_input = node.inputs[0]
            resize_output = node.outputs[0]

            transpose_in_v = gs.Variable("Resize_Transpose_in_v_{}".format(cnt))
            transpose_in = gs.Node(
                "Transpose",
                "Resize_Transpose_in_{}".format(cnt),
                attrs={"perm": [0, 3, 1, 2]},
                inputs=[resize_input],
                outputs=[transpose_in_v],
            )
            graph.nodes.append(transpose_in)
            node.inputs[0] = transpose_in_v

            transpose_out_v = gs.Variable("Resize_Transpose_out_v_{}".format(cnt))
            transpose_out_v.inputs = [node]
            transpose_out = gs.Node(
                "Transpose",
                "Resize_Transpose_out_{}".format(cnt),
                attrs={"perm": [0, 2, 3, 1]},
                inputs=[transpose_out_v],
            )
            graph.nodes.append(transpose_out)
            resize_output.inputs[0] = transpose_out
            cnt += 1
    print("modify resize count: ", cnt)


def modify_concat(graph):
    cnt = 0
    for node in graph.nodes:
        if node.op == "Concat" and node.o().op == "GroupNorm":
            node.attrs["axis"] = 3
            cnt += 1
    print("modify concat count: ", cnt)


def replace_fp8_qdq(graph):
    for n in graph.nodes:
        if n.op == "QuantizeLinear":
            n.op = "TRT_FP8QuantizeLinear"
            n.inputs.pop(2)
        if n.op == "DequantizeLinear":
            n.op = "TRT_FP8DequantizeLinear"
            n.inputs.pop(2)


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./onnx_fp8/unet.onnx",
        help="Input ONNX file for UNet",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="./onnx_fp8_surgeoned/sdxl_fp8_graphsurgeon.onnx",
        help="Output ONNX filename",
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            print("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    model = onnx.load(args.onnx_path)
    graph = gs.import_onnx(model)
    replace_fp8_qdq(graph)
    add_fp8conv(graph)
    add_groupnorm(graph)
    modify_unsqueeze(graph)
    add_transpose_after_convin(graph)
    remove_transformer_transpose(graph)
    modify_shortcut_conv(graph)
    add_transpose_before_convout(graph)
    modify_concat(graph)
    modify_resize(graph)

    onnx.save(gs.export_onnx(graph.cleanup()), args.output_onnx, save_as_external_data=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
