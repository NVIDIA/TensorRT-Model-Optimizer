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

from argparse import ArgumentParser

import numpy as np
import onnx
import onnx_graphsurgeon as gs

argparser = ArgumentParser("Ensure that a custom op in ONNX is supported as a TRT plugin.")
argparser.add_argument("--onnx", type=str, help="Input ONNX model path.")
argparser.add_argument(
    "--custom_ops",
    type=str,
    nargs="+",
    default=None,
    help="A space-separated list with custom ops to ensure ORT support in those layers.",
)


if __name__ == "__main__":
    args = argparser.parse_args()

    trt_plugin_domain = "trt.plugins"
    trt_plugin_version = 1

    graph = gs.import_onnx(onnx.load(args.onnx))
    has_custom_op = False
    custom_ops = args.custom_ops or []
    for node in graph.nodes:
        # Note: nodes with module="ai.onnx*" have domain=None. Sometimes, however, custom ops might get that module
        # assigned to them, so users would need to force those layers to be in the 'trt.plugins' domain by using
        # the --custom_ops flag.
        if node.op in custom_ops or node.domain is not None:
            has_custom_op = True
            custom_ops.append(node.op)
            node.domain = trt_plugin_domain
    custom_ops = np.unique(custom_ops)

    if has_custom_op:
        model = gs.export_onnx(graph)
        model.opset_import.append(onnx.helper.make_opsetid(trt_plugin_domain, trt_plugin_version))
        onnx.save(model, args.onnx.replace(".onnx", "_ort_support.onnx"))
        print(f"Custom ops detected: {custom_ops}!")
    else:
        print("Custom op not found in model!")
