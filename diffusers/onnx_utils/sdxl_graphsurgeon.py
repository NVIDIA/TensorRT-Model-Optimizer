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
import os
import re
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs

ONNX_LARGE_FILE_THRESHOLD = 2**31


def fuse_mha_qkv_int8_sq(graph):
    tensors = graph.tensors()
    keys = tensors.keys()

    # mha  : fuse QKV QDQ nodes
    # mhca : fuse KV QDQ nodes
    q_pat = (
        "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
        ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
    )
    k_pat = (
        "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
        ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
    )
    v_pat = (
        "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
        ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
    )

    qs = list(
        sorted(
            map(
                lambda x: x.group(0),  # type: ignore
                filter(lambda x: x is not None, [re.match(q_pat, key) for key in keys]),
            )
        )
    )
    ks = list(
        sorted(
            map(
                lambda x: x.group(0),  # type: ignore
                filter(lambda x: x is not None, [re.match(k_pat, key) for key in keys]),
            )
        )
    )
    vs = list(
        sorted(
            map(
                lambda x: x.group(0),  # type: ignore
                filter(lambda x: x is not None, [re.match(v_pat, key) for key in keys]),
            )
        )
    )

    removed = 0
    assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
    for q, k, v in zip(qs, ks, vs):
        is_mha = all(["attn1" in tensor for tensor in [q, k, v]])
        is_mhca = all(["attn2" in tensor for tensor in [q, k, v]])
        assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

        if is_mha:
            tensors[k].outputs[0].inputs[0] = tensors[q]
            tensors[v].outputs[0].inputs[0] = tensors[q]
            del tensors[k]
            del tensors[v]

            removed += 2

        else:  # is_mhca
            tensors[k].outputs[0].inputs[0] = tensors[v]
            del tensors[k]

            removed += 1

    print(f"removed {removed} QDQ nodes")
    return removed  # expected 72 for L2.5


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./unet.onnx",
        help="Input ONNX file for UNet",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="./sdxl_graphsurgeon.onnx",
        help="Output ONNX filename",
    )

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            print("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    commandline entrance of the graphsurgeon. Example commands:
        python3 -m code.sdxl.tensorrt.sdxl_graphsurgeon --onnx-path=build/models/SDXL/onnx_models\
            /unetxl/model.onnx --output-onnx=/tmp/unetxl_graphsurgeon/model.onnx
    """
    model = onnx.load(args.onnx_path)
    graph = gs.import_onnx(model)
    fuse_mha_qkv_int8_sq(graph)
    model = gs.export_onnx(graph.cleanup())
    os.makedirs(Path(args.output_onnx).parent.absolute(), exist_ok=True)
    if model.ByteSize() > ONNX_LARGE_FILE_THRESHOLD:
        onnx.save_model(
            model,
            args.output_onnx,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )
    else:
        onnx.save(model, args.output_onnx)


if __name__ == "__main__":
    args = parse_args()
    main(args)
