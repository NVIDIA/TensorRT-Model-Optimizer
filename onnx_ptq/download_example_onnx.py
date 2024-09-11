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

import timm
import torch
from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
from optimum.onnxruntime import ORTModelForCausalLM


def torch_to_onnx(
    model,
    input_tensor,
    onnx_path,
    fp16,
    input_names=["input"],
    output_names=["output"],
    opset=20,
    do_constant_folding=True,
):
    model.eval()
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=do_constant_folding,
    )
    if fp16:
        onnx_fp32_model = load_model(onnx_path)
        onnx_fp16_model = float16_converter.convert_float_to_float16(
            onnx_fp32_model, keep_io_types=False
        )
        save_model(onnx_fp16_model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize llama model in INT4 mode.")

    parser.add_argument(
        "--vit",
        action="store_true",
        help="Export timm/vit_base_patch16_224 model to ONNX.",
    )
    parser.add_argument(
        "--llama",
        action="store_true",
        help="Export meta-llama/Llama-2-7b-chat-hf to ONNX.",
    )
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 format.")
    parser.add_argument("--batch_size", type=int, default=1, help="Output ONNX model batch size.")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the ONNX model."
    )

    args = parser.parse_args()

    if args.vit:
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
        input = torch.randn(args.batch_size, 3, 224, 224)
        torch_to_onnx(model, input, args.output_path, args.fp16)

    if args.llama:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            use_auth_token=True,
        )
        model.save_pretrained(os.path.dirname(args.output_path))
