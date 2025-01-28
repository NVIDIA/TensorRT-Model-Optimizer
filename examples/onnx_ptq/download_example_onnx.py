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
