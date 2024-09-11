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

import timm
import torch
from evaluation import evaluate_accuracy
from torchvision.datasets import ImageNet

from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="""Path to the image classification ONNX model with input shape of
        [batch_size,3,224,224] and output shape of [1,1000]""",
    )
    parser.add_argument(
        "--imagenet_path", type=str, required=True, help="Path to the imagenet dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="use for timm.create_model to load data config",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--eval_data_size", type=int, default=None, help="Number of examples to evaluate"
    )
    # By default, TensorRT autotunes tensor types to generate the fastest engine. When you specify
    # to TensorRT that a network is strongly typed, it infers a type for each intermediate and
    # output tensor using the rules in the operator type specification. For networks quantized in
    # INT4 or FP8 mode, stronglyTyped as the mode is recommended for TensorRT deployment. Though
    # INT8 networks are generally compiled with int8 mode, certain INT8 ViT networks compiled with
    # stronglyTyped precision have shown better performance.
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="best",
        choices=["fp8", "fp16", "fp32", "int4", "int8", "bf16", "best", "stronglyTyped"],
        help="Quantization mode for the TensorRT engine. \
            Supported options: fp8, fp16, fp32, int8, bf16, best, stronglyTyped",
    )

    args = parser.parse_args()

    deployment = {
        "runtime": "TRT",
        "version": "10.3",
        "precision": args.quantize_mode,
    }

    # Create an ONNX bytes object with the specified path
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # Get the runtime client
    client = RuntimeRegistry.get(deployment)

    # Compile the TRT model
    compiled_model = client.ir_to_compiled(onnx_bytes)

    print(f"Size of the TensorRT engine: {len(compiled_model)/(1024**2)} MB")

    # Create the device model
    device_model = DeviceModel(client, compiled_model, metadata={})

    if args.imagenet_path:
        model = timm.create_model(args.model_name, pretrained=False, num_classes=1000)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageNet(root=args.imagenet_path, split="val", transform=transforms)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        top1_accuracy, top5_accuracy = evaluate_accuracy(
            device_model, val_loader, args.eval_data_size, args.batch_size, topk=(1, 5)
        )
        print(f"The top1 accuracy of the model is {top1_accuracy}%")
        print(f"The top5 accuracy of the model is {top5_accuracy}%")

    print(f"Inference latency of the model is {device_model.get_latency()} ms")


if __name__ == "__main__":
    main()
