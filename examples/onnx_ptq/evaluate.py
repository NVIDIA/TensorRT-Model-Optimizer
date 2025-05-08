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
import csv

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
        "--imagenet_path", type=str, default=None, help="Path to the imagenet dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
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
        default="stronglyTyped",
        choices=["fp8", "fp16", "fp32", "int4", "int8", "int8_iq", "bf16", "best", "stronglyTyped"],
        help="Quantization mode for the TensorRT engine. \
            Supported options: fp8, fp16, fp32, int8, int8_iq(implicit quantization), bf16, best, stronglyTyped",
    )
    parser.add_argument(
        "--results_path", type=str, default=None, help="Save the results to the specified path"
    )

    args = parser.parse_args()

    deployment = {
        "runtime": "TRT",
        "precision": args.quantize_mode,
    }

    # Create an ONNX bytes object with the specified path
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # Get the runtime client
    client = RuntimeRegistry.get(deployment)

    # Compile the ONNX model to TRT engine and create the device model
    compiled_model = client.ir_to_compiled(onnx_bytes)
    device_model = DeviceModel(client, compiled_model, metadata={})

    top1_accuracy, top5_accuracy = 0.0, 0.0
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

    latency = device_model.get_latency()
    print(f"Inference latency of the model is {latency} ms")

    if args.results_path:
        results: list[list[str | float]] = [
            ["Metric", "Value"],
            ["Top 1", top1_accuracy],
            ["Top 5", top5_accuracy],
            ["Latency", latency],
        ]
        with open(args.results_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)


if __name__ == "__main__":
    main()
