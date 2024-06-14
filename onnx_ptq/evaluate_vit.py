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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--eval_data_size", type=int, default=None, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int4", "int8"],
        help="Quantization mode used for the input model. Supported options: fp16, fp32, int4, int8",
    )
    args = parser.parse_args()

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = ImageNet(root=args.imagenet_path, split="val", transform=transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    deployment = {
        "runtime": "TRT",
        "accelerator": "GPU",
        "version": "10.0",
        "precision": args.quantize_mode,
        "onnx_opset": "13",
    }

    # Create an ONNX bytes object with the specified path
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # Get the runtime client
    client = RuntimeRegistry.get(deployment)

    # Compile the TRT model
    compiled_model = client.ir_to_compiled(onnx_bytes)

    # Create the device model
    device_model = DeviceModel(client, compiled_model, metadata={})

    top1_accuracy = evaluate_accuracy(
        device_model, val_loader, args.eval_data_size, args.batch_size
    )
    print(f"The top1 accuracy of the model is {top1_accuracy}%")


if __name__ == "__main__":
    main()
