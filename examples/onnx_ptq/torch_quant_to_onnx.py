# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re

import timm
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from download_example_onnx import export_to_onnx

import modelopt.torch.quantization as mtq

"""
This script is used to quantize a timm model using dynamic quantization like MXFP8 or NVFP4.

The script will:
1. Given the model name, create a timm torch model.
2. Quantize the torch model in MXFP8 or NVFP4 mode.
3. Export the quantized torch model to ONNX format.
"""


mp.set_start_method("spawn", force=True)  # Needed for data loader with multiple workers

QUANT_CONFIG_DICT = {
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
}


def filter_func(name):
    """Filter function to exclude certain layers from quantization."""
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|"
        r"pos_embed|time_text_embed|context_embedder|norm_out|x_embedder|patch_embed).*"
    )
    return pattern.match(name) is not None


def load_calibration_data(model_name, data_size, batch_size, device):
    """Load and prepare calibration data."""
    dataset = load_dataset("zh-plus/tiny-imagenet")
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    images = dataset["train"][:data_size]["image"]
    calib_tensor = [transforms(img) for img in images]
    calib_tensor = [t.to(device) for t in calib_tensor]
    return torch.utils.data.DataLoader(
        calib_tensor, batch_size=batch_size, shuffle=True, num_workers=4
    )


def quantize_model(model, config, data_loader=None):
    """Quantize the model using the given config and calibration data."""
    if data_loader is not None:

        def forward_loop(model):
            for batch in data_loader:
                model(batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
    else:
        quantized_model = mtq.quantize(model, config)

    mtq.disable_quantizer(quantized_model, filter_func)
    return quantized_model


def get_model_input_shape(model_name, batch_size):
    """Get the input shape from timm model configuration."""
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config["input_size"]
    return (batch_size, *tuple(input_size))  # Add batch dimension


def main():
    parser = argparse.ArgumentParser(description="Quantize timm models to MXFP8 or NVFP4")

    # Model hyperparameters
    parser.add_argument(
        "--timm_model_name",
        default="vit_base_patch16_224",
        help="The timm model name to quantize.",
        type=str,
    )
    parser.add_argument(
        "--quantize_mode",
        choices=["mxfp8", "nvfp4", "int4_awq"],
        default="mxfp8",
        help="Type of quantization to apply (mxfp8, nvfp4, int4_awq)",
    )
    parser.add_argument(
        "--onnx_save_path",
        required=True,
        help="The path to save the ONNX model.",
        type=str,
    )
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=512,
        help="Number of images to use in calibration [1-512]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration and ONNX model export.",
    )

    args = parser.parse_args()

    # Get input shape from model config
    input_shape = get_model_input_shape(args.timm_model_name, args.batch_size)

    # Create model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.timm_model_name, pretrained=True, num_classes=1000).to(device)

    # Select quantization config
    config = QUANT_CONFIG_DICT[args.quantize_mode]
    data_loader = (
        None
        if args.quantize_mode == "mxfp8"
        else load_calibration_data(
            args.timm_model_name,
            args.calibration_data_size,
            input_shape[0],  # batch size
            device,
        )
    )

    # Quantize model
    quantized_model = quantize_model(model, config, data_loader)

    # Export to ONNX
    export_to_onnx(
        quantized_model,
        input_shape,
        args.onnx_save_path,
        device,
        weights_dtype="fp16",
    )

    print(f"Quantized ONNX model is saved to {args.onnx_save_path}")


if __name__ == "__main__":
    main()
