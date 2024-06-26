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

"""Utility to dump imagenet data for calibration."""

import argparse

import numpy as np
import timm
from datasets import load_dataset


def main():
    """Prepares calibration data from ImageNet dataset and saves input dictionary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=500,
        help="Number[1-100000] of images to use in calibration.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to save the image tensor data in FP16 format."
    )
    parser.add_argument(
        "--output_path", type=str, default="calib.npy", help="Path to output npy file."
    )

    args = parser.parse_args()
    dataset = load_dataset("zh-plus/tiny-imagenet")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = dataset["train"][0 : args.calibration_data_size]["image"]

    calib_tensor = []
    for image in images:
        calib_tensor.append(transforms(image))

    calib_tensor = np.stack(calib_tensor, axis=0)
    np.save(args.output_path, calib_tensor)


if __name__ == "__main__":
    main()
