#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""This module provides the command line interface (CLI) entry point for AutoCast."""

import argparse
import sys

import onnx

from modelopt.onnx.autocast.convert import (
    DEFAULT_DATA_MAX,
    DEFAULT_INIT_CONVERSION_MAX_BYTES,
    DEFAULT_INIT_MAX,
    convert,
)
from modelopt.onnx.autocast.logging_config import configure_logging, logger


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for AutoCast."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output filename to save the converted ONNX model. If None, save it in the same dir as "
        "the original ONNX model with an appropriate suffix.",
    )
    parser.add_argument(
        "--low_precision_type",
        "-t",
        type=str,
        default="fp16",
        help="Precision to reduce to",
        choices=["fp16", "bf16"],
    )
    parser.add_argument(
        "--calibration_data",
        "-d",
        type=str,
        help="File path to inputs for reference runner, either NPZ or Polygraphy JSON file. "
        "If not provided, random inputs will be used",
    )
    parser.add_argument(
        "--nodes_to_exclude",
        "-n",
        type=str,
        nargs="*",
        default=[],
        help="List of regex patterns to match node names that should remain in FP32",
    )
    parser.add_argument(
        "--op_types_to_exclude",
        "-op",
        type=str,
        nargs="*",
        default=[],
        help="List of op types that should remain in FP32",
    )
    parser.add_argument(
        "--data_max",
        type=float,
        default=DEFAULT_DATA_MAX,
        help="Maximum absolute value for node outputs, nodes with outputs greater than this value will remain in FP32",
    )
    parser.add_argument(
        "--init_max",
        type=float,
        default=DEFAULT_INIT_MAX,
        help="Maximum absolute value for initializers, nodes with initializers greater than this value will remain in "
        "FP32",
    )
    parser.add_argument(
        "--init_conversion_max_bytes",
        type=int,
        default=DEFAULT_INIT_CONVERSION_MAX_BYTES,
        help="Maximum size in bytes for initializer conversion. Larger initializers will be cast at runtime.",
    )
    parser.add_argument(
        "--keep_io_types",
        action="store_true",
        help="Keep the input and output types of the model, otherwise they will be converted to FP16",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main(argv=None):
    """Main entry point for AutoCast command line interface.

    Args:
        argv: List of command line arguments.

    Returns:
        onnx.ModelProto: The converted mixed precision model.
    """
    parser = get_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    model_out = convert(
        onnx_path=args.onnx_path,
        low_precision_type=args.low_precision_type,
        nodes_to_exclude=args.nodes_to_exclude,
        op_types_to_exclude=args.op_types_to_exclude,
        data_max=args.data_max,
        init_max=args.init_max,
        keep_io_types=args.keep_io_types,
        calibration_data=args.calibration_data,
        init_conversion_max_bytes=args.init_conversion_max_bytes,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = args.onnx_path.replace(".onnx", f".{args.low_precision_type}.onnx")

    onnx.save(model_out, output_path)
    logger.info(f"Converterd model saved to {output_path}")
    return model_out


if __name__ == "__main__":
    main(sys.argv[1:])
