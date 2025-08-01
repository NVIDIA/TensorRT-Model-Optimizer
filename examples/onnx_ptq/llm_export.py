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

"""This script is used to export a LLM model to ONNX and perform quantization."""

import argparse
import os
import shutil
import time

import onnx
import onnx_graphsurgeon as gs
import torch
from packaging.version import Version
from transformers import AutoTokenizer

from modelopt.onnx.llm_export.utils.export_utils import (
    ModelLoader,
    WrapperModelForCausalLM,
    llm_to_onnx,
)


def llm_arguments():
    """Parse the arguments for the llm export script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_dir", type=str, help="The folder of HF PyTorch model ckpt", required=False
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp8", "nvfp4"],
        help="The precision of onnx export",
    )

    parser.add_argument(
        "--lm_head",
        type=str,
        default="fp16",
        choices=["fp16"],
        help="The precision of lm_head. Currently only fp16 is tested and supported",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to store the generated ONNX model",
        required=True,
    )

    parser.add_argument(
        "--onnx_path",
        type=str,
        help="Pass this option when you have existing onnx to surgeon",
        required=False,
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        default=False,
        help="Save the original ONNX from torch.onnx.export without any modification",
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="The path of dataset for quantization", required=False
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="The path of config.json, in case it is not with the PyTorch or ONNX file",
        default=None,
    )
    return parser


def get_config_path(args):
    """Look for config.json. It is recommended to keep a copy per ONNX path.

    Args:
        args: argparse.Namespace

    Returns:
        str: The path of config.json
    """
    if args.config_path and os.path.exists(args.config_path):
        return args.config_path
    if args.torch_dir:
        torch_config = os.path.join(args.torch_dir, "config.json")
        if os.path.exists(torch_config):
            return torch_config
    if args.onnx_path:
        onnx_config = os.path.join(os.path.dirname(args.onnx_path), "config.json")
        if os.path.exists(onnx_config):
            return onnx_config
    print("Warning: cannot find config.json. Please pass in --config_path.")
    return None


def export_raw_llm(
    model,
    output_dir,
    dtype,
    config_path,
    torch_dir,
    lm_head_precision="fp16",
    dataset_dir="",
    wrapper_cls=WrapperModelForCausalLM,
    extra_inputs={},
    extra_dyn_axes={},
):
    """Export raw llm model to ONNX and perform quantization.

    Args:
        model: torch.nn.module
        output_dir: str
        dtype: str
        config_path: str
        torch_dir: str, Used for loading tokenizer for quantization
        dataset_dir: str, Used for quantization
        wrapper_cls: class, Used for wrapping the model
        extra_inputs: dict, Used for extra inputs
        extra_dyn_axes: dict, Used for extra dynamic axes
    """
    os.makedirs(output_dir, exist_ok=True)

    if dtype == "fp16":
        print("Loading fp16 ONNX model...")

        llm_to_onnx(
            wrapper_cls(model), output_dir, extra_inputs=extra_inputs, extra_dyn_axes=extra_dyn_axes
        )
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))

    # Need to quantize model to fp8, int4 or nvfp4
    if dtype in ["fp8", "nvfp4"]:
        # Avoid import modelopt when no quantization is needed
        from modelopt.onnx.llm_export.utils.quantization_utils import quantize
        from modelopt.torch.export import export_hf_checkpoint
        from modelopt.torch.quantization.utils import is_quantized_linear

        tokenizer = AutoTokenizer.from_pretrained(torch_dir, trust_remote_code=True)
        modelopt_state = os.path.join(torch_dir, "modelopt_state.pth")
        if not os.path.exists(modelopt_state):
            model = quantize(model, tokenizer, dtype, lm_head_precision, dataset_dir)

            if dtype == "nvfp4":
                # This is required for nvfp4 ONNX export
                for module in model.modules():
                    assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
                    if isinstance(module, torch.nn.Linear):
                        module.input_quantizer._trt_high_precision_dtype = "Half"
                        module.input_quantizer._onnx_quantizer_type = "dynamic"
                        module.weight_quantizer._onnx_quantizer_type = "static"

            if dtype in {"fp8", "nvfp4"}:
                print(f"Exporting {dtype} ONNX model from quantized PyTorch model...")
                llm_to_onnx(
                    wrapper_cls(
                        model,
                    ),
                    output_dir,
                    extra_inputs=extra_inputs,
                    extra_dyn_axes=extra_dyn_axes,
                )
                shutil.copy(config_path, os.path.join(output_dir, "config.json"))

            # Compress weights
            quantized_model_dir = f"{output_dir}_{dtype}_quantized"
            os.makedirs(quantized_model_dir, exist_ok=True)
            with torch.inference_mode():
                export_hf_checkpoint(model, dtype=torch.float16, export_dir=quantized_model_dir)

    return model.state_dict()


def surgeon_llm(
    raw_onnx_path,
    output_dir,
    dtype,
    config_path,
    lm_head_precision="fp16",
):
    """Surgeon raw llm onnx to fit TRT.

    For example, insert quantization q/dq nodes.

    Args:
        raw_onnx_path: str
        output_dir: str
        dtype: str
        config_path: str
        lm_head_precision: str
    """

    t0 = time.time()
    graph = gs.import_onnx(onnx.load(raw_onnx_path))
    t1 = time.time()
    print(f"Importing ONNX graph takes {t1 - t0}s.")
    graph.fold_constants().cleanup().toposort()

    if dtype == "fp8" or lm_head_precision == "fp8":
        from modelopt.onnx.llm_export.utils.surgeon_utils import fold_fp8_qdq_to_dq

        graph = fold_fp8_qdq_to_dq(graph)

    os.makedirs(output_dir, exist_ok=True)
    t2 = time.time()

    onnx_model = gs.export_onnx(graph)

    if dtype == "nvfp4":
        t4 = time.time()
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

        onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)
        t5 = time.time()
        print(f"nvfp4 qdq to 2 dqs inserted in {t5 - t4}.")

    output_onnx_name = f"{output_dir}/model.onnx"
    print(
        f"Saving ONNX files in {output_dir}. All existing ONNX in the folder will be overwritten."
    )
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if (
                os.path.isfile(file_path) or os.path.islink(file_path)
            ) and ".json" not in file_path:
                os.unlink(file_path)

        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    onnx.save_model(
        onnx_model,
        output_onnx_name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="onnx_model.data",
        convert_attribute=True,
    )

    if os.path.exists(config_path):
        if config_path.endswith("config.json"):
            shutil.copy(config_path, os.path.join(output_dir, "config.json"))
        else:
            shutil.copy(
                os.path.join(config_path, "config.json"), os.path.join(output_dir, "config.json")
            )

    t3 = time.time()
    print(f"Surgeon LLM completed in {t3 - t2}s.")


def check_dtype_support(args):
    """Check whether the dtype is supported by DriveOS LLM SDK.

    Returns False if it is not supported because of:
        1. Modelopt < 0.23.0 does not support nvfp4
    """

    def get_modelopt_version():
        try:
            import modelopt

            return Version(modelopt.__version__)
        except Exception as e:
            print(f"Modelopt version cannot be parsed. Reason: {e!s}")

    if (args.dtype == "nvfp4") and get_modelopt_version() < Version("0.23.0"):
        print(
            "nvfp4 is not supported by installed modelopt version. Please upgrade to 0.23.0 or above for nvfp4 export."
        )
        return False

    return True


def main(args):
    """Main function to export the LLM model to ONNX."""
    assert args.torch_dir or args.onnx_path, (
        "You need to provide either --torch_dir or --onnx_path to process the export script."
    )
    start_time = time.time()

    if not check_dtype_support(args):
        return

    if args.onnx_path:
        raw_onnx_path = args.onnx_path

    model_loader = ModelLoader(
        args.torch_dir,
        args.config_path,
    )

    if args.torch_dir:
        # Exporting ONNX from PyTorch model
        model = model_loader.load_model()
        onnx_dir = args.output_dir + "_raw" if args.save_original else args.output_dir
        # Surgeon graph based on precision
        raw_onnx_path = f"{onnx_dir}/model.onnx"
        extra_inputs, extra_dyn_axes = {}, {}
        export_raw_llm(
            model=model,
            output_dir=onnx_dir,
            dtype=args.dtype,
            config_path=args.config_path,
            torch_dir=args.torch_dir,
            lm_head_precision=args.lm_head,
            dataset_dir=args.dataset_dir,
            wrapper_cls=WrapperModelForCausalLM,
            extra_inputs=extra_inputs,
            extra_dyn_axes=extra_dyn_axes,
        )

    # Providing the config path to config.json results in a hf validation error for internvl_chat.
    surgeon_llm(
        raw_onnx_path=raw_onnx_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        config_path=args.config_path,
        lm_head_precision=args.lm_head,
    )

    end_time = time.time()
    print(
        f"LLM ONNX saved to {args.output_dir} with {args.dtype} precision in {end_time - start_time}s."
    )


if __name__ == "__main__":
    parser = llm_arguments()
    args = parser.parse_args()
    args.config_path = get_config_path(args)
    main(args)
