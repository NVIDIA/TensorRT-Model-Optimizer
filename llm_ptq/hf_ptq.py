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
import copy
import random
import time

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from example_utils import get_model, get_model_type, get_tokenizer, is_model_on_gpu
from tqdm import tqdm

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
import modelopt.torch.utils.dataset_utils as dataset_utils
from modelopt.torch.export import export_tensorrt_llm_checkpoint, export_to_vllm

RAND_SEED = 1234

QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
}


def quantize_model(model, quant_cfg, calib_dataloader=None):
    # The calibration loop for the model can be setup using the modelopt API.
    #
    # Example usage:
    # from modelopt.torch.utils.dataset_utils import create_forward_loop
    # model = ...  # Initilaize the model
    # tokenizer = ...  # Initilaize the tokenizer
    # quant_cfg = ...  # Setup quantization configuration
    # forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    # mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        print("Calibrating the model...")
        for data in tqdm(calib_dataloader):
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")

    return model


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    if args.medusa:
        from medusa.model.medusa_model import MedusaModel

        model = MedusaModel.from_pretrained(
            args.pyt_ckpt_path,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model_type = get_model_type(model)
        tokenizer = model.get_tokenizer()
    else:
        model = get_model(args.pyt_ckpt_path, args.device)
        model_type = get_model_type(model)
        tokenizer = get_tokenizer(args.pyt_ckpt_path, model_type=model_type)

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    if args.sparsity_fmt != "dense":
        if args.batch_size == 0:
            # Sparse algorithm takes more GPU memory so we reduce the batch_size by 2.
            args.batch_size = max(dataset_utils.get_max_batch_size(model) // 2, 1)
            if args.batch_size > args.calib_size:
                args.batch_size = args.calib_size

        print(f"Use calib batch_size {args.batch_size}")

        # Different calibration datasets are also available, e.g., "pile" and "wikipedia"
        # Please also check the docstring for the datasets available
        calib_dataloader = dataset_utils.get_dataset_dataloader(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            device=device,
        )
        model = mts.sparsify(
            model,
            args.sparsity_fmt,
            config={"data_loader": calib_dataloader, "collect_func": lambda x: x},
        )
        mts.export(model)

    if args.qformat in ["fp8", "int8_sq", "int4_awq", "w4a8_awq"] and not args.naive_quantization:
        if args.qformat != "fp8":
            assert is_model_on_gpu(model), (
                f"Model must be fully loaded onto GPUs for {args.qformat} calibration. "
                "Please make sure the system has enough GPU memory to load the model."
            )

        if "awq" in args.qformat:
            print(
                "\n####\nAWQ calibration could take longer than other calibration methods. "
                "Consider reducing calib_size to reduce calibration time.\n####\n"
            )

        if args.batch_size == 0:
            args.batch_size = dataset_utils.get_max_batch_size(model)
            if args.batch_size > args.calib_size:
                args.batch_size = args.calib_size

        print(f"Use calib batch_size {args.batch_size}")

        calib_dataloader = dataset_utils.get_dataset_dataloader(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            device=device,
        )
        if args.qformat in QUANT_CFG_CHOICES:
            quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {args.qformat}")

        if "awq" in args.qformat:
            quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[args.qformat])
            weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = args.awq_block_size

        if args.medusa:
            # kv_cache quantization will casue segfault with medusa in TRTLLM 0.10
            # this can be resolved in future TRTLLM version
            enable_quant_kv_cache = False
        else:
            # Always turn on FP8 kv cache to save memory footprint.
            # For int8_sq, we do not quantize kv cache to preserve accuracy.
            enable_quant_kv_cache = "int8" not in args.qformat
        print(f'{"Enable" if enable_quant_kv_cache else "Disable"} KV cache quantization')
        quant_cfg["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
            "num_bits": 8 if args.qformat == "int8_sq" else (4, 3),
            "axis": None,
            "enable": enable_quant_kv_cache,
        }

        if args.medusa:
            # Turn off FP8 Medusa.
            quant_cfg["quant_cfg"]["*medusa_head*"] = {"enable": False}  # type: ignore[index]

        # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
        if model_type == "gemma" and args.qformat == "int8_sq":
            quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}  # type: ignore[index]

        # Medusa generate can only run on single GPU per official implementation.
        # Therefore, we skip generate here in case multiple GPUs are needed for inference

        # Only run single sample for preview
        input_ids = next(iter(calib_dataloader))[0:1]
        if not args.medusa:
            generated_ids_before_ptq = model.generate(input_ids, max_new_tokens=100)

        model = quantize_model(model, quant_cfg, calib_dataloader)
        # Lets print the quantization summary
        mtq.print_quant_summary(model)

        # Medusa generate can only run on single GPU per official implementation.
        # Therefore, we skip generate here in case multiple GPUs are needed for inference
        if not args.medusa:
            # Run some samples
            generated_ids_after_ptq = model.generate(input_ids, max_new_tokens=100)

            print("--------")
            print(f"example test input: {tokenizer.batch_decode(input_ids)}")
            print("--------")
            print(
                "example outputs before ptq: "
                f"{tokenizer.batch_decode(generated_ids_before_ptq[:, input_ids.shape[1]:])}"
            )
            print("--------")
            print(
                f"example outputs after ptq: {tokenizer.batch_decode(generated_ids_after_ptq[:, input_ids.shape[1]:])}"
            )
    else:
        assert model_type != "dbrx", f"Does not support export {model_type} without quantizaton"
        print(f"No quantization applied, export {device} model")

    if getattr(model.config, "model_type", None) in ["t5"]:
        trtllm_dtype = torch.bfloat16
    else:
        trtllm_dtype = torch.float16

    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.export_path

        # Move meta tensor back to device before exporting.
        remove_hook_from_module(model, recurse=True)

        start_time = time.time()
        if args.deployment == "vllm":
            export_to_vllm(
                model,
                tokenizer,
                export_path=export_path,
            )
        elif args.deployment == "tensorrt_llm":
            export_tensorrt_llm_checkpoint(
                model,
                model_type,
                trtllm_dtype,
                export_dir=export_path,
                inference_tensor_parallel=args.inference_tensor_parallel,
                inference_pipeline_parallel=args.inference_pipeline_parallel,
                naive_fp8_quantization=args.naive_quantization,
            )
        else:
            raise NotImplementedError(f"{args.deployment} not supported")

        end_time = time.time()
        print(
            f"Quantized model exported to :{export_path}. Total time used {end_time - start_time}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path", help="Specify where the PyTorch checkpoint path is", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qformat", help="Quantization format.", default="fp8")
    parser.add_argument(
        "--batch_size",
        help="Batch size for calibration. Default to 0 as we calculate max batch size on-the-fly",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--calib_size", help="Number of samples for calibration.", type=int, default=512
    )
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument(
        "--dataset_name", help="name of dataset.", type=str, default="cnn_dailymail"
    )
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--awq_block_size", default=128)
    parser.add_argument(
        "--sparsity_fmt",
        help="Sparsity format.",
        default="dense",
        choices=["dense", "sparsegpt"],
    )
    parser.add_argument("--naive_quantization", default=False, action="store_true")
    parser.add_argument(
        "--medusa",
        help="Specify whether this is a Medusa model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--deployment",
        required=False,
        default="tensorrt_llm",
        choices=["vllm", "tensorrt_llm"],
        help=("Deployment platform"),
    )
    args = parser.parse_args()

    main(args)
