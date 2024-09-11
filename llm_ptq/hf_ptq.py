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
import json
import os
import random
import time

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from example_utils import get_model, get_model_type, get_tokenizer, is_model_on_gpu
from tqdm import tqdm

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
import modelopt.torch.speculative as mtsp
import modelopt.torch.utils.dataset_utils as dataset_utils
from modelopt.torch.export import export_hf, export_tensorrt_llm_checkpoint

RAND_SEED = 1234

QUANT_CFG_CHOICES = {
    "int8": "INT8_DEFAULT_CFG",
    "int8_sq": "INT8_SMOOTHQUANT_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",
}


def auto_quantize(
    model, qformat, auto_quantize_compression, calib_dataloader, calibrate_loop, batch_size=1
):
    model, _ = mtq.auto_quantize(
        model,
        data_loader=calib_dataloader,
        loss_func=lambda output, batch: output.loss,
        constraints={"weight_compression": auto_quantize_compression},
        quantization_formats=[
            QUANT_CFG_CHOICES[qformat],
            None,
        ],  # TRTLLM only support one quantization format or None
        collect_func=lambda x: x,
        num_calib_steps=len(calib_dataloader),
        num_score_steps=min(
            len(calib_dataloader), 128 // batch_size
        ),  # Limit the number of score steps to avoid long calibration time
        verbose=True,
    )

    # We need to explicitly calibrate for kv cache quantization
    enable_kv_cache_quantization = "int8" not in args.qformat and "hf" not in args.export_fmt
    if enable_kv_cache_quantization:
        mtq.set_quantizer_by_cfg(
            model,
            quant_cfg={"*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True}},
        )
        # Lets calibrate only the output quantizer this time. Let's disable all other quantizers.
        with mtq.set_quantizer_by_cfg_context(
            model, {"*": {"enable": False}, "*output_quantizer": {"enable": True}}
        ):
            mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
    return model


def quantize_model(model, quant_cfg, args, calib_dataloader=None):
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
            model(**data)

    assert not (
        args.auto_quantize_compression and args.inference_pipeline_parallel > 1
    ), "Auto Quantization is not supported for pipeline parallel size > 1"

    print("Starting quantization...")
    start_time = time.time()
    if args.auto_quantize_compression:
        model = auto_quantize(
            model,
            args.qformat,
            args.auto_quantize_compression,
            calib_dataloader,
            calibrate_loop,
            args.batch_size,
        )
    else:
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    # Check arguments for unified_hf export format and set to default if unsupported arguments are provided
    if args.export_fmt == "hf":
        assert (
            not args.auto_quantize_compression
        ), "Per-layer quantization config not supported by unified export api."
        assert not args.medusa, "Medusa not supported by unified export api."
        assert (
            args.sparsity_fmt == "dense"
        ), f"Sparsity format {args.sparsity_fmt} not supported by unified export api."

    if args.medusa:
        with open(os.path.join(args.pyt_ckpt_path, "config.json")) as f:
            medusa_config = json.load(f)
            model = get_model(medusa_config["base_model_name_or_path"], args.device)
            model_type = get_model_type(model)
            tokenizer = get_tokenizer(
                medusa_config["base_model_name_or_path"], model_type=model_type
            )
    else:
        model = get_model(args.pyt_ckpt_path, args.device)
        model_type = get_model_type(model)
        tokenizer = get_tokenizer(args.pyt_ckpt_path, model_type=model_type)

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    if args.sparsity_fmt != "dense":
        if args.batch_size == 0:
            # Sparse algorithm takes more GPU memory so we reduce the batch_size by 4.
            args.batch_size = max(dataset_utils.get_max_batch_size(model) // 4, 1)
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

    if args.medusa:
        config = {
            "medusa_num_heads": medusa_config["medusa_num_heads"],
            "medusa_num_layers": medusa_config["medusa_num_layers"],
        }
        mtsp.convert(model, [("medusa", config)])
        mtsp.plugins.transformers.load_medusa_head(model, medusa_config["medusa_head_path"])

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
            # TODO: Enable auto-batch size calculation for AutoQuantize
            assert (
                args.auto_quantize_compression is None
            ), "AutoQuantize requires batch_size to be specified, please specify batch_size."
            # Calibration/sparsification will actually take much more memory than regular inference
            # due to intermediate tensors for fake quantization. Setting sample_memory_usage_ratio
            # to 2 to avoid OOM for AWQ/SmoothQuant fake quantization as it will take more memory than inference.
            sample_memory_usage_ratio = 2 if "awq" in args.qformat or "sq" in args.qformat else 1.1
            args.batch_size = dataset_utils.get_max_batch_size(
                model, sample_memory_usage_ratio=sample_memory_usage_ratio
            )
            if args.batch_size > args.calib_size:
                args.batch_size = args.calib_size

        print(f"Use calib batch_size {args.batch_size}")

        calib_dataloader = dataset_utils.get_dataset_dataloader(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            device=device,
            include_labels=args.auto_quantize_compression is not None,
        )
        if args.qformat in QUANT_CFG_CHOICES:
            quant_cfg = getattr(mtq, QUANT_CFG_CHOICES[args.qformat])
        else:
            raise ValueError(f"Unsupported quantization format: {args.qformat}")

        if "awq" in args.qformat:
            quant_cfg = copy.deepcopy(getattr(mtq, QUANT_CFG_CHOICES[args.qformat]))
            weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = args.awq_block_size

            # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
            if "w4a8_awq" == args.qformat and model_type in ["gemma", "mpt"]:
                quant_cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 1}

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we do not quantize kv cache to preserve accuracy.
        # We turn off FP8 kv cache for unified_hf checkpoint
        enable_quant_kv_cache = "int8" not in args.qformat and "hf" not in args.export_fmt
        print(f'{"Enable" if enable_quant_kv_cache else "Disable"} KV cache quantization')
        quant_cfg["quant_cfg"]["*output_quantizer"] = {
            "num_bits": 8 if args.qformat == "int8_sq" else (4, 3),
            "axis": None,
            "enable": enable_quant_kv_cache,
        }

        # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
        if model_type == "gemma" and args.qformat == "int8_sq":
            quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

        # Medusa generate can only run on single GPU per official implementation.
        # Therefore, we skip generate here in case multiple GPUs are needed for inference
        # Only run single sample for preview
        input_ids = next(iter(calib_dataloader))["input_ids"][0:1]
        if not args.medusa and not args.vlm:
            generated_ids_before_ptq = model.generate(input_ids, max_new_tokens=100)

        model = quantize_model(model, quant_cfg, args, calib_dataloader)
        # Lets print the quantization summary
        mtq.print_quant_summary(model)

        # Medusa generate can only run on single GPU per official implementation.
        # Therefore, we skip generate here in case multiple GPUs are needed for inference
        if not args.medusa and not args.vlm:
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
        export_dtype = torch.bfloat16
    else:
        export_dtype = torch.float16

    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.export_path

        # Move meta tensor back to device before exporting.
        remove_hook_from_module(model, recurse=True)

        start_time = time.time()
        if args.export_fmt == "tensorrt_llm":
            export_tensorrt_llm_checkpoint(
                model,
                model_type,
                export_dtype,
                export_dir=export_path,
                inference_tensor_parallel=args.inference_tensor_parallel,
                inference_pipeline_parallel=args.inference_pipeline_parallel,
                naive_fp8_quantization=args.naive_quantization,
            )
        elif args.export_fmt == "hf":
            export_hf(
                model,
                export_dtype,
                export_dir=export_path,
            )
        else:
            raise NotImplementedError(f"{args.export_fmt} not supported")

        # Export the tokenizer as well.
        tokenizer.save_pretrained(export_path)

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
    parser.add_argument(
        "--qformat",
        help=(
            "Quantization format. If --auto_quantize_compression is set, this argument specifies the quantization "
            "format for optimal per-layer AutoQuantize search."
        ),
        default="fp8",
    )
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
        "--auto_quantize_compression",
        default=None,
        type=float,
        help=(
            "Weight compression constraint for AutoQuantize. If not set, "
            "regular quantization without AutoQuantize search will be applied."
        ),
    )
    parser.add_argument(
        "--vlm",
        help="Specify whether this is a visual-language model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--export_fmt",
        required=False,
        default="tensorrt_llm",
        choices=["tensorrt_llm", "hf"],
        help=("Checkpoint export format"),
    )
    args = parser.parse_args()

    main(args)
