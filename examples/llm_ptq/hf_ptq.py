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
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from example_utils import (
    apply_kv_cache_quant,
    build_quant_cfg,
    copy_custom_model_files,
    get_model,
    get_processor,
    get_tokenizer,
    is_enc_dec,
)
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    WhisperProcessor,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
from modelopt.torch.export import (
    export_hf_checkpoint,
    export_tensorrt_llm_checkpoint,
    get_model_type,
)
from modelopt.torch.export.model_utils import is_multimodal_model
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.plugins.accelerate import init_quantized_weights
from modelopt.torch.quantization.utils import is_quantized
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
    get_supported_datasets,
)
from modelopt.torch.utils.image_processor import MllamaImageProcessor
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from modelopt.torch.utils.speech_dataset_utils import get_speech_dataset_dataloader
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader

RAND_SEED = 1234

QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "fp8_pb_wo": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    "w4a8_nvfp4_fp8": mtq.W4A8_NVFP4_FP8_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
}

mto.enable_huggingface_checkpointing()


def _run_vl_preview_generation(model, tokenizer, model_path, stage_name):
    """Run preview generation for VL models using sample images.

    Args:
        model: The VL model
        tokenizer: The tokenizer
        model_path: Path to the model (for loading image processor)
        stage_name: Description of the stage (e.g., "before quantization")

    Returns:
        Generated response text for logging/comparison
    """
    import os

    from PIL import Image
    from transformers import AutoImageProcessor

    try:
        print(f"Loading sample images for {stage_name} preview...")

        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load sample images from the images directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "images")

        image_files = ["example1a.jpeg", "example1b.jpeg"]
        images = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
                print(f"  Loaded: {img_file}")
            else:
                print(f"  Warning: {img_file} not found")

        if not images:
            print("No sample images found - skipping VL preview generation")
            return None

        # Process images
        image_features = image_processor(images)

        # Move image features to the same device as the model
        model_device = model.device
        for key, value in image_features.items():
            if hasattr(value, "to"):  # Check if it's a tensor
                image_features[key] = value.to(model_device)
                print(f"  Moved {key} to {model_device}")

        # Generate response
        question = "Describe these images briefly."
        generation_config = {
            "max_new_tokens": 50,
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
        }

        print(f"Generating VL response ({stage_name})...")
        response = model.chat(
            tokenizer=tokenizer,
            question=question,
            generation_config=generation_config,
            **image_features,
        )

        print(f"✅ VL generation {stage_name} successful!")
        print(f"Question: {question}")
        print(f"Response: {response}")

        # Return the response for comparison/logging
        return response

    except Exception as e:
        print(f"❌ VL preview generation {stage_name} failed: {e}")
        print("This may indicate issues with the quantized model")
        return None


def auto_quantize(
    model, qformat, auto_quantize_bits, calib_dataloader, calibrate_loop, batch_size=1
):
    qformat_list = qformat.split(",")
    assert qformat_list, "No quantization formats provided"
    # Check if all provided quantization formats are supported
    assert all(
        qformat
        in [
            "fp8",
            "int8_sq",
            "int8_wo",
            "int4_awq",
            "nvfp4",
            "nvfp4_awq",
            "w4a8_awq",
            "fp8_pb_wo",
            "w4a8_mxfp4_fp8",
            "nvfp4_mlp_only",
        ]
        for qformat in qformat_list
    ), "One or more quantization formats provided are not supported for unified checkpoint export"

    def loss_func(output, data):
        # For transformers AutoModelForCausalLM models, the outputs are wrapped in `CausalLMOutputWithPast`
        # which contains the loss attribute.
        return output.loss

    model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": auto_quantize_bits},
        data_loader=calib_dataloader,
        forward_step=lambda model, batch: model(**batch),
        loss_func=loss_func,
        # TRTLLM only support one quantization format or None (do not quantize, internally supported)
        quantization_formats=[QUANT_CFG_CHOICES[format] for format in qformat_list],
        num_calib_steps=len(calib_dataloader),
        num_score_steps=len(calib_dataloader),
        verbose=True,
        disabled_layers=["*lm_head*"],
    )

    # We need to explicitly calibrate for kv cache quantization
    enable_quant_kv_cache = args.kv_cache_qformat != "none"
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")
    if enable_quant_kv_cache:
        kv_cache_quant_cfg = getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"]
        kv_cache_quant_cfg.pop("default")  # keep other quantizers from auto_quantize

        mtq.set_quantizer_by_cfg(
            model,
            quant_cfg=kv_cache_quant_cfg,
        )
        # Lets calibrate only the quantizers for kv cache quantization this time. Let's disable all others.
        with mtq.set_quantizer_by_cfg_context(
            model, {"*": {"enable": False}, **kv_cache_quant_cfg}
        ):
            mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
    return model


def quantize_model(model, quant_cfg, args, calib_dataloader=None, calibration_only=False):
    # The calibration loop for the model can be setup using the modelopt API.
    #
    # Example usage:
    # from modelopt.torch.utils.dataset_utils import create_forward_loop
    # model = ...  # Initialize the model
    # tokenizer = ...  # Initialize the tokenizer
    # quant_cfg = ...  # Setup quantization configuration
    # forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    # mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    # The calibrate_loop is a custom defined method to run the model with the input data.
    # The basic version looks like:
    #
    # def calibrate_loop(model, dataloader):
    #     for data in dataloader:
    #         model(**data)
    #
    # We also provided a util method to generate the forward_loop with additional error handlings.

    use_calibration = args.auto_quantize_bits or need_calibration(quant_cfg)

    if not use_calibration:
        warnings.warn("Dynamic quantization. Calibration skipped.")
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader) if use_calibration else None

    assert not (args.auto_quantize_bits and args.inference_pipeline_parallel > 1), (
        "Auto Quantization is not supported for pipeline parallel size > 1"
    )

    print("Starting quantization...")
    start_time = time.time()
    if args.auto_quantize_bits:
        model = auto_quantize(
            model,
            args.qformat,
            args.auto_quantize_bits,
            calib_dataloader,
            calibrate_loop,
            args.batch_size,
        )
    elif calibration_only:
        model = mtq.calibrate(model, quant_cfg["algorithm"], forward_loop=calibrate_loop)
    else:
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


def main(args):
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    # launch a memory monitor to read the currently used GPU memory.
    launch_memory_monitor()

    # Force eager execution for all model types.
    torch.compiler.set_stance("force_eager")

    # Check that only one quantization format is provided for non auto_quant case
    if not args.auto_quantize_bits:
        assert len(args.qformat.split(",")) == 1, (
            "Quantization supports only one quantization format."
        )

    if not args.auto_quantize_bits:
        assert (
            args.qformat
            in [
                "int8_wo",
                "int4_awq",
                "fp8",
                "nvfp4",
                "nvfp4_awq",
                "w4a8_awq",
                "fp8_pb_wo",
                "w4a8_mxfp4_fp8",
                "nvfp4_mlp_only",
            ]
            or args.kv_cache_qformat in KV_QUANT_CFG_CHOICES
        ), f"Quantization format {args.qformat} not supported for HF export path"

    # If low memory mode is enabled, we compress the model while loading the HF checkpoint.
    calibration_only = False
    if not args.low_memory_mode:
        model = get_model(
            args.pyt_ckpt_path,
            args.device,
            gpu_mem_percentage=args.gpu_max_mem_percentage,
            trust_remote_code=args.trust_remote_code,
            use_seq_device_map=args.use_seq_device_map,
            attn_implementation=args.attn_implementation,
        )
    else:
        assert args.qformat in QUANT_CFG_CHOICES, (
            f"Quantization format is not supported for low memory mode. Supported formats: {QUANT_CFG_CHOICES.keys()}"
        )
        quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        if args.kv_cache_qformat != "none":
            quant_cfg = apply_kv_cache_quant(
                quant_cfg, getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"]
            )

        # Do not use real quant GEMM so the calibration can be more accurate.
        with init_quantized_weights(
            quant_cfg, gpu_mem_percentage=args.gpu_max_mem_percentage, quant_gemm=False
        ):
            model_kwargs = {"trust_remote_code": args.trust_remote_code}
            if args.attn_implementation is not None:
                model_kwargs["attn_implementation"] = args.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(
                args.pyt_ckpt_path,
                **model_kwargs,
            )
        calibration_only = True
    model_is_already_quantized = is_quantized(model)

    model_type = get_model_type(model)

    # Special handling for Nemotron VL models that aren't detected by standard model type detection
    # For HF export, we want to keep vision unquantized, so we treat it as a regular language model
    # and only quantize the language components
    if model_type != "mllama" and is_multimodal_model(model):
        print(
            f"Detected multimodal model: {type(model).__name__}. "
            f"For HF export, will quantize language components only, keeping vision unquantized."
        )
        # Keep as regular model type to use text-only calibration

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device
    processor = None
    tokenizer = None

    full_model = model

    if model_type == "mllama":
        processor = get_processor(
            args.pyt_ckpt_path,
            model_type,
            device,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif model_type == "whisper":
        processor = get_processor(
            args.pyt_ckpt_path, model_type, device, trust_remote_code=args.trust_remote_code
        )
    else:
        if args.dataset is None:
            args.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
            warnings.warn(
                "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
            )
        # Adjust calib_size to match dataset length by extending or truncating as needed
        args.calib_size = (args.calib_size + [args.calib_size[-1]] * len(args.dataset))[
            : len(args.dataset)
        ]
        tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)

        default_padding_side = tokenizer.padding_side
        # Left padding usually provides better calibration result.
        tokenizer.padding_side = "left"

        # We only quantize the language model for VLMs other than the type supported above.
        if hasattr(model, "language_model"):
            parent_model = model  # llama4 case
            if isinstance(type(model).__dict__.get("language_model"), property):
                assert hasattr(model, "model") and hasattr(model.model, "language_model"), (
                    "Expected language_model in model.model, but attribute not found. "
                    "This may indicate an unsupported model structure."
                )
                parent_model = model.model  # gemma3, qwen2.5 VL case

            disabled_quant_cfg = {
                "quant_cfg": {"default": {"enable": False}},
                "algorithm": "max",
            }

            for name, child in parent_model.named_children():
                # Apply disabled quant to all children except language_model so we can exclude them during HF export.
                if name != "language_model":
                    mtq.quantize(child, disabled_quant_cfg, forward_loop=None)

            model = model.language_model
            model_type = get_model_type(model)

    if model_type == "phi4mm":
        warnings.warn("Please set the default input_mode to InputMode.LANGUAGE before quantizing.")

    if args.sparsity_fmt != "dense":
        if args.batch_size == 0:
            # Sparse algorithm takes more GPU memory so we reduce the batch_size by 4.
            args.batch_size = max(get_max_batch_size(model) // 4, 1)
            args.batch_size = min(args.batch_size, sum(args.calib_size))

        print(f"Use calib batch_size {args.batch_size}")

        # Different calibration datasets are also available, e.g., "pile" and "wikipedia"
        # Please also check the docstring for the datasets available
        assert tokenizer is not None and isinstance(
            tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        ), "The PreTrainedTokenizer must be set"
        calib_dataloader = get_dataset_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            max_sample_length=args.calib_seq,
            device=device,
        )
        model = mts.sparsify(
            model,
            args.sparsity_fmt,
            config={"data_loader": calib_dataloader, "collect_func": lambda x: x},
        )
        mts.export(model)

    if args.auto_quantize_bits or args.qformat in QUANT_CFG_CHOICES:
        if "awq" in args.qformat:
            print(
                "\n####\nAWQ calibration could take longer than other calibration methods. "
                "Consider reducing calib_size to reduce calibration time.\n####\n"
            )

        if args.batch_size == 0:
            # Calibration/sparsification will actually take much more memory than regular inference
            # due to intermediate tensors for fake quantization. Setting sample_memory_usage_ratio
            # to 2 to avoid OOM for AWQ/SmoothQuant fake quantization as it will take more memory than inference.
            sample_memory_usage_ratio = 2 if "awq" in args.qformat or "sq" in args.qformat else 1.1
            # Whisper model expects mel-spectrogram input features of length 3000
            # Whisper model needs input of shape (batch_size, num_mel_bins, 3000)
            # As the encoder of Whisper doesn't have embedding layer, input dtype has to be float
            # For non-Whisper models (language models), sample_input will be set up inside get_max_batch_size()
            if model_type == "whisper":
                max_sample_length = 3000
                num_mel_bins = model.config.num_mel_bins
                sample_input_single_batch = (
                    torch.ones([1, num_mel_bins, max_sample_length], dtype=model.dtype).to(
                        model.device
                    )
                    * 100
                )
            else:
                sample_input_single_batch = None

            run_auto_quant = args.auto_quantize_bits is not None

            args.batch_size = get_max_batch_size(
                model,
                max_sample_length=args.calib_seq,
                sample_memory_usage_ratio=sample_memory_usage_ratio if not run_auto_quant else 1.0,
                sample_input_single_batch=sample_input_single_batch,
                enable_grad=run_auto_quant,
            )
            args.batch_size = min(args.batch_size, sum(args.calib_size))

        print(f"Use calib batch_size {args.batch_size}")

        calib_dataloader = None
        if model_type == "mllama":
            assert processor is not None and isinstance(processor, MllamaImageProcessor), (
                "The MllamaImageProcessor must be set."
            )
            assert len(args.calib_size) == 1, (
                "mllama only supports one dataset for calibration, can extend this in the future"
            )
            calib_dataloader = get_vlm_dataset_dataloader(
                dataset_name=args.dataset[0] if args.dataset else "scienceqa",
                processor=processor,
                batch_size=args.batch_size,
                num_samples=args.calib_size[0],
            )
        elif model_type == "whisper":
            assert processor is not None and isinstance(processor, WhisperProcessor), (
                "The AutoProcessor must be set."
            )
            assert len(args.calib_size) == 1, (
                "whisper only supports one dataset for calibration, can extend this in the future"
            )
            calib_dataloader, first_text = get_speech_dataset_dataloader(
                dataset_name=args.dataset[0] if args.dataset else "peoples_speech",
                processor=processor,
                batch_size=args.batch_size,
                num_samples=args.calib_size[0],
                device=device,
                dtype=model.dtype,
            )
        else:
            assert tokenizer is not None and isinstance(
                tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
            ), "The PreTrainedTokenizer must be set"
            calib_dataloader = get_dataset_dataloader(
                dataset_name=args.dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_samples=args.calib_size,
                device=device,
                include_labels=args.auto_quantize_bits is not None,
            )

        quant_cfg = build_quant_cfg(
            args.qformat,
            args.kv_cache_qformat,
            args.awq_block_size,
            args.auto_quantize_bits,
            model_type,
            QUANT_CFG_CHOICES,
            KV_QUANT_CFG_CHOICES,
        )

            # For Nemotron VL models, disable quantization of vision components
            is_nemotron_vl = (
                "nemotron" in args.pyt_ckpt_path.lower() and "vl" in args.pyt_ckpt_path.lower()
            )
            if is_nemotron_vl:
                print("Disabling quantization for vision components in Nemotron VL model")
                quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}
                quant_cfg["quant_cfg"]["*image*"] = {"enable": False}
                # Also disable radio model components specifically
                quant_cfg["quant_cfg"]["*radio*"] = {"enable": False}
                quant_cfg["quant_cfg"]["*visual*"] = {"enable": False}

        if not model_is_already_quantized or calibration_only:
            # Only run single sample for preview
            input_ids = next(iter(calib_dataloader))[
                "input_features" if model_type == "whisper" else "input_ids"
            ][0:1]

            # For Nemotron VL models, try text-only generation first, then VL generation as additional test
            is_nemotron_vl = (
                "nemotron" in args.pyt_ckpt_path.lower() and "vl" in args.pyt_ckpt_path.lower()
            )
            if is_nemotron_vl:
                print("Running text-only preview generation for Nemotron VL model...")
                try:
                    # Try text-only generation using model.chat with None for images
                    question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    generation_config = {
                        "max_new_tokens": 100,
                        "do_sample": False,
                        "eos_token_id": tokenizer.eos_token_id,
                    }

                    # Use model.chat with None for images (text-only mode)
                    text_response = full_model.chat(
                        tokenizer, None, question, generation_config, history=None
                    )
                    generated_ids_before_ptq = text_response  # Store text response
                    print(f"✅ Text-only generation successful: {text_response[:100]}...")

                except Exception as e:
                    print(f"Text-only generation failed: {e}")
                    print("Falling back to standard generate() method...")
                    try:
                        generated_ids_before_ptq = full_model.generate(
                            input_ids, max_new_tokens=100
                        )
                    except Exception as e2:
                        print(f"Standard generation also failed: {e2}")
                        generated_ids_before_ptq = None

                # Run additional VL test with images
                print("Running additional VL test with images...")
                _run_vl_preview_generation(
                    full_model, tokenizer, args.pyt_ckpt_path, "before quantization (VL test)"
                )

            else:
                try:
                    generated_ids_before_ptq = full_model.generate(input_ids, max_new_tokens=100)
                except Exception as e:
                    print(
                        "Error during model generation. Please check if your transformers version is "
                        "compatible with the model."
                    )
                    print(f"Error details: {e}")
                    raise
            if model_type == "gptoss" and args.qformat == "nvfp4_mlp_only":
                print("Applying nvfp4 quantization (MoE only) for gpt-oss")

            # quantize the model
            model = quantize_model(model, quant_cfg, args, calib_dataloader, calibration_only)

            # For VL models, update full_model to use the quantized language model
            if is_nemotron_vl and hasattr(full_model, "language_model"):
                print("Updating full_model with quantized language_model...")
                full_model.language_model = model
            if args.verbose:
                mtq.print_quant_summary(model)

            # Run some samples
            torch.cuda.empty_cache()
            generated_ids_after_ptq = None
            if model_type != "llama4" and not is_nemotron_vl:
                # Our fake quantizer may not be fully compatible with torch.compile.
                generated_ids_after_ptq = full_model.generate(input_ids, max_new_tokens=100)
            elif is_nemotron_vl:
                print("Running text-only preview generation for quantized Nemotron VL model...")
                try:
                    # Try text-only generation using model.chat with None for images
                    question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    generation_config = {
                        "max_new_tokens": 100,
                        "do_sample": False,
                        "eos_token_id": tokenizer.eos_token_id,
                    }

                    # Use model.chat with None for images (text-only mode)
                    text_response = full_model.chat(
                        tokenizer, None, question, generation_config, history=None
                    )
                    generated_ids_after_ptq = text_response  # Store text response
                    print(f"✅ Text-only generation successful: {text_response[:100]}...")

                except Exception as e:
                    print(f"Text-only generation failed: {e}")
                    generated_ids_after_ptq = None

                # Run additional VL test with images
                print("Running additional VL test with images...")
                _run_vl_preview_generation(
                    full_model, tokenizer, args.pyt_ckpt_path, "after quantization (VL test)"
                )

            else:
                warnings.warn(
                    "Llama4 Maverick generation after quantization has a bug. Skipping generation sample."
                )

            def input_decode(input_ids):
                if processor is not None and isinstance(processor, MllamaImageProcessor):
                    return processor.tokenizer.batch_decode(input_ids)
                elif processor is not None and isinstance(processor, WhisperProcessor):
                    return first_text
                elif tokenizer is not None:
                    return tokenizer.batch_decode(input_ids)
                else:
                    raise ValueError("The processor or tokenizer must be set")

            def output_decode(generated_ids, input_shape):
                if is_enc_dec(model_type):
                    if processor is not None and isinstance(processor, WhisperProcessor):
                        return processor.tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0]
                    elif tokenizer is not None:
                        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                elif processor is not None and isinstance(processor, MllamaImageProcessor):
                    return processor.tokenizer.batch_decode(generated_ids[:, input_shape:])
                elif tokenizer is not None:
                    return tokenizer.batch_decode(generated_ids[:, input_shape:])
                else:
                    raise ValueError("The processor or tokenizer must be set")

            if generated_ids_after_ptq is not None:
                print("--------")
                if is_nemotron_vl:
                    # For Nemotron VL models, generated_ids are text strings from model.chat()
                    print("Nemotron VL model text-only generation results:")
                    print(f"Text response before quantization: {generated_ids_before_ptq}")
                    print("--------")
                    print(f"Text response after quantization: {generated_ids_after_ptq}")
                    print("--------")
                    print("Note: Additional VL tests with images were run separately above")
                else:
                    # For regular LLMs, generated_ids are token tensors that need decoding
                    print(f"example test input: {input_decode(input_ids)}")
                    print("--------")
                    print(
                        f"example outputs before ptq: {output_decode(generated_ids_before_ptq, input_ids.shape[1])}"
                    )
                    print("--------")
                    print(
                        f"example outputs after ptq: {output_decode(generated_ids_after_ptq, input_ids.shape[1])}"
                    )
        else:
            warnings.warn("Skipping quantization: model is already quantized.")

    else:
        assert model_type != "dbrx", f"Does not support export {model_type} without quantizaton"
        print(f"qformat: {args.qformat}. No quantization applied, export {device} model")

    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.export_path

        # Check if the model is a multimodal/VLM model
        is_vlm = is_multimodal_model(full_model)

        if is_vlm:
            # Save original model config and the processor config to the export path for VLMs.
            print(f"Saving original model config to {export_path}")

            config_kwargs = {"trust_remote_code": args.trust_remote_code}
            if args.attn_implementation is not None:
                config_kwargs["attn_implementation"] = args.attn_implementation
            AutoConfig.from_pretrained(args.pyt_ckpt_path, **config_kwargs).save_pretrained(
                export_path
            )

            # Try to save processor config if available (skip for Nemotron VL models)
            if not is_nemotron_vl:
                try:
                    print(f"Saving processor config to {export_path}")
                    AutoProcessor.from_pretrained(
                        args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
                    ).save_pretrained(export_path)
                except Exception as e:
                    print(f"Warning: Could not save processor config: {e}")
                    print("This is normal for some VLM architectures that don't use AutoProcessor")
            else:
                print("Skipping AutoProcessor for Nemotron VL (uses separate AutoImageProcessor)")

            # For Nemotron VL models, save image processor using proper HuggingFace APIs
            if is_nemotron_vl:
                import os
                import shutil

                # Try to save image processor config using HuggingFace API
                try:
                    print("Saving image processor config using AutoImageProcessor...")
                    image_processor = AutoImageProcessor.from_pretrained(
                        args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
                    )
                    image_processor.save_pretrained(export_path)
                    print("  ✅ Image processor config saved successfully")
                except Exception as e:
                    print(f"  Warning: Could not save image processor config: {e}")

                # Manually copy image_processing.py as it contains custom code that save_pretrained doesn't handle
                print("Copying custom image processing implementation...")
                src_path = os.path.join(args.pyt_ckpt_path, "image_processing.py")
                dst_path = os.path.join(export_path, "image_processing.py")

                if os.path.exists(src_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                        print("  ✅ Copied: image_processing.py")
                    except Exception as copy_e:
                        print(f"  Warning: Could not copy image_processing.py: {copy_e}")
                else:
                    print("  Warning: image_processing.py not found in source model")

        if model_type == "mllama":
            full_model_config = model.config
            model = model.language_model
            # TRT-LLM expects both the vision_config and text_config to be set for export.
            setattr(model.config, "vision_config", full_model_config.vision_config)
            setattr(model.config, "text_config", full_model_config.text_config)
            setattr(model.config, "architectures", full_model_config.architectures)

        start_time = time.time()
        if (
            model_type in ["t5", "bart", "whisper"]
            or args.sparsity_fmt != "dense"
            or "int8_sq" in args.qformat
        ):
            warnings.warn(
                "Still exporting TensorRT-LLM checkpoints for models not supported by the TensorRT-LLM torch runtime."
            )

            # Move meta tensor back to device before exporting.
            remove_hook_from_module(model, recurse=True)

            export_tensorrt_llm_checkpoint(
                model,
                model_type,
                export_dir=export_path,
                inference_tensor_parallel=args.inference_tensor_parallel,
                inference_pipeline_parallel=args.inference_pipeline_parallel,
            )

            # Copy custom model files (Python files and JSON configs) for TensorRT-LLM export
            copy_custom_model_files(args.pyt_ckpt_path, export_path, args.trust_remote_code)
        else:
            # Check arguments for unified_hf export format and set to default if unsupported arguments are provided
            assert args.sparsity_fmt == "dense", (
                f"Sparsity format {args.sparsity_fmt} not supported by unified export api."
            )

            if args.inference_tensor_parallel != 1 or args.inference_pipeline_parallel != 1:
                warnings.warn(
                    "Unified HF export format does not specify inference tensor parallel or pipeline parallel. "
                    "They will be set at deployment time."
                )

            export_hf_checkpoint(
                full_model,
                export_dir=export_path,
            )

        # Copy custom model files (Python files and JSON configs) if trust_remote_code is used
        copy_custom_model_files(args.pyt_ckpt_path, export_path, args.trust_remote_code)

        # Restore default padding and export the tokenizer as well.
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            tokenizer.save_pretrained(export_path)

        end_time = time.time()
        print(
            f"Quantized model exported to :{export_path}. Total time used {end_time - start_time}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        help="Specify where the PyTorch checkpoint path is",
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qformat",
        help=(
            "Quantization format. If --auto_quantize_bits is set, this argument specifies the quantization "
            "format for optimal per-layer auto_quantize search."
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
        "--calib_size",
        help=(
            "Number of samples for calibration. If a comma separated list of values is provided, "
            "each value will be used as the calibration size for the corresponding dataset. "
            "This argument will be parsed and converted as a list of ints."
        ),
        type=str,
        default="512",
    )
    parser.add_argument(
        "--calib_seq",
        help="Maximum sequence length for calibration.",
        type=int,
        default=512,
    )
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument(
        "--dataset",
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
        type=str,
        default=None,
    )
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--awq_block_size", default=0, type=int)
    parser.add_argument(
        "--sparsity_fmt",
        help="Sparsity format.",
        default="dense",
        choices=["dense", "sparsegpt"],
    )
    parser.add_argument(
        "--auto_quantize_bits",
        default=None,
        type=float,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization without auto_quantize search will be applied."
        ),
    )
    parser.add_argument(
        "--kv_cache_qformat",
        required=False,
        default="fp8",
        choices=KV_QUANT_CFG_CHOICES.keys(),
        help="Specify KV cache quantization format, default to fp8 if not provided",
    )
    parser.add_argument(
        "--export_fmt",
        required=False,
        default="hf",
        choices=["tensorrt_llm", "hf"],
        help="Deprecated. Please avoid using this argument.",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu_max_mem_percentage",
        help=(
            "Specify the percentage of available GPU memory to use for loading the model when "
            "device_map is set to sequential. "
            "By default, 80%% of the available GPU memory is used."
        ),
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--use_seq_device_map",
        help=(
            "Use device_map=sequential to load the model onto GPUs. This ensures the model is loaded "
            "utilizing the percentage of available GPU memory as specified by the value passed with gpu_max_mem flag."
            "Helpful in cases where device_map=auto loads model unevenly on GPUs causing GPU OOM during quantization."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="Print verbose output (e.g. quantization summary). Disable by --no-verbose.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--low_memory_mode",
        help=(
            "Use low memory mode for quantization."
            "This is an experimental feature and may not work for all quantization formats."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--attn_implementation",
        help=(
            "Specify the attention implementation to use. "
            "This arg will be passed to the HF model loading if specified."
        ),
        default="eager",
        type=str,
    )

    args = parser.parse_args()

    if args.export_fmt != "hf":
        warnings.warn("Deprecated. --export_fmt forced to hf.")

    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(num_sample) for num_sample in args.calib_size.split(",")]
    main(args)
