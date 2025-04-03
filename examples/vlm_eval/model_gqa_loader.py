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
import copy
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
)
from modelopt.torch.utils.image_processor import MllamaImageProcessor
from modelopt.torch.utils.vlm_dataset_utils import get_vlm_dataset_dataloader

sys.path.append(str(Path(__file__).resolve().parent / "../llm_ptq"))
sys.path.append(str(Path(__file__).resolve().parent / "../vlm_ptq"))
from example_utils import get_processor, get_tokenizer
from utils import add_common_args
from vlm_eval_utils import save_jsonl


def quantize_model(model, args, tokenizer, processor=None):
    sample_memory_usage_ratio = (
        2 if "AWQ" in args.quant_cfg or "SMOOTHQUANT" in args.quant_cfg else 1.1
    )
    batch_size = get_max_batch_size(model, sample_memory_usage_ratio=sample_memory_usage_ratio)
    calib_size = args.calib_size
    if batch_size > calib_size:
        batch_size = calib_size

    # Handle Mllama models with VLM dataset
    if processor is not None and isinstance(processor, MllamaImageProcessor):
        calib_dataloader = get_vlm_dataset_dataloader(
            dataset_name="scienceqa",  # Default dataset for Mllama
            processor=processor,
            batch_size=batch_size,
            num_samples=calib_size,
        )
    else:
        calib_dataloader = get_dataset_dataloader(
            dataset_name="cnn_dailymail",
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=calib_size,
            device=model.device,
        )
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    quant_cfg = getattr(mtq, args.quant_cfg)
    if "AWQ" in args.quant_cfg:
        quant_cfg = copy.deepcopy(getattr(mtq, args.quant_cfg))
        weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
    enable_quant_kv_cache = args.quant_cfg not in ["INT8_SMOOTHQUANT_CFG"]
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")
    quant_cfg["quant_cfg"]["*output_quantizer"] = {
        "num_bits": 8 if args.quant_cfg == "INT8_SMOOTHQUANT_CFG" else (4, 3),
        "axis": None,
        "enable": enable_quant_kv_cache,
    }

    print("Starting quantization...")
    start_time = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument(
        "--quant_cfg",
        type=str,
        default=None,
        help="Specify the modelopt quantization configuration for simulated evaluation",
        choices=[
            "INT8_SMOOTHQUANT_CFG",
            "FP8_DEFAULT_CFG",
            "INT4_AWQ_CFG",
            "W4A8_AWQ_BETA_CFG",
        ],
    )
    parser.add_argument(
        "--calib_size", type=int, default=512, help="Number of samples for calibration."
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    # Load data
    instances = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev", token=True
    )
    images = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
    id2image = {}
    for row in images:
        id2image[row["id"]] = row["image"].convert("RGB")

    # Load model
    if args.llm_engine_dir is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.set_level(args.log_level)
        # Load TensorRT engine
        model = MultimodalModelRunner(args)
        # Run batch inference
        outputs = []
        batch_size = args.batch_size
        if model.model_type in ["phi-3-vision"]:
            # Phi-3-vision doesn't support batch inference for now
            batch_size = 1
        for index in tqdm(range(0, len(instances), batch_size)):
            batch = instances[index : index + batch_size]
            raw_images = [id2image[imageId] for imageId in batch["imageId"]]
            questions = batch["question"]
            questions = [
                q + "\nAnswer the question using a single word or phrase." for q in questions
            ]
            if model.model_type in ["llava"]:
                input_text = ["\n" + question for question in questions]
            elif model.model_type in ["vila"]:
                input_text = ["<image>\n" + question for question in questions]
            elif model.model_type in ["phi-3-vision"]:
                input_text = questions[0]
            _, output_text = model.run(input_text, raw_images, args.max_new_tokens)
            outputs.extend(
                [
                    {
                        "question_id": id,
                        "prompt": batch["question"][index],
                        "text": output_text[index][0],
                    }
                    for index, id in enumerate(batch["id"])
                ]
            )

    else:
        # Load HF model
        if "vila" in args.hf_model_dir.lower():
            sys.path.append(os.path.join(args.hf_model_dir, "..", "VILA"))
            import llava

            model = llava.load(args.hf_model_dir)
            from llava import conversation as conversation_lib

            if "8b" in args.hf_model_dir.lower():
                conv_mode = "llama_3"
            elif "40b" in args.hf_model_dir.lower():
                conv_mode = "hermes-2"
            else:
                conv_mode = "vicuna_v1"

            conversation_lib.default_conversation = conversation_lib.conv_templates[
                conv_mode
            ].copy()

            generation_config = GenerationConfig.from_pretrained(args.hf_model_dir + "/llm")
            generation_config.update(**{"max_new_tokens": args.max_new_tokens})
        elif "llama" in args.hf_model_dir.lower():
            model = MllamaForConditionalGeneration.from_pretrained(
                args.hf_model_dir,
                device_map="auto",
                trust_remote_code=args.trust_remote_code,
                torch_dtype="auto",
            )
            # processor = AutoProcessor.from_pretrained(args.hf_model_dir)
            processor = get_processor(
                args.hf_model_dir, "mllama", model.device, trust_remote_code=args.trust_remote_code
            )

        else:
            processor = AutoProcessor.from_pretrained(
                args.hf_model_dir, trust_remote_code=args.trust_remote_code
            )
            if "llava" in args.hf_model_dir.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    args.hf_model_dir, device_map="auto", torch_dtype="auto"
                )
                # To be deprecated for new version transformers
                processor.patch_size = model.config.vision_config.patch_size
                processor.vision_feature_select_strategy = (
                    model.config.vision_feature_select_strategy
                )
            elif "phi" in args.hf_model_dir.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    args.hf_model_dir,
                    device_map="auto",
                    trust_remote_code=args.trust_remote_code,
                    torch_dtype="auto",
                    _attn_implementation="flash_attention_2",
                )
            else:
                raise ValueError(f"Unsupported model: {args.hf_model_dir}")
        # Evaluation for simulated quantization
        if args.quant_cfg:
            tokenizer = get_tokenizer(args.hf_model_dir, trust_remote_code=args.trust_remote_code)
            if "vila" in args.hf_model_dir.lower():
                model.llm = quantize_model(model.llm, args, tokenizer)
            elif "llava" in args.hf_model_dir.lower():
                model.language_model = quantize_model(model.language_model, args, tokenizer)
            elif "phi" in args.hf_model_dir.lower():
                model = quantize_model(model, args, tokenizer)
            elif "llama" in args.hf_model_dir.lower():
                model = quantize_model(model, args, tokenizer, processor)
            else:
                raise ValueError(f"Unsupported model: {args.hf_model_dir}")
        if "llama" in args.hf_model_dir.lower():
            processor = processor.tokenizer

        outputs = []
        for instance in tqdm(instances):
            image = id2image[instance["imageId"]]
            question = instance["question"]
            if "vila" in args.hf_model_dir.lower():
                response = model.generate_content(
                    [image, question], generation_config=generation_config
                )
            else:
                if "llava" in args.hf_model_dir.lower():
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image"},
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                        "cuda:0", model.dtype
                    )
                    response = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                    )
                elif "phi" in args.hf_model_dir.lower():
                    conversation = [
                        {"role": "user", "content": f"<|image_1|>\n{question}"},
                    ]
                    prompt = processor.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                        "cuda:0", model.dtype
                    )
                    response = model.generate(
                        **inputs,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                elif "llama" in args.hf_model_dir.lower():
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": question
                                    + "\nAnswer the question using a single word or phrase.",
                                },
                            ],
                        }
                    ]
                    prompt = processor.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )

                    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0", model.dtype)
                    response = model.generate(
                        **inputs,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                else:
                    raise ValueError(f"Unsupported model: {args.hf_model_dir}")
                response = processor.decode(
                    response[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )

            outputs.append({"question_id": instance["id"], "prompt": question, "text": response})
    save_jsonl(args.answers_file, outputs)


if __name__ == "__main__":
    main()
