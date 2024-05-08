# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/dfa685fc0a5102346e3048dcfc9db8096d7d2378/python/models/llama2/LLaMA-2%20E2E%20Notebook.ipynb
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

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
import logging
import os
import time

import numpy as np
import onnx
import torch
from datasets import load_dataset
from transformers import LlamaConfig, LlamaTokenizer

from modelopt.onnx.quantization.int4 import quantize_int4

# Set logging level to info
logging.getLogger().setLevel(logging.INFO)

pt_to_np = {"torch.int64": np.int64, "torch.float32": np.float32, "torch.float16": np.float16}


def get_initial_inputs(config, tokenizer, prompt, device, use_fp16, use_buffer_share):
    """
    Get initial inputs for the model for inference.
    These values will include the input_ids, attention_mask, and position_ids and past_key_values.

    Args:
        config: Huggingface config of the model.
        tokenizer: Tokenizer to encode and decode text.
        prompt: List of prompts to be supplied for inference.
        device: Device used to run the inference.
        use_fp16: Flag to select the float16 dtype in torch.
        use_buffer_share: True when --use_gqa is passed during the onnx export process
    """
    tokenizer.pad_token = "[PAD]"
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int64)
    attention_mask = torch.tensor(
        encodings_dict["attention_mask"], device=device, dtype=torch.int64
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    inputs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "position_ids": position_ids.contiguous(),
    }

    batch_size, sequence_length = input_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads, head_size = (
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    for i in range(config.num_hidden_layers):
        past_key = torch.zeros(
            batch_size,
            num_heads,
            max_sequence_length if use_buffer_share else 0,
            head_size,
            device=device,
            dtype=torch_dtype,
        )
        past_value = torch.zeros(
            batch_size,
            num_heads,
            max_sequence_length if use_buffer_share else 0,
            head_size,
            device=device,
            dtype=torch_dtype,
        )
        inputs.update(
            {
                f"past_key_values.{i}.key": past_key.contiguous(),
                f"past_key_values.{i}.value": past_value.contiguous(),
            }
        )

    return inputs


def main(args):
    # User settings
    cache_dir = "/tmp/cache_dir"
    calib_size = 32

    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    prompt = dataset["article"][:calib_size]
    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    # device = torch.device("cpu")

    config = LlamaConfig.from_pretrained(args.model_name, use_auth_token=True, cache_dir=cache_dir)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name, use_auth_token=True, cache_dir=cache_dir
    )

    # Get model and its initial inputs
    inputs = get_initial_inputs(
        config, tokenizer, prompt, device, args.use_fp16, args.use_buffer_share
    )

    logging.info("Quantizing the model using awq_clip algorithm...")
    onnx_model = onnx.load(args.onnx_path)
    inputs = {input_name: torch_tensor.cpu().numpy() for input_name, torch_tensor in inputs.items()}

    t = time.time()
    quantized_onnx_model = quantize_int4(args.quantize_mode, onnx_model, [inputs])
    logging.info(f"Quantization process took {time.time() - t} seconds")

    t = time.time()
    onnx.save_model(
        quantized_onnx_model,
        args.output_path,
        save_as_external_data=True,
        location=os.path.basename(args.output_path) + "_data",
        size_threshold=0,
    )
    logging.info(f"Saving to {args.output_path} took {time.time() - t} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize llama model in INT4 mode.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the llama model variant."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Exported llama ONNX model path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output quantized model path.",
    )
    parser.add_argument(
        "--use_fp16",
        type=bool,
        default=False,
        help="True when KV cache inputs/outputs are in float16.",
    )
    parser.add_argument(
        "--use_buffer_share",
        type=bool,
        default=False,
        help="True when --use_gqa was passed during export.",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="int4_awq_clip",
        choices=["int4_awq_clip", "int4_rtn_dq"],
        help="Algorithm to use for the quantization process. Supported options: int4_awq_clip, int4_rtn_dq",
    )

    args = parser.parse_args()
    main(args)
