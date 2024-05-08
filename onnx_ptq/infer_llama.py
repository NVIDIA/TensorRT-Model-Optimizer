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

import numpy as np
import torch
from quantize_llama import get_initial_inputs
from transformers import LlamaConfig, LlamaTokenizer

# NOTE: modelopt.torch._deploy is an experimental API subject to significant changes.
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes


def create_dynamic_shapes(batch_size, max_sequence_length=64):
    """Creates a dictionary of dynamic shapes for the TRT engine.

    Args:
        batch_size: Batch size of the input tensors.
        sequence_length: Sequence length of the input tensors."""
    shape_dict = {
        "input_ids": [batch_size, max_sequence_length],
        "attention_mask": [batch_size, max_sequence_length + 1],
        "position_ids": [batch_size, max_sequence_length],
    }
    # TODO: Add dynamic shapes for past key values
    dynamic_shapes = {
        "optShapes": shape_dict,
    }
    return dynamic_shapes


def generate_trt_outputs(
    inputs, deployment, onnx_path, batch_size, sequence_length, tokenizer, config, max_length=64
):
    """Generates the TRT outputs for the given inputs and deployment configuration.

    Args:
        inputs: Dictionary of input tensors for the model.
        deployment: Dictionary of deployment configuration for the TRT engine.
        onnx_path: Path to the ONNX model checkpoint.
        batch_size: Batch size of the input tensors.
        sequence_length: Sequence length of the input tensors.
        tokenizer: Tokenizer to decode the output logits.
        config: Huggingface config of the model.
        max_length: Maximum length of the generated text.
    """
    onnx_bytes = OnnxBytes(onnx_path, external_data_format=True).to_bytes()
    client = RuntimeRegistry.get(deployment)
    compilation_args = {"dynamic_shapes": create_dynamic_shapes(batch_size)}
    compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)
    device_model = DeviceModel(client, compiled_model, metadata={})

    input_ids_id = 0
    attention_mask_id = 1
    position_ids_id = 2
    logits_id = 0
    all_token_ids = inputs["input_ids"].clone().to("cpu")
    batch_size, sequence_length = all_token_ids.shape

    current_length = sequence_length
    has_eos = torch.zeros(batch_size, device="cpu", dtype=torch.bool)

    while current_length <= max_length:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        trt_int4_output = device_model(inputs)
        if trt_int4_output[logits_id].shape[1] > 1:
            prompt_end_indices = inputs[attention_mask_id].sum(1) - 1
            idxs = (
                prompt_end_indices.unsqueeze(dim=1)
                .repeat(1, config.vocab_size)
                .view(batch_size, 1, config.vocab_size)
                .to("cpu")
            )
            next_token_logits = torch.gather(trt_int4_output[0], 1, idxs).squeeze()
        else:
            next_token_logits = trt_int4_output[logits_id][:, -1, :]

        # get the next tokens by taking the argmax
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Check if we previously reached EOS token id or if generated token id is EOS token id
        has_eos = has_eos | next_tokens == tokenizer.eos_token_id

        # Determine which new tokens to add to list of all token ids
        # Add EOS token ids for batch entries that ended early
        # (ragged batching scenario where some batch entries ended early and some haven't)
        tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape(
            [batch_size, 1]
        )
        # Append the tokens to the input sequences
        all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

        # Return early if all batch entries have reached EOS token id
        current_length += 1
        if torch.all(has_eos) or current_length > max_length:
            break

        # Update inputs for next inference run
        # We just pass the newly generated token as input_ids
        inputs[input_ids_id] = all_token_ids
        # We just pass the position id which the tokens above correspond to
        inputs[position_ids_id] = (
            torch.max(inputs[position_ids_id], dim=1)[0].reshape(batch_size, 1) + 1
        )
        # If the token generated is not a eos token, then we append 1 to the attention mask
        inputs[attention_mask_id] = torch.cat(
            [
                inputs[attention_mask_id].to("cpu"),
                (~has_eos).to(torch.int64).reshape(batch_size, 1),
            ],
            1,
        )

    trt_text = tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
    return trt_text


def decode_logits(logits, tokenizer):
    """Decodes the logits to text using the given tokenizer."""
    if type(logits) == np.ndarray:
        logits = torch.from_numpy(logits)
    token_ids = torch.argmax(logits, dim=-1)
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the llama model variant."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the ONNX checkpoint of the LLAMA model",
    )
    # TODO: Modify the example for higher batch sizes.
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the language model",
        default="I want to book a vacation to Hawaii. First, I need to ",
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

    args = parser.parse_args()
    cache_dir = "/tmp/cache_dir"

    prompt = [args.prompt]
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    config = LlamaConfig.from_pretrained(args.model_name, use_auth_token=True, cache_dir=cache_dir)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name, use_auth_token=True, cache_dir=cache_dir
    )

    inputs = get_initial_inputs(
        config, tokenizer, prompt, device, args.use_fp16, args.use_buffer_share
    )
    batch_size = inputs["input_ids"].shape[0]
    sequence_length = inputs["input_ids"].shape[1]

    int4_trt_deployment = {
        "version": "10.0",
        "runtime": "TRT",
        "accelerator": "GPU",
        "precision": "int4",
        "onnx_opset": "13",
    }

    int4_trt_text = generate_trt_outputs(
        inputs,
        int4_trt_deployment,
        args.onnx_path,
        batch_size,
        sequence_length,
        tokenizer,
        config,
    )

    print(f"TRT-llama response: {int4_trt_text}")


if __name__ == "__main__":
    main()
