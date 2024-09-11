# Adapted from: https://github.com/lm-sys/FastChat/blob/322e9238dd586815ea405ddb70607958773a0464/fastchat/llm_judge/gen_model_answer.py

# Copyright 2024 Large Model Systems Organization (LMSYS)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import openai
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template, load_model
from fastchat.utils import str_to_torch_dtype
from tqdm import tqdm

from examples.public.llm_eval.quantization_utils import get_tokenizer
from modelopt.deploy.llm import LLM

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


# Model Optimizer modification
# Reference:
# https://github.com/lm-sys/FastChat/blob/5c0443edb8545babb37d495ade86c75ebbf68009/fastchat/llm_judge/common.py#L407
def chat_completion_openai(model, conv, temperature, top_p, max_tokens, api_dict=None):
    """Chat completion with the OpenAI API.

    Args:
        model: The model handle.
        conv: The FastChat conversation template.
        temperature: The temperature for sampling.
        top_p: The nucleus sampling parameter.
        max_tokens: The maximum number of tokens to generate.
        api_dict: The API settings.

    Returns:
        The generated output.
    """

    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    top_p,
    temperature,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    engine_dir,
    vocab_file,
    nim_model,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                top_p=top_p,
                temperature=temperature,
                engine_dir=engine_dir,
                vocab_file=vocab_file,
                nim_model=nim_model,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    top_p=None,
    temperature=None,
    engine_dir=None,
    vocab_file=None,
    nim_model=None,
):
    # Model Optimizer modification
    if engine_dir:
        if vocab_file:
            from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

            tokenizer = get_nemo_tokenizer(vocab_file)
        else:
            model_ckpt_path = model_path
            tokenizer = get_tokenizer(model_ckpt_path)
        if engine_dir:
            # get model type
            last_part = os.path.basename(engine_dir)
            model_type = last_part.split("_")[0]
            # Some models require to set pad_token and eos_token based on external config (e.g., qwen)
            if model_type == "qwen":
                tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
                tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

            assert LLM is not None, "tensorrt_llm APIs could not be imported."
            model = LLM(engine_dir, tokenizer=tokenizer)
        else:
            raise ValueError("engine_dir is required for TensorRT LLM inference.")
    elif not nim_model:
        model, tokenizer = load_model(
            model_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )

    for question in tqdm(questions):
        if not temperature:
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7
        top_p = top_p if top_p is not None else 1.0
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # Model Optimizer modification
                if nim_model:
                    api_key = os.getenv("OPENAI_API_KEY")

                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable is not set")
                    api_dict = {}
                    api_dict["api_base"] = os.getenv(
                        "OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1"
                    )
                    api_dict["api_key"] = api_key
                    output = chat_completion_openai(
                        nim_model, conv, temperature, top_p, max_new_token, api_dict
                    )
                    conv.update_last_message(output)
                    turns.append(output)
                    continue

                input_ids = tokenizer([prompt]).input_ids

                # Model Optimizer modification
                if temperature <= 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    if not engine_dir:
                        output_ids = model.generate(
                            torch.as_tensor(input_ids).cuda(),
                            do_sample=do_sample,
                            temperature=temperature,
                            max_new_tokens=max_new_token,
                            top_p=top_p,
                        )
                        if model.config.is_encoder_decoder:
                            output_ids = output_ids[0]
                        else:
                            output_ids = output_ids[0][len(input_ids[0]) :]
                    else:
                        # Model Optimizer modification
                        output_ids = model.generate_tokens(
                            [prompt],
                            max_new_tokens=max_new_token,
                            temperature=temperature,
                            top_p=top_p,
                            keep_input_prompt=False,
                        )[0]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "category": question["category"],
                "turns": turns,
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            qid = json.loads(line)["question_id"]
            answers[qid] = line

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="The nucleus sampling parameter.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument("--num-gpus-total", type=int, default=1, help="The total number of GPUs.")
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        help="The path to the TensorRT LLM engine directory.",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="The path to the vocabulary file.",
    )
    parser.add_argument(
        "--nim-model",
        type=str,
        help="The NIM model handle to use",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        top_p=args.top_p,
        temperature=args.temperature,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        engine_dir=args.engine_dir,
        vocab_file=args.vocab_file,
        nim_model=args.nim_model,
    )

    reorg_answer_file(answer_file)
