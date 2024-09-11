# Adapted from https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/human_eval/main.py

# MIT License
#
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
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
# SOFTWARE.

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
from argparse import Namespace
from typing import Union

from fire import Fire
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from modeling import EvalModel, select_model
from quantization_utils import MAX_OUTPUT_LEN, MAX_SEQ_LEN, get_tokenizer, quantize_model
from tqdm import tqdm

from modelopt.deploy.llm import LLM


def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)

    return results


STOP_TOKENS = [
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
    "\n```",
    "<filename>",
    "<file_sep>",
    "<|endoftext|>",
    "<|file_separator|>",
]


def filter_code(completion: str) -> str:
    min_stop_index = len(completion)
    for stop_token in STOP_TOKENS:
        stop_index = completion.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return completion[:min_stop_index]


def gen_prompt(prompt: str) -> str:
    return prompt.strip()


def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))


def evaluate(model: Union[EvalModel, LLM], data_path: str, **kwargs) -> dict:
    dataset = read_problems(data_path)
    model_path = kwargs.get("model_path", "")
    n_sample = kwargs.get("n_sample", 1)
    best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = gen_prompt(prompt)
            temperature = best_temperature[n_sample]
            if isinstance(model, EvalModel):
                if temperature > 0:
                    completion = model.run(prompt, temperature=temperature, do_sample=True)
                else:
                    completion = model.run(prompt)
            elif isinstance(model, LLM):
                completion = model.generate_text(
                    [prompt], MAX_OUTPUT_LEN, temperature=temperature, keep_input_prompt=False
                )[0]

            completion = fix_indents(completion)
            sample = dict(task_id=task_id, completion=filter_code(completion))
            if i == 0:
                print("Prompt: ", "-" * 100)
                print(prompt)
                print("Completion: ", "-" * 100)
                print(sample["completion"])
            samples.append(sample)
            progress_bar.update(1)
    progress_bar.close()

    model_name = model_path.replace("/", "_")
    pred_filename = f"humaneval_{model_name}_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    return result


def main(
    data_path: str = "human_eval/HumanEval.jsonl.gz",
    quant_cfg: str = None,
    batch_size: int = 0,
    calib_size: int = 512,
    **kwargs,
):
    args = Namespace(**locals())
    print(locals())

    # Model Optimizer modification
    if vocab_file := kwargs.get("vocab_file", None):
        from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

        tokenizer = get_nemo_tokenizer(vocab_file)
    else:
        model_ckpt_path = kwargs["model_path"]
        tokenizer = get_tokenizer(model_ckpt_path)
    if kwargs.get("engine_dir", None):
        model = LLM(engine_dir=kwargs["engine_dir"], tokenizer=tokenizer)
    else:
        model = select_model(
            max_input_length=MAX_SEQ_LEN, max_output_length=MAX_OUTPUT_LEN, **kwargs
        )
        if quant_cfg:
            model.load()
            quantize_model(
                model=model,
                quant_cfg=quant_cfg,
                tokenizer=tokenizer,
                batch_size=batch_size,
                calib_size=calib_size,
            )

    result = evaluate(model, data_path, **kwargs)
    print(result)
    return result["pass@1"]


if __name__ == "__main__":
    Fire(main)
