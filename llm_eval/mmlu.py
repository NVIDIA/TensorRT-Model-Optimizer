# Adapted from https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/mmlu.py

# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
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

import os
import random
from argparse import Namespace
from typing import Union

import numpy as np
import pandas as pd
from fire import Fire
from modeling import EvalModel, select_model
from quantization_utils import MAX_SEQ_LEN, get_tokenizer, quantize_model
from tqdm import tqdm

try:
    from modelopt.deploy.llm import LLM
except ImportError:
    LLM = None  # type: ignore[misc]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RAND_SEED = 1234


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, model: Union[EvalModel, LLM], dev_df, test_df):
    cors = []
    all_probs = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        def check_valid_length(model, prompt):
            if isinstance(model, EvalModel):
                return model.check_valid_length(prompt)
            else:
                return len(model.tokenizer.encode(prompt)) <= model.max_input_len

        while not check_valid_length(model, prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        if isinstance(model, EvalModel):
            pred = model.run(prompt)
        elif isinstance(model, LLM):
            pred = model.generate_text([prompt], 2, keep_input_prompt=False)[0]

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(
    data_dir: str = "data/mmlu",
    ntrain: int = 5,
    quant_cfg: str = None,
    auto_quantize_compression: float = None,
    batch_size: int = 0,
    calib_size: int = 512,
    dtype: str = "bfloat16",
    **kwargs,
):
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    args = Namespace(**locals())
    print(locals())

    data_fullpath = os.path.join(data_dir, "test")

    subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(data_fullpath) if "_test.csv" in f]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in get_subcategories().values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}
    # Model Optimizer modification
    if vocab_file := kwargs.get("vocab_file", None):
        from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

        tokenizer = get_nemo_tokenizer(vocab_file)
    else:
        model_ckpt_path = kwargs["model_path"]
        tokenizer = get_tokenizer(model_ckpt_path)
    if kwargs.get("engine_dir", None):
        # get model type
        last_part = os.path.basename(kwargs["engine_dir"])
        model_type = last_part.split("_")[0]
        # Some models require to set pad_token and eos_token based on external config (e.g., qwen)
        if model_type == "qwen":
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

        assert LLM is not None, "tensorrt_llm APIs could not be imported."
        model = LLM(kwargs["engine_dir"], tokenizer=tokenizer)
    else:
        model = select_model(
            max_input_length=MAX_SEQ_LEN, max_output_length=2, dtype=dtype, **kwargs
        )
        assert isinstance(model, EvalModel)
        if quant_cfg:
            model.load()
            quantize_model(
                model=model,
                quant_cfg=quant_cfg,
                tokenizer=tokenizer,
                batch_size=batch_size,
                calib_size=calib_size,
                auto_quantize_compression=auto_quantize_compression,
            )

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[
            :ntrain
        ]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = evaluate(args, subject, model, dev_df, test_df)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    return weighted_acc


if __name__ == "__main__":
    Fire(main)
