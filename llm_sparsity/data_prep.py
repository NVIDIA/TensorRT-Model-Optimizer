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
import json
import os

from datasets import load_dataset

dataset_id = "cnn_dailymail"
dataset_config = "3.0.0"
text_column = "article"
summary_column = "highlights"
instruction_template = "Summarize the following news article:"


def preprocess_function(sample):
    # create list of samples
    inputs = []

    for i in range(0, len(sample[text_column])):
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample[text_column][i]
        x["output"] = sample[summary_column][i]
        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset from the hub
    dataset = load_dataset(dataset_id, name=dataset_config)

    # process dataset
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=list(dataset["train"].features)
    )

    # save dataset to disk
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, "cnn_train.json"), "w") as write_f:
        json.dump(tokenized_dataset["train"]["text"], write_f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.save_path, "cnn_eval.json"), "w") as write_f:
        json.dump(tokenized_dataset["test"]["text"], write_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
