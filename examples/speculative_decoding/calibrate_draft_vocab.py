# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import os

import torch
from transformers import AutoTokenizer

from modelopt.torch.speculative.utils import calibrate_frequent_vocab


def main():
    parser = argparse.ArgumentParser(description="Calibrate draft vocab and save to .pt file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for tokenizer")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (jsonl)")
    parser.add_argument(
        "--eagle_config",
        type=str,
        required=True,
        default="eagle_config.json",
        help="Path to eagle_config.json",
    )
    parser.add_argument(
        "--calibrate_size",
        type=int,
        default=None,
        help="Number of samples to use for calibration. If None, use all dataset.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="draft_vocab_cache", help="Path to save .pt file"
    )
    args = parser.parse_args()

    with open(args.eagle_config) as f:
        eagle_config = json.load(f)
    if "draft_vocab_size" not in eagle_config:
        print("No draft vocab size specified in eagle_config.json, no need to calibrate for d2t.")
        return

    print("Calibrating vocab...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.data) as f:
        conversations = [json.loads(line)["conversations"] for line in f]
        if args.calibrate_size:
            conversations = conversations[: args.calibrate_size]
        conversations = [item for sublist in conversations for item in sublist]

    d2t = calibrate_frequent_vocab(tokenizer, conversations, eagle_config["draft_vocab_size"])
    model_name = os.path.basename(os.path.normpath(args.model))
    vocab_path = os.path.join(args.save_dir, model_name, "d2t.pt")
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    torch.save(d2t, vocab_path)
    print(f"Saved calibrated vocab to {vocab_path}")


if __name__ == "__main__":
    main()
