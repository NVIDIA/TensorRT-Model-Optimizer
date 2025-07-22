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

"""Script to download the MagPie dataset from Hugging Face and convert it to JSONL format.

The script downloads the MagPie-Llama-3.1-Pro-300K-Filtered dataset and converts each
conversation to the format:
{"role": "user", "content": <user_message>}
{"role": "assistant", "content": <assistant_response>}

Usage:
    python download_magpie_dataset.py --output magpie_dataset.jsonl
"""

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def convert_conversation_to_jsonl(conversation: list[dict[str, str]]):
    """Convert a conversation to the required JSONL format.

    Args:
        conversation: List of conversation turns with 'from' and 'value' keys.

    Returns:
        List of dictionaries with 'role' and 'content' keys.
    """
    jsonl_entries = []

    for turn in conversation:
        role = turn.get("from", "").lower()
        content = turn.get("value", "").strip()

        if role == "human":
            jsonl_entries.append({"role": "user", "content": content})
        elif role == "gpt":
            jsonl_entries.append({"role": "assistant", "content": content})
        # Skip other roles or malformed entries

    return jsonl_entries


def process_dataset_to_jsonl(dataset, output_path: str):
    """Process the dataset and write to JSONL file.

    Args:
        dataset: List of dataset samples.
        output_path: Path to output JSONL file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(dataset, desc="Processing conversations"):
            conversations = sample.get("conversations", [])
            if not conversations:
                continue

            # Convert conversation to JSONL format
            jsonl_entries = convert_conversation_to_jsonl(conversations)

            # Write each entry to the file
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="magpie.jsonl",
        help="Output JSONL file path (default: magpie.jsonl)",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if output_dir := os.path.dirname(args.output):
        os.makedirs(output_dir, exist_ok=True)

    print("Loading MagPie dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split="train")

    # Process and save to JSONL
    process_dataset_to_jsonl(dataset, args.output)

    print(f"Dataset conversion complete! Output saved to: {args.output}")
