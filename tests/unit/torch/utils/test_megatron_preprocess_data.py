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

import json
import os
from pathlib import Path

import pytest
from _test_utils.import_helper import skip_if_no_megatron

# Skip the test if megatron is not available
skip_if_no_megatron()
datasets = pytest.importorskip("datasets")
_ = pytest.importorskip("transformers")

from modelopt.torch.utils.plugins.megatron_preprocess_data import megatron_preprocess_data


def download_and_prepare_minipile_dataset(output_dir: Path) -> Path:
    """Download the nanotron/minipile_100_samples dataset and convert to JSONL format.

    Args:
        output_dir: Directory to save the JSONL file

    Returns:
        Path to the created JSONL file
    """
    # Download the dataset
    dataset = datasets.load_dataset("nanotron/minipile_100_samples", split="train")

    # Convert to JSONL format
    jsonl_file = output_dir / "minipile_100_samples.jsonl"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in dataset:
            # Extract the text field and write as JSONL
            json_obj = {"text": item["text"]}
            f.write(json.dumps(json_obj) + "\n")

    return jsonl_file


def test_megatron_preprocess_data_with_minipile_dataset(tmp_path):
    """Test megatron_preprocess_data function with nanotron/minipile_100_samples dataset.

    This test:
    1. Downloads the HuggingFace dataset "nanotron/minipile_100_samples"
    2. Converts it to JSONL format
    3. Passes it to megatron_preprocess_data
    4. Verifies that output files are created
    """
    # Download and prepare the dataset
    input_jsonl = download_and_prepare_minipile_dataset(tmp_path)

    # Verify the input file was created and has content
    assert input_jsonl.exists(), "Input JSONL file should exist"
    assert input_jsonl.stat().st_size > 0, "Input JSONL file should not be empty"

    # Set up output paths
    output_prefix = tmp_path / "test_output"

    # Test the megatron_preprocess_data function
    megatron_preprocess_data(
        input_path=str(input_jsonl),
        output_prefix=str(output_prefix),
        tokenizer_name_or_path="gpt2",  # Use a small, common tokenizer
        json_keys=["text"],
        append_eod=False,
        workers=1,
        partitions=1,
        log_interval=10,
    )

    # Verify that output files were created
    expected_bin_file = f"{output_prefix}_text_document.bin"
    expected_idx_file = f"{output_prefix}_text_document.idx"

    assert os.path.exists(expected_bin_file), (
        f"Expected binary file {expected_bin_file} should exist"
    )
    assert os.path.exists(expected_idx_file), (
        f"Expected index file {expected_idx_file} should exist"
    )

    # Verify the files have content (non-zero size)
    assert os.path.getsize(expected_bin_file) > 0, "Binary file should not be empty"
    assert os.path.getsize(expected_idx_file) > 0, "Index file should not be empty"

    # Optional: Verify the input JSONL file structure
    with open(input_jsonl, encoding="utf-8") as f:
        first_line = f.readline().strip()
        first_item = json.loads(first_line)
        assert "text" in first_item, "Each JSONL item should have a 'text' field"
        assert isinstance(first_item["text"], str), "Text field should be a string"


def test_megatron_preprocess_data_with_custom_parameters(tmp_path):
    """Test megatron_preprocess_data with different parameters."""
    # Create a minimal test dataset
    input_jsonl = tmp_path / "test_data.jsonl"

    # Create some test data
    test_data = [
        {"text": "This is a test sentence for preprocessing."},
        {"text": "Another test sentence with different content."},
        {"text": "A third sentence to make sure the function works correctly."},
    ]

    with open(input_jsonl, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    output_prefix = tmp_path / "custom_test_output"

    # Test with different parameters
    megatron_preprocess_data(
        input_path=str(input_jsonl),
        output_prefix=str(output_prefix),
        tokenizer_name_or_path="gpt2",
        json_keys=["text"],
        append_eod=True,  # Test with end-of-document token
        max_document_length=50,  # Test with character limit
        workers=1,
        partitions=1,
        log_interval=1,
    )

    # Verify output files exist
    expected_bin_file = f"{output_prefix}_text_document.bin"
    expected_idx_file = f"{output_prefix}_text_document.idx"

    assert os.path.exists(expected_bin_file), (
        f"Expected binary file {expected_bin_file} should exist"
    )
    assert os.path.exists(expected_idx_file), (
        f"Expected index file {expected_idx_file} should exist"
    )
    assert os.path.getsize(expected_bin_file) > 0, "Binary file should not be empty"
    assert os.path.getsize(expected_idx_file) > 0, "Index file should not be empty"
