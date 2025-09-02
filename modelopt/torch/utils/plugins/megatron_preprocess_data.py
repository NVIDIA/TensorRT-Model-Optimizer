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

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data to tokenize for pretraining.

Usage:

```python
from modelopt.torch.utils.plugins import megatron_preprocess_data

megatron_preprocess_data(
    input_path="path/to/input/data",
    output_prefix="path/to/output/dir",
    tokenizer_name_or_path="hf_model_name",
    json_keys=["name of json key(s) to tokenize"],
)
```
"""

import glob
import gzip
import json
import math
import multiprocessing
import os
import sys
import time

from megatron.core.datasets import indexed_dataset
from transformers import AutoTokenizer


class _Encoder:
    tokenizer: AutoTokenizer = None

    def __init__(self, tokenizer_name_or_path, json_keys, append_eod, max_document_length):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.json_keys = json_keys
        self.append_eod = append_eod
        self.max_document_length = max_document_length

    def initializer(self):
        # Use Encoder class as a container for global data
        _Encoder.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.json_keys:
            text = data[key]

            # Truncate text by character length if specified
            if self.max_document_length is not None and len(text) > self.max_document_length:
                original_length = len(text)
                text = text[: self.max_document_length]
                print(
                    f"Warning: Document truncated from {original_length} to {self.max_document_length} characters"
                )

            # Tokenize the entire text as one document
            encoded = _Encoder.tokenizer.encode(text)

            if len(encoded) > 0 and self.append_eod:
                encoded.append(_Encoder.tokenizer.eos_token_id)

            ids[key] = encoded
            lens[key] = [len(encoded)] if len(encoded) > 0 else []
        return ids, lens, len(json_line)


class _Partition:
    def __init__(self, vocab_size, json_keys, log_interval, workers):
        self.vocab_size = vocab_size
        self.json_keys = json_keys
        self.log_interval = log_interval
        self.workers = workers

    def _print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(
                f"Processed {count} documents",
                f"({count / elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr,
            )

    def process_json_file(self, file_name, encoder):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, encoding="utf-8")

        startup_start = time.time()
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"

        for key in self.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(self.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc:
                builders[key].add_document(doc[key], sentence_lens[key])
            self._print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        for key in builders:
            builders[key].finalize(output_idx_files[key])


def _get_file_name(input_path, output_prefix, file_id):
    file_name, extension = os.path.splitext(input_path)
    input_file_name = file_name + "_" + str(file_id) + extension
    output_prefix_name = output_prefix + "_" + str(file_id)
    file_names = {
        "partition": input_file_name,
        "output_prefix": output_prefix_name,
    }
    return file_names


def _check_files_exist(in_ss_out_names, key, num_partitions):
    return all(os.path.exists(in_ss_out_names[i][key]) for i in range(num_partitions))


def megatron_preprocess_data(
    input_path: str,
    output_prefix: str,
    tokenizer_name_or_path: str,
    json_keys: list[str] = ["text"],
    append_eod: bool = False,
    max_document_length: int | None = None,
    workers: int = 1,
    partitions: int = 1,
    log_interval: int = 1000,
    keep_sequential_samples: bool = False,
):
    """Process large data for pretraining.

    Args:
        input_path (str): Path to directory containing input JSON files
        output_prefix (str): Path to binary output file without suffix
        tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer to use
        json_keys (list, optional): List of keys to extract from json. Defaults to ["text"]
        append_eod (bool, optional): Append an <eod> token to the end of a document. Defaults to False
        max_document_length (int, optional): Maximum document length in characters. Defaults to None
        workers (int, optional): Number of worker processes to launch. Defaults to 1
        partitions (int, optional): Number of file partitions. Defaults to 1
        log_interval (int, optional): Interval between progress updates. Defaults to 1000
        keep_sequential_samples (bool, optional): Ensure ordering of samples in .jsonl files is preserved when using
            partitions>1. Defaults to False
    """
    in_ss_out_names = []
    if partitions == 1:
        file_names = {
            "partition": input_path,
            "output_prefix": output_prefix,
        }
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(input_path)

        # Count total number of lines across .jsonl files
        if keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename) as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += fc + 1
            partition_size = math.ceil(total_sample_count / partitions)

        # create .jsonl partition files
        for idx in range(partitions):
            in_ss_out_name = _get_file_name(input_path, output_prefix, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if partitions were already created
        partitions_present = _check_files_exist(in_ss_out_names, "partition", partitions)

        if not partitions_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(partitions):
                partitioned_input_file = open(in_ss_out_names[idx]["partition"], "w")
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if keep_sequential_samples:
                line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, "rt")
                else:
                    fin = open(in_file_name, encoding="utf-8")

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % partitions

                fin.close()

            for idx in range(partitions):
                partitioned_input_files[idx].close()

    assert workers % partitions == 0
    vocab_size = AutoTokenizer.from_pretrained(tokenizer_name_or_path).vocab_size

    encoder = _Encoder(tokenizer_name_or_path, json_keys, append_eod, max_document_length)
    partition = _Partition(vocab_size, json_keys, log_interval, workers // partitions)

    # encode partition files in parallel
    processes = []
    for name in in_ss_out_names:
        p = multiprocessing.Process(
            target=partition.process_json_file,
            args=((name["partition"], name["output_prefix"]), encoder),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if partitions == 1:
        return

    # merge bin/idx partitions
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    level = "document"

    for key in json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(vocab_size),
        )

        for name in in_ss_out_names:
            partition_output_prefix = name["output_prefix"]
            full_partition_output_prefix = "{}_{}_{}".format(partition_output_prefix, key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])
