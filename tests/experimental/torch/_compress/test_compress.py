# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import os
from functools import partial
from pathlib import Path

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from experimental.torch._compress.compress_test_utils import (
    create_and_save_small_llama_model,
    create_tokenizer,
    save_dummy_dataset,
    setup_puzzle_dir,
)
from puzzle_tools.hydra_utils import register_hydra_resolvers

from modelopt.torch._compress import compress
from modelopt.torch._compress.decilm.converters.convert_llama3_to_decilm import (
    convert_llama3_to_decilm,
)
from modelopt.torch._compress.runtime import NativeDdpRuntime

# The e2e test to compress a model based on Local Neural Architecture Search (Mixed Integer Programing NAS search)
# using a one-click command.
#
# Note: Bypass is disabled now in the test.

# How to run this test (currently only supported internally at Nvidia).
#
# Have both modelopt and puzzle source code in the same directory:
# /workspace/modelopt
# /workspace/puzzletron
#
# submit_job --partition interactive --time 0 \
# --image gitlab-master.nvidia.com/deci/puzzletron:trtllm_main \
# --workdir $MODELOPT SRC DIRECTORY --interactive --gpu 1
#
# pip install mip
# pip install lru-dict
#
# export PYTHONPATH=$PYTHONPATH:/workspace/puzzletron/v1
#
# pytest -s -v ./tests/experimental/torch/_compress/test_compress.py::test_compress -o addopts=""


def test_compress(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_compress_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_compress_multiprocess_job(project_root_path: Path, tmp_path: Path, rank: int, size: int):
    register_hydra_resolvers()

    #
    # The inputs for the compress() algorihm.
    #
    puzzle_dir = tmp_path
    dataset_path = puzzle_dir / "dummy_dataset"
    hydra_config_dir = project_root_path / "tests/experimental/torch/_compress/resources/configs"
    hydra_config_name = "Llama-3_1-8B"

    with NativeDdpRuntime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    ) as runtime:
        #
        # Test setup
        #
        if rank == 0:
            # Setup puzzle_dir and dataset
            setup_puzzle_dir(puzzle_dir)
            save_dummy_dataset(dataset_path)

            #
            # Step 1: Create and save a teacher model to compress
            # This mimics the normal pipeline where we start with a Llama model
            #

            # Create a small Llama model (not DeciLM) to match the normal conversion pipeline
            tokenizer = create_tokenizer(project_root_path)
            # TODO: change it to "ckpts/llama" once the conversion script is fixed
            # Currently, the build replacement library step will fail with such a path.
            llama_checkpoint_path = puzzle_dir / "input_model/llama"
            create_and_save_small_llama_model(
                llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
            )

            # Use the full conversion pipeline (matches normal usage)
            convert_llama3_to_decilm(
                input_dir=llama_checkpoint_path,
                output_dir=puzzle_dir / "ckpts/teacher",
            )
        runtime.wait_for_everyone()

        # Compress the model using a one-click approach
        compress.compress(
            str(hydra_config_dir), hydra_config_name, str(puzzle_dir), str(dataset_path), runtime
        )

        #
        # Check assertions
        #
        if rank == 0:
            # assertions for the score_pruning_activations step 1
            rank = int(os.environ["RANK"])
            rank_filepath = (
                f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
            )
            assert (puzzle_dir / rank_filepath).is_file()

            # assertions for the pruning_ckpts step 2
            assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()

            # assertions for the build_library_and_stats step 4

            assert (puzzle_dir / "replacement_library.json").is_file()
            assert (puzzle_dir / "subblock_stats.json").is_file()

            # assertions for the scoring step 5
            solution_0_filepath = (
                puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
            )

            assert solution_0_filepath.exists()

            # assertions for the mip_and_realize_models step 6
            solution_0_ckpt_config_path = (
                puzzle_dir
                / "mip/puzzle_solutions/target_memory_780000MiB/solutions--checkpoints/solution_0/config.json"
            )

            assert solution_0_ckpt_config_path.exists()
            assert (
                puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB/solutions.json"
            ).exists()

        runtime.wait_for_everyone()

    print("PYTEST SUMMARY: test_compress_model() test has finished successfully")
