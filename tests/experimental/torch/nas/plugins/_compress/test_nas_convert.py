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

import os
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from experimental.torch._compress.test_compress import (
    _create_and_save_small_llama_model,
    _save_dummy_dataset,
    _setup_puzzle_dir,
)
from puzzle_tools.hydra_utils import register_hydra_resolvers
from transformers import AutoTokenizer

import modelopt.torch.nas as mtn
from modelopt.torch.nas.plugins._compress.compress_nas_plugin import CompressModel


@pytest.fixture
def project_root_path(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath)


#
# See tests/gpu/torch/_compress/test_compress.py for instructions on how to run this test
# TODO: Remove those instructions once this test runs automatically on CI
#
def test_nas_convert(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nas_convert_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_nas_convert_multiprocess_job(
    project_root_path: Path, tmp_path: Path, rank: int, size: int
):
    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    #
    # Step 1: Setup the puzzle_dir, dataset, hydra_config_dir, and input model
    # needed for the mnt.convert() step
    #
    puzzle_dir = tmp_path
    dataset_path = puzzle_dir / "dummy_dataset"
    hydra_config_dir = project_root_path / "tests/experimental/torch/_compress/resources/configs"
    hydra_config_name = "Llama-3_1-8B"

    # Setup puzzle_dir and dataset
    _setup_puzzle_dir(puzzle_dir)
    _save_dummy_dataset(dataset_path)

    # Create a small Llama model (input to the mnt.convert() step)
    tokenizer_path = project_root_path / "tests/experimental/torch/_compress/resources/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    llama_checkpoint_path = puzzle_dir / "ckpts/llama"
    _create_and_save_small_llama_model(
        llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
    )

    #
    # Run the mnt.convert() step
    #
    input_model = CompressModel()
    mtn.convert(
        input_model,
        mode=[
            (
                "compress",
                {
                    "input_model_path": str(llama_checkpoint_path),
                    "hydra_config_dir": str(hydra_config_dir),
                    "hydra_config_name": hydra_config_name,
                    "puzzle_dir": str(puzzle_dir),
                    "dataset_path": str(dataset_path),
                },
            )
        ],
    )

    #
    # Check assertions
    #

    # assertions for the score_pruning_activations step 1
    rank = int(os.environ["RANK"])
    rank_filepath = f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
    assert (puzzle_dir / rank_filepath).is_file()

    # assertions for the pruning_ckpts step 2
    assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()
