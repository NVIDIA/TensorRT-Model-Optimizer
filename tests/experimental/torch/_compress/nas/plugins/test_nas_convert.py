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

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from experimental.torch._compress.conftest import (
    create_and_save_small_llama_model,
    create_tokenizer,
    save_dummy_dataset,
    setup_puzzle_dir,
)
from puzzle_tools.hydra_utils import register_hydra_resolvers

import modelopt.torch.nas as mtn
from modelopt.torch._compress.nas.plugins.compress_nas_plugin import CompressModel
from modelopt.torch._compress.runtime import NativeDdpRuntime


@pytest.fixture
def project_root_path(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath)


#
# See tests/experimental/torch/_compress/test_compress.py for instructions on how to run this test
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
    # The inputs for the nas.convert() step.
    #
    puzzle_dir = tmp_path
    llama_checkpoint_path = puzzle_dir / "ckpts/llama"
    dataset_path = puzzle_dir / "dummy_dataset"
    hydra_config_dir = project_root_path / "tests/experimental/torch/_compress/resources/configs"
    hydra_config_name = "Llama-3_1-8B"

    runtime = NativeDdpRuntime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    )

    with runtime as runtime:
        if rank == 0:
            # Setup puzzle_dir and dataset
            setup_puzzle_dir(puzzle_dir)
            save_dummy_dataset(dataset_path)

            # Create a small Llama model
            tokenizer = create_tokenizer(project_root_path)
            create_and_save_small_llama_model(
                llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
            )
        runtime.wait_for_everyone()

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
                        "puzzle_dir": str(puzzle_dir),
                        "input_model_path": str(llama_checkpoint_path),
                        "hydra_config_dir": str(hydra_config_dir),
                        "hydra_config_name": hydra_config_name,
                        "dataset_path": str(dataset_path),
                    },
                )
            ],
        )

        #
        # Check assertions
        #
        if rank == 0:
            # assertions for the score_pruning_activations step
            rank = int(os.environ["RANK"])
            rank_filepath = (
                f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
            )
            assert (puzzle_dir / rank_filepath).is_file()

            # assertions for the pruning_ckpts step
            assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()

        runtime.wait_for_everyone()

    print("PYTEST SUMMARY: test_nas_convert() test has finished successfully")
