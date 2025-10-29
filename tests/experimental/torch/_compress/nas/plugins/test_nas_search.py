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

#
# See tests/experimental/torch/_compress/test_compress.py for instructions on how to run this test
# TODO: Remove those instructions once this test runs automatically on CI
#
import datetime
from functools import partial
from pathlib import Path

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from experimental.torch._compress.test_utils import (
    create_and_save_small_llama_model,
    create_tokenizer,
    save_dummy_dataset,
    setup_puzzle_dir,
)
from puzzle_tools.hydra_utils import register_hydra_resolvers

import modelopt.torch.nas as mtn
from modelopt.torch._compress.nas.plugins.compress_nas_plugin import CompressModel
from modelopt.torch._compress.runtime import NativeDdpRuntime


def test_nas_search(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nas_search_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_nas_search_multiprocess_job(
    project_root_path: Path, tmp_path: Path, rank: int, size: int
):
    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    #
    # The inputs for the nas.convert()/nas.search() steps.
    #
    puzzle_dir = tmp_path
    # TODO: change it to "ckpts/llama" once the conversion script is fixed (internal NVidia modelopt bug: issues/17)
    llama_checkpoint_path = puzzle_dir / "ckpts/teacher"
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

        # Converted model is the same as the input model, but with the search space set up:
        # (HF model imported to DeciLM format, pruning scores pruned checkpoints and are saved)
        converted_model = mtn.convert(
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
        # Run the mnt.search() step
        #
        mtn.search(
            converted_model,
            constraints={},  # this is not used as the search space is defined in the hydra config
            dummy_input=None,  # Not used
            config={},  # this is not used as the search space is defined in the hydra config
        )

        #
        # Check assertions for mnt.search() step
        #
        if rank == 0:
            # assertions for the build_library_and_stats step
            assert (puzzle_dir / "replacement_library.json").is_file()
            assert (puzzle_dir / "subblock_stats.json").is_file()

            # assertions for the scoring step
            solution_0_filepath = (
                puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
            )

            assert solution_0_filepath.exists()

            # assertions for the mip_and_realize_models step
            solution_0_ckpt_config_path = (
                puzzle_dir
                / "mip/puzzle_solutions/target_memory_780000MiB/solutions--checkpoints/solution_0/config.json"
            )

            assert solution_0_ckpt_config_path.exists()
            assert (
                puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB/solutions.json"
            ).exists()

        runtime.wait_for_everyone()

    print("PYTEST SUMMARY: test_nas_search() test has finished successfully")
