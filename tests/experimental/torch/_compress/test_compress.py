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
import shutil
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from datasets import Dataset, DatasetDict
from puzzle_tools.hydra_utils import register_hydra_resolvers
from scripts.convert_llama3_to_decilm import convert_llama3_to_decilm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

from modelopt.torch._compress import compress
from modelopt.torch._compress.runtime import NativeDdpRuntime


@pytest.fixture
def project_root_path(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath)


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

    # Set environment variables expected by NativeDDP_Runtime
    os.environ["WANDB_DISABLED"] = "true"

    puzzle_dir = tmp_path
    dataset_path = puzzle_dir / "dummy_dataset"
    hydra_config_dir = project_root_path / "tests/experimental/torch/_compress/resources/configs"

    _runtime = NativeDdpRuntime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    )

    with _runtime as runtime:
        #
        # Test setup
        #
        if rank == 0:
            # Setup puzzle_dir and dataset
            _setup_puzzle_dir(puzzle_dir)
            _save_dummy_dataset(dataset_path)

            #
            # Step 1: Create and save a teacher model to compress
            # This mimics the normal pipeline where we start with a Llama model
            #
            tokenizer_path = (
                project_root_path / "tests/experimental/torch/_compress/resources/tokenizer"
            )

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Create a small Llama model (not DeciLM) to match the normal conversion pipeline
            hf_ckpt_teacher_dir = "ckpts/teacher"
            llama_checkpoint_path = puzzle_dir / hf_ckpt_teacher_dir
            _create_and_save_small_llama_model(
                llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
            )

            # Use the full conversion pipeline (matches normal usage)
            convert_llama3_to_decilm(
                input_dir=llama_checkpoint_path,
                output_dir=llama_checkpoint_path,
            )
        runtime.wait_for_everyone()

        # Compress the model using a one-click approach
        compress.compress(
            str(hydra_config_dir), "Llama-3_1-8B", str(puzzle_dir), str(dataset_path), runtime
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


def _create_and_save_small_llama_model(
    output_path: str, vocab_size: int, tokenizer: PreTrainedTokenizerBase
):
    """
    Create and save a small Llama model for testing the conversion pipeline.
    This mimics having a real Llama checkpoint that needs to be converted.
    """
    os.makedirs(output_path, exist_ok=True)

    # Create a minimal Llama config (small for testing)
    # Note: intermediate_size must be divisible by 256 per DeciLM config requirements
    # Note: hidden_size must give head_dim >= 8 for Flash Attention 2 compatibility
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,  # 32 heads times 8 head_dim = 256 (matches bypass config expectations)
        intermediate_size=512,  # Must be divisible by 256
        num_hidden_layers=2,
        num_attention_heads=32,  # Matches original test
        num_key_value_heads=8,  # GQA: 32÷4=8 (matches original n_heads_in_group=4)
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
    )

    # Create and save the Llama model
    model = LlamaForCausalLM(llama_config)
    model.to(dtype=torch.bfloat16).save_pretrained(output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save config
    llama_config.save_pretrained(output_path)


def _setup_puzzle_dir(puzzle_dir: str):
    if Path(puzzle_dir).exists():
        shutil.rmtree(puzzle_dir)
        Path(puzzle_dir).mkdir(parents=True, exist_ok=True)


def _save_dummy_dataset(dataset_path: str):
    # dummy sample
    sample = [
        {"role": "user", "content": "please cite Lorem Ipsum?"},
        {
            "role": "assistant",
            "content": (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed in blandit ante. "
                "Sed tempus erat urna, ac elementum nisl facilisis quis. Aliquam consectetur mollis massa, "
                "in elementum sem venenatis posuere. Fusce lorem arcu, egestas vel massa sollicitudin, "
                "dictum mollis purus. Proin in ullamcorper elit. Nam tellus nisi, volutpat a mattis vel, "
                "pretium in purus. Nunc at lectus facilisis risus scelerisque rhoncus eu nec ex. "
                "Maecenas semper, tellus non placerat vulputate, urna felis facilisis diam, "
                "sit amet vestibulum erat sapien nec libero. Praesent non massa velit. Donec faucibus mi eros. "
                "Nam turpis nulla, congue sit amet mi at, porttitor scelerisque elit. Nunc id sodales lorem, "
                "nec tincidunt leo. Quisque a neque nec ligula porttitor auctor. "
                "Nunc accumsan nunc ac tellus congue vehicula. Praesent tellus eros, luctus non gravida dapibus, "
                "faucibus eu ex. Quisque bibendum leo pharetra, tristique est vitae, hendrerit nunc. "
                "Duis nec congue dolor. Donec commodo ipsum non efficitur volutpat. "
                "Nulla risus nulla, efficitur et urna at, imperdiet sodales lorem. "
                "Suspendisse erat est, sollicitudin at nisl tincidunt, vehicula hendrerit lectus. "
                "Nam quis nisi ullamcorper, rhoncus massa vel, tempus purus. "
                "Duis pulvinar eros vel nulla pellentesque, at dapibus justo laoreet. "
                "Praesent tortor orci, vulputate fermentum dapibus nec, feugiat vitae tortor. "
                "Donec mollis convallis massa quis iaculis."
            ),
        },
    ]

    # Prepare train and val splits with sample repeated, 2500 samples are for
    # 128 samples with block-size 8192 and LLama3 tokenizer
    data = [{"conversation": sample}] * 2500

    # For train-val splits
    data_dict = DatasetDict({"train": Dataset.from_list(data), "valid": Dataset.from_list(data)})
    data_dict.save_to_disk(dataset_path)
