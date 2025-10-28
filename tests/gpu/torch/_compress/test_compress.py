import datetime
import os
import os.path as osp
import shutil
from pathlib import Path

import pytest
import torch
from logger import mprint
from puzzle_tools.hydra_utils import register_hydra_resolvers
from puzzle_tools.runtime import NativeDDP_Runtime
from scripts.convert_llama3_to_decilm import convert_llama3_to_decilm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

from modelopt.torch._compress import compress
from tests.integration.puzzle_tools.e2e_puzzletron_test.dummy_dataset import save_dummy_dataset


@pytest.fixture(scope="module", autouse=True)
def setup_test_module():
    register_hydra_resolvers()


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
# ../puzzletron/v1/scripts/torch_dist_runner.sh \
# pytest -s -v ./tests/gpu/torch/puzzletron/test_compress_model.py -o addopts=""
#
# TODO: Remove those instructions once this test runs automatically on CI


def test_compress(project_root_path):
    # The input to puzzletron.compress().
    os.environ["WANDB_DISABLED"] = "true"
    puzzle_dir = "/tmp/pytest-shared/test_compress_model"
    dataset_path = osp.join(puzzle_dir, "dummy_dataset")
    hydra_config_dir = osp.join(
        project_root_path,
        "tests/gpu/torch/_compress/resources/configs",
    )

    _runtime = NativeDDP_Runtime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    )

    with _runtime as runtime:
        #
        # Test setup
        #
        if runtime.global_rank == 0:
            # Setup puzzle_dir and dataset
            setup_puzzle_dir(puzzle_dir)
            save_dummy_dataset(dataset_path)

            #
            # Step 1: Create and save a teacher model to compress
            # This mimics the normal pipeline where we start with a Llama model
            #
            tokenizer_path = osp.join(
                project_root_path, "tests/gpu/torch/_compress/resources/tokenizer"
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Create a small Llama model (not DeciLM) to match the normal conversion pipeline
            hf_ckpt_teacher_dir = "ckpts/teacher"
            llama_checkpoint_path = osp.join(puzzle_dir, hf_ckpt_teacher_dir)
            create_and_save_small_llama_model(
                llama_checkpoint_path,
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer,
            )

            # Use the full conversion pipeline (matches normal usage)
            convert_llama3_to_decilm(
                input_dir=llama_checkpoint_path,
                output_dir=llama_checkpoint_path,
            )
        runtime.wait_for_everyone()

        # Compress the model using a one-click approach
        compress.compress(hydra_config_dir, "Llama-3_1-8B", puzzle_dir, dataset_path, runtime)

        #
        # Check assertions
        #
        if runtime.global_rank == 0:
            # assertions for the score_pruning_activations step 1
            rank = int(os.environ["RANK"])
            rank_filepath = (
                f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
            )
            assert os.path.isfile(osp.join(puzzle_dir, rank_filepath))

            # assertions for the pruning_ckpts step 2
            assert os.path.exists(osp.join(puzzle_dir, "ckpts/ffn_256_attn_no_op"))

            # assertions fo bypass distillation step 3
            # TODO: Add bypass distillation step
            # assert os.path.exists(osp.join(hydra_cfg.bypass.experiment_dir, "latest/config.json"))

            # assertions for the build_library_and_stats step 4
            assert os.path.isfile(osp.join(puzzle_dir, "replacement_library.json"))
            assert os.path.isfile(osp.join(puzzle_dir, "subblock_stats.json"))

            # assertions for the scoring step 5
            solution_0_filepath = osp.join(
                puzzle_dir,
                "single_sequence_replacement_solutions--validation/solution_0.json",
            )
            assert os.path.exists(solution_0_filepath)

            # assertions for the mip_and_realize_models step 6
            solution_0_ckpt_config_path = osp.join(
                puzzle_dir,
                "mip/puzzle_solutions/target_memory_780000MiB/solutions--checkpoints/solution_0/config.json",
            )
            assert os.path.exists(solution_0_ckpt_config_path)
            assert os.path.exists(
                osp.join(
                    puzzle_dir,
                    "mip/puzzle_solutions/target_memory_780000MiB/solutions.json",
                )
            )

        runtime.wait_for_everyone()

        mprint("PYTEST SUMMARY: test_compress_model() test has finished successfully")


def create_and_save_small_llama_model(
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


def setup_puzzle_dir(puzzle_dir: str):
    if Path(puzzle_dir).exists():
        shutil.rmtree(puzzle_dir)
        Path(puzzle_dir).mkdir(parents=True, exist_ok=True)
