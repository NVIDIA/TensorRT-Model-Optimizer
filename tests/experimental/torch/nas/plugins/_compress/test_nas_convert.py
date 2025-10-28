from pathlib import Path

import pytest
from gpu.torch._compress.test_compress import create_and_save_small_llama_model
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
    #
    # Step 1: Setup the puzzle_dir, dataset, hydra_config_dir, and input model
    # needed for the mnt.convert() step
    #
    puzzle_dir = tmp_path
    hydra_config_dir = project_root_path / "tests/gpu/torch/puzzletron/resources/configs"
    # dataset_path = puzzle_dir / "dummy_dataset"

    # Setup puzzle_dir and dataset
    # setup_puzzle_dir(puzzle_dir)
    # save_dummy_dataset(dataset_path)

    # Create a small Llama model (input to the mnt.convert() step)
    tokenizer_path = project_root_path / "tests/gpu/torch/_compress/resources/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    hf_ckpt_teacher_dir = "ckpts/teacher"
    llama_checkpoint_path = puzzle_dir / hf_ckpt_teacher_dir
    # TODO: the same as in tests/gpu/torch/_compress/test_compress.py (refactor it)
    create_and_save_small_llama_model(
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
                    "hydra_config_dir": str(hydra_config_dir),
                    "puzzle_dir": str(puzzle_dir),
                    "dataset_path": "",  # dataset_path,
                },
            )
        ],
    )

    #
    # Check assertions
    #
