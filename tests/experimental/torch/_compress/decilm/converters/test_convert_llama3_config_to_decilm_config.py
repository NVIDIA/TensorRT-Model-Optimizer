import json
import os.path as osp
from pathlib import Path

import pytest
from gpu.torch._compress.test_compress import create_and_save_small_llama_model
from transformers import AutoTokenizer

from modelopt.torch._compress.decilm.converters.convert_llama3_to_decilm import (
    convert_llama3_to_decilm,
)


@pytest.fixture
def project_root_path(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath)


def test_convert_llama3_config_to_decilm_config(project_root_path: Path, tmp_path: Path):
    tokenizer_path = osp.join(project_root_path, "tests/gpu/torch/_compress/resources/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    llama_checkpoint_path = tmp_path / "llama_checkpoint"
    create_and_save_small_llama_model(
        llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
    )

    # Convert the Llama model to a DeciLM model
    decilm_checkpoint_path = tmp_path / "decilm_checkpoint"
    convert_llama3_to_decilm(
        input_dir=llama_checkpoint_path,
        output_dir=decilm_checkpoint_path,
    )

    # Assert that the converted config has the correct number of block_configs
    config_path = decilm_checkpoint_path / "config.json"
    assert config_path.exists(), f"Config file not found at {config_path}"

    with open(config_path) as f:
        decilm_config = json.load(f)

    # Verify block_configs exists and has the correct length
    assert "block_configs" in decilm_config, "block_configs not found in converted config"
    actual_num_block_configs = len(decilm_config["block_configs"])
    assert actual_num_block_configs == 2
