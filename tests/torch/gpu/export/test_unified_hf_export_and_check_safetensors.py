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

import subprocess
from pathlib import Path

import pytest
from _test_utils.torch_export.export_utils import get_tiny_llama_and_tokenizer
from safetensors import safe_open


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory):
    # Build a tiny LLaMA model on the fly and store in a temp directory
    model_dir = tmp_path_factory.mktemp("tiny_llama_model")
    tiny_llama, tokenizer = get_tiny_llama_and_tokenizer()
    tokenizer.save_pretrained(model_dir)
    tiny_llama.save_pretrained(model_dir)
    return model_dir


# Here we map each qformat -> the suffix we expect in the generated safetensors directory
@pytest.mark.parametrize(
    "qformat,expected_suffix",
    [
        ("fp8", "tiny_llama-fp8"),
        ("nvfp4", "tiny_llama-nvfp4"),
        ("nvfp4_awq", "tiny_llama-nvfp4-awq"),
        ("int4_awq", "tiny_llama-int4-awq"),
        ("w4a8_awq", "tiny_llama-w4a8-awq"),
    ],
)
def test_unified_hf_export_and_check_safetensors(
    tmp_path, tiny_llama_dir, qformat, expected_suffix
):
    """
    1) Generates a .safetensors file by running hf_ptq.py with each --qformat.
    2) Checks the generated directory for the expected .safetensors file:
       model.safetensors
    3) Optionally, loads the file via safe_open to verify presence of data.

    :param tmp_path: pytest fixture for a temporary directory.
    :param qformat: the quantization format to test (e.g., 'fp8', 'nvfp4', etc.).
    :param expected_suffix: the directory suffix where the .safetensors is expected.
    """
    current_file_dir = Path(__file__).parent
    hf_ptq_script_path = (current_file_dir / "../../../../examples/llm_ptq/hf_ptq.py").resolve()

    # Create an output directory in tmp_path
    # We'll replicate the naming convention, e.g. "tiny_llama-fp8"
    output_dir = tmp_path / expected_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Command to generate .safetensors
    cmd = [
        "python",
        str(hf_ptq_script_path),
        "--pyt_ckpt_path",
        str(tiny_llama_dir),
        "--qformat",
        qformat,
        "--export_fmt",
        "hf",
        "--export_path",
        str(output_dir),
        "--trust_remote_code",
    ]

    # Run the command
    subprocess.run(cmd, check=True)

    # Now we expect a file named model.safetensors in output_dir
    generated_file = output_dir / "model.safetensors"
    assert generated_file.exists(), (
        f"Expected .safetensors file not found for qformat={qformat}: {generated_file}"
    )

    # Load the safetensors to do further checks
    with safe_open(generated_file, framework="pt") as f:
        tensor_names = list(f.keys())
        assert len(tensor_names) > 0, f"No tensors found in {generated_file} for qformat={qformat}!"
        for name in tensor_names:
            tensor = f.get_tensor(name)
            # Basic sanity checks
            assert tensor.shape is not None, f"Tensor '{name}' shape is None!"
            assert tensor.dtype is not None, f"Tensor '{name}' dtype is None!"

    # TODO: Load a pre-dumped log to compare textually or use pre-defined dict for sanity checks
