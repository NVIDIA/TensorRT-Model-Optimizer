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

import os
import subprocess
import time
from pathlib import Path

from _test_utils.torch_dist.dist_utils import get_free_port

MODELOPT_ROOT = Path(__file__).parent.parent.parent.parent


def _extend_cmd_parts(cmd_parts: list[str], **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            cmd_parts.extend([f"--{key}", str(value)])
    if kwargs.get("trust_remote_code", False):
        cmd_parts.append("--trust_remote_code")
    return cmd_parts


def run_example_command(cmd_parts: list[str], example_path: str):
    print(f"[{example_path}] Running command: {cmd_parts}")
    subprocess.run(cmd_parts, cwd=MODELOPT_ROOT / "examples" / example_path, check=True)


def run_command_in_background(cmd_parts, example_path, stdout=None, stderr=None, text=True):
    print(f"Running command in background: {' '.join(str(part) for part in cmd_parts)}")
    process = subprocess.Popen(
        cmd_parts,
        cwd=MODELOPT_ROOT / "examples" / example_path,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )
    return process


def run_llm_autodeploy_command(
    model: str, quant: str, effective_bits: float, output_dir: str, **kwargs
):
    # Create temporary directory for saving the quantized checkpoint
    port = get_free_port()
    quantized_ckpt_dir = os.path.join(output_dir, "quantized_model")
    kwargs.update(
        {
            "hf_ckpt": model,
            "quant": quant,
            "effective_bits": effective_bits,
            "save_quantized_ckpt": quantized_ckpt_dir,
            "port": port,
        }
    )

    server_handler = None
    try:
        # Quantize and deploy the model to the background
        cmd_parts = _extend_cmd_parts(["scripts/run_auto_quant_and_deploy.sh"], **kwargs)
        # Pass None to stdout and stderr to see the output in the console
        server_handler = run_command_in_background(
            cmd_parts, "llm_autodeploy", stdout=None, stderr=None
        )

        # Wait for the server to start. We might need to buil
        time.sleep(100)

        # Test the deployment
        run_example_command(
            ["python", "api_client.py", "--prompt", "What is AI?", "--port", str(port)],
            "llm_autodeploy",
        )
    finally:
        if server_handler:
            server_handler.terminate()


def run_llm_ptq_command(*, model: str, quant: str, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "build")
    kwargs.setdefault("calib", 16)

    cmd_parts = _extend_cmd_parts(["scripts/huggingface_example.sh", "--no-verbose"], **kwargs)
    run_example_command(cmd_parts, "llm_ptq")


def run_vlm_ptq_command(*, model: str, type: str, quant: str, **kwargs):
    kwargs.update({"model": model, "type": type, "quant": quant})
    kwargs.setdefault("tasks", "build")
    kwargs.setdefault("calib", 16)

    cmd_parts = _extend_cmd_parts(["scripts/huggingface_example.sh"], **kwargs)
    run_example_command(cmd_parts, "vlm_ptq")


def run_diffusers_cmd(cmd_parts: list[str]):
    run_example_command(cmd_parts, "diffusers/quantization")


def run_llm_sparsity_command(
    *, model: str, output_dir: str, sparsity_fmt: str = "sparsegpt", **kwargs
):
    kwargs.update(
        {"model_name_or_path": model, "sparsity_fmt": sparsity_fmt, "output_dir": output_dir}
    )
    kwargs.setdefault("calib_size", 16)
    kwargs.setdefault("device", "cuda")
    kwargs.setdefault("dtype", "fp16")
    kwargs.setdefault("model_max_length", 1024)

    cmd_parts = _extend_cmd_parts(["python", "hf_pts.py"], **kwargs)
    run_example_command(cmd_parts, "llm_sparsity")


def run_llm_sparsity_ft_command(
    *, model: str, restore_path: str, output_dir: str, data_path: str, **kwargs
):
    kwargs.update(
        {
            "model": model,
            "restore_path": restore_path,
            "output_dir": output_dir,
            "data_path": data_path,
        }
    )
    kwargs.setdefault("num_epochs", 0.01)
    kwargs.setdefault("max_length", 128)
    kwargs.setdefault("train_bs", 1)
    kwargs.setdefault("eval_bs", 1)

    cmd_parts = _extend_cmd_parts(["bash", "launch_finetune.sh"], **kwargs)
    run_example_command(cmd_parts, "llm_sparsity")
