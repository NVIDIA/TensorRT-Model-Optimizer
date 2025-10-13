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


def run_example_command(
    cmd_parts: list[str], example_path: str, env_vars: dict = {}, setup_free_port: bool = False
):
    example_dir = MODELOPT_ROOT / "examples" / example_path
    example_abspath = os.path.abspath(example_dir)

    cmd_str = " ".join(str(part) for part in cmd_parts)
    print(f"Running command: cd {example_abspath} && {cmd_str}")

    if env_vars is not None and len(env_vars) > 0:
        env = env_vars
    else:
        env = os.environ.copy()

    if setup_free_port:
        free_port = get_free_port()
        env["MASTER_PORT"] = str(free_port)

    subprocess.run(cmd_parts, cwd=example_dir, env=env, check=True)


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

        # Wait for the server to start. We might need to build
        time.sleep(100)

        # Test the deployment
        run_example_command(
            ["python", "api_client.py", "--prompt", "What is AI?", "--port", str(port)],
            "llm_autodeploy",
        )
    finally:
        if server_handler:
            server_handler.terminate()


def run_onnx_quantization_trtexec_command(
    *, onnx_path: str, static_plugins: str | None = None, env_vars: dict | None = None, **kwargs
):
    """Run TensorRT execution command for ONNX model.

    Args:
        onnx_path: Path to the input ONNX model
        static_plugins: Path to static plugins
        env_vars: Environment variables to set
        **kwargs: Additional arguments to pass to the command
    """
    base_cmd = ["trtexec", "--stronglyTyped", f"--onnx={onnx_path}"]

    if static_plugins:
        kwargs.update(
            {
                "staticPlugins": static_plugins,
            }
        )

    cmd_parts = _extend_cmd_parts(base_cmd, **kwargs)
    run_example_command(cmd_parts, "onnx_ptq", env_vars=env_vars)


def run_onnx_autocast_cli_command(
    *,
    onnx_path: str,
    output_path: str,
    low_precision_type: str = "fp16",
    keep_io_types: bool = True,
    providers: list[str] = ["trt", "cuda", "cpu"],
    trt_plugins: str | None = None,
    env_vars: dict | None = None,
    **kwargs,
):
    """Run ONNX autocast CLI command for model precision conversion.
    Args:
        onnx_path: Path to the input ONNX model
        output_path: Path to save the converted ONNX model
        low_precision_type: Target precision type (fp16, bf16, etc.)
        keep_io_types: Whether to preserve input/output data types
        providers: List of execution providers to use
        trt_plugins: Path to TensorRT plugins
        env_vars: Environment variables to set
        **kwargs: Additional arguments to pass to the command
    """
    # Update kwargs with required parameters
    kwargs.update(
        {
            "onnx_path": onnx_path,  # Path to input ONNX model
            "output_path": output_path,  # Path to save converted model
            "low_precision_type": low_precision_type,  # Target precision (fp16, bf16)
            "providers": providers,  # Execution providers list
        }
    )

    # Add flag to preserve input/output data types if requested
    if keep_io_types:
        kwargs.update({"keep_io_types": None})

    # Add TensorRT plugins path if specified
    if trt_plugins:
        kwargs.update({"trt_plugins": trt_plugins})

    cmd_parts = _extend_cmd_parts(["python", "-m", "modelopt.onnx.autocast"], **kwargs)
    run_example_command(cmd_parts, "onnx_ptq", env_vars=env_vars)


def run_torch_timm_onnx_command(
    *,
    quantize_mode: str,
    onnx_save_path: str,
    timm_model_name: str = "vit_base_patch16_224",
    calib_size: int = 512,
    env_vars: dict | None = None,
    **kwargs,
):
    """Run torch to ONNX conversion command with quantization.
    args:
        quantize_mode: quantization mode to use
        onnx_save_path: path to save the onnx model
        timm_model_name: name of the timm model to use
        calib_size: size of calibration dataset
        env_vars: environment variables to set
        **kwargs: additional arguments to pass to the command
    """
    kwargs.update(
        {
            "quantize_mode": quantize_mode,
            "timm_model_name": timm_model_name,
            "onnx_save_path": onnx_save_path,
            "calibration_data_size": calib_size,
        }
    )

    cmd_parts = _extend_cmd_parts(["python", "torch_quant_to_onnx.py"], **kwargs)
    run_example_command(cmd_parts, "onnx_ptq", env_vars=env_vars)


def run_llm_export_onnx_command(
    *,
    torch_dir: str,
    dtype: str,
    lm_head: str,
    output_dir: str,
    save_original: bool = False,
    config_path: str | None = None,
    dataset_dir: str | None = None,
    calib_size: int = 512,
    trust_remote_code: bool = False,
    env_vars: dict | None = None,
    **kwargs,
):
    """Run LLM export command for model conversion to ONNX.

    Args:
        torch_dir: local HF model path or name
        dtype: Target precision ("fp16", "fp8", "int4_awq", "nvfp4")
        lm_head: LM head precision (only "fp16" supported)
        output_dir: Output directory for exported model
        save_original: Save original ONNX before optimization
        config_path: Path to config.json (optional)
        dataset_dir: Calibration dataset path
        calib_size: Number of calibration samples
        trust_remote_code: Trust remote code when loading models
        env_vars: Environment variables
        **kwargs: Additional command arguments
    """
    # prepare command arguments
    cmd_args = {
        "torch_dir": torch_dir,
        "dtype": dtype,
        "lm_head": lm_head,
        "output_dir": output_dir,
        "calib_size": calib_size,
    }

    # add optional configuration parameters
    if save_original:
        cmd_args["save_original"] = None
    if config_path:
        cmd_args["config_path"] = config_path
    if dataset_dir:
        cmd_args["dataset_dir"] = dataset_dir
    if trust_remote_code:
        cmd_args["trust_remote_code"] = None

    # merge any additional keyword arguments
    cmd_args.update(kwargs)

    # Execute the command
    cmd_parts = _extend_cmd_parts(["python", "llm_export.py"], **cmd_args)
    run_example_command(cmd_parts, "onnx_ptq", env_vars=env_vars)


def run_llm_ptq_command(*, model: str, quant: str, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
    kwargs.setdefault("calib", 16)

    cmd_parts = _extend_cmd_parts(["scripts/huggingface_example.sh", "--no-verbose"], **kwargs)
    run_example_command(cmd_parts, "llm_ptq")


def run_vlm_ptq_command(*, model: str, quant: str, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
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
