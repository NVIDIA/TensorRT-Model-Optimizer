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

from warnings import warn

from _test_utils.examples.run_command import run_diffusers_cmd

# Use the tiny models for faster testing
# NOTE: For tiny sdxl and sd3, we have to use Float dtype instead of Half from transformers v4.49.0
SDXL_ARGS = ["--model", "sdxl-1.0", "--override-model-path", "hf-internal-testing/tiny-sdxl-pipe"]
SDXL_DTYPE = "Float"


# fmt: off
def test_sdxl_fp8_level3(fp8_args, tmp_path, cuda_capability):
    torch_ckpt_path = tmp_path / "sdxl.fp8.level3.pt"
    onnx_dir = tmp_path / "sdxl.fp8.level3.onnx"
    restore_onnx_dir = tmp_path / "sdxl.fp8.level3.restore.onnx"

    # FP8 Level 3
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *SDXL_ARGS, *fp8_args,
            "--model-dtype", SDXL_DTYPE,
            "--trt-high-precision-dtype", SDXL_DTYPE,
            "--quantized-torch-ckpt-save-path", torch_ckpt_path,
            "--onnx-dir", onnx_dir,
        ],
    )

    # FP8 Level 3 Restore
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *SDXL_ARGS, *fp8_args,
            "--model-dtype", SDXL_DTYPE,
            "--trt-high-precision-dtype", SDXL_DTYPE,
            "--restore-from", torch_ckpt_path,
            "--onnx-dir", restore_onnx_dir,
        ],
    )

    # Inference - DQ only
    if cuda_capability >= (8, 9):
        run_diffusers_cmd(
            [
                "python", "diffusion_trt.py",
                *SDXL_ARGS,
                "--model-dtype", SDXL_DTYPE,
                "--onnx-load-path", onnx_dir / "model.onnx",
                "--dq-only",
            ],
        )
    else:
        warn("CUDA capability >= 8.9 is required for FP8 inference!")


def test_sdxl_int8_level3(int8_args, tmp_path):
    torch_ckpt_path = tmp_path / "sdxl.int8.level3.pt"
    onnx_dir = tmp_path / "sdxl.int8.level3.onnx"

    # INT8 Level 3
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *SDXL_ARGS, *int8_args,
            "--model-dtype", SDXL_DTYPE,
            "--trt-high-precision-dtype", SDXL_DTYPE,
            "--batch-size", "2",
            "--quantized-torch-ckpt-save-path", torch_ckpt_path,
            "--onnx-dir", onnx_dir,
        ],
    )

    # Inference
    run_diffusers_cmd(
        [
            "python", "diffusion_trt.py",
            *SDXL_ARGS,
            "--model-dtype", SDXL_DTYPE,
            "--restore-from", torch_ckpt_path,
        ],
    )

    # Inference - DQ only
    run_diffusers_cmd(
        [
            "python", "diffusion_trt.py",
            *SDXL_ARGS,
            "--model-dtype", SDXL_DTYPE,
            "--onnx-load-path", onnx_dir / "model.onnx",
            "--dq-only",
        ],
    )
