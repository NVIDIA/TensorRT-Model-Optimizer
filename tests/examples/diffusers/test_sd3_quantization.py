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


from _test_utils.examples.run_command import run_diffusers_cmd

# Use the tiny models for faster testing
# NOTE: For tiny sdxl and sd3, we have to use Float dtype instead of Half from transformers v4.49.0
SD3_ARGS = ["--model", "sd3-medium", "--override-model-path", "hf-internal-testing/tiny-sd3-pipe"]
SD3_DTYPE = "Float"


# fmt: off
def test_sd3_int8_level3(int8_args, tmp_path):
    torch_ckpt_path = tmp_path / "sd3-medium.int8.level3.pt"
    onnx_dir = tmp_path / "sd3-medium.int8.level3.onnx"
    restore_onnx_dir = tmp_path / "sd3-medium.int8.level3.restore.onnx"

    # INT8 Level 3
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *SD3_ARGS, *int8_args,
            "--model-dtype", SD3_DTYPE,
            "--trt-high-precision-dtype", SD3_DTYPE,
            "--batch-size", "2",
            "--quantized-torch-ckpt-save-path", torch_ckpt_path,
            "--onnx-dir", onnx_dir,
        ],
    )

    # INT8 Level 3 Restore
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *SD3_ARGS, *int8_args,
            "--model-dtype", SD3_DTYPE,
            "--trt-high-precision-dtype", SD3_DTYPE,
            "--restore-from", torch_ckpt_path,
            "--onnx-dir", restore_onnx_dir,
        ],
    )

    # Inference - DQ only
    run_diffusers_cmd(
        [
            "python", "diffusion_trt.py",
            *SD3_ARGS,
            "--model-dtype", SD3_DTYPE,
            "--onnx-load-path", onnx_dir / "model.onnx",
            "--dq-only",
        ],
    )
