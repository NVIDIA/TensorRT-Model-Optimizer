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
FLUX_ARGS = [
    "--model",
    "flux-schnell",
    "--override-model-path",
    "hf-internal-testing/tiny-flux-pipe",
]
FLUX_DTYPE = "BFloat16"


# fmt: off
def test_flux_int8_level3(int8_args, tmp_path):
    torch_ckpt_path = tmp_path / "flux-schnell.int8.level3.pt"
    onnx_dir = tmp_path / "flux-schnell.int8.level3.onnx"

    # INT8 Level 3
    run_diffusers_cmd(
        [
            "python", "quantize.py",
            *FLUX_ARGS, *int8_args,
            "--model-dtype", FLUX_DTYPE,
            "--trt-high-precision-dtype", FLUX_DTYPE,
            "--batch-size", "1",
            "--quantized-torch-ckpt-save-path", torch_ckpt_path,
            "--onnx-dir", onnx_dir,
        ],
    )

    # Inference - DQ only
    run_diffusers_cmd(
        [
            "python", "diffusion_trt.py",
            *FLUX_ARGS,
            "--model-dtype", FLUX_DTYPE,
            "--onnx-load-path", onnx_dir / "model.onnx",
            "--dq-only",
        ],
    )
