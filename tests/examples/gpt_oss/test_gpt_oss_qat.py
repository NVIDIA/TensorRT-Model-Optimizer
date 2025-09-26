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

import pytest
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch_misc import minimum_gpu


class GPTOSS:
    """Test GPT-OSS-20B QAT (Quantization-Aware Training) pipeline.

    This test suite covers the complete GPT-OSS-20B optimization pipeline:

    Step 1: test_gpt_oss_sft_training - Supervised Fine-Tuning (SFT)
           Input: openai/gpt-oss-20b
           Output: gpt-oss-20b-sft

    Step 2: test_gpt_oss_qat_training - Quantization-Aware Training (QAT)
           Input: gpt-oss-20b-sft (from Step 1)
           Output: gpt-oss-20b-qat

    Step 3: test_gpt_oss_mxfp4_conversion - MXFP4 Weight-Only Conversion
           Input: gpt-oss-20b-qat (from Step 2)
           Output: gpt-oss-20b-qat-real-mxfp4

    Each step can be run independently (with mock inputs) or as part of the full pipeline.
    """

    def __init__(self, model_path):
        self.model_path = model_path

    def gpt_oss_sft_training(self, tmp_path):
        """Test supervised fine-tuning (SFT) of GPT-OSS-20B model - Step 1."""
        output_dir = tmp_path / "gpt-oss-20b-sft"

        # Command for SFT training (Step 1)
        cmd_parts = [
            "accelerate",
            "launch",
            "--config_file",
            "configs/zero3.yaml",
            "sft.py",
            "--config",
            "configs/sft_full.yaml",
            "--model_name_or_path",
            self.model_path,
            "--output_dir",
            str(output_dir),
        ]

        run_example_command(cmd_parts, "gpt-oss")

        # Verify SFT output directory exists
        assert output_dir.exists(), "SFT output directory should exist after training"

    def gpt_oss_qat_training(self, tmp_path):
        """Test quantization-aware training (QAT) with MXFP4 configuration - Step 2."""
        # This test assumes test_gpt_oss_sft_training has been run first
        # Look for the SFT output directory from step 1
        sft_dir = tmp_path / "gpt-oss-20b-sft"

        # If SFT directory doesn't exist, create a mock one for standalone testing
        if not sft_dir.exists():
            sft_dir.mkdir()

            # Create minimal config.json for the mock model
            config_content = {
                "model_type": "gpt_oss",
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_hidden_layers": 44,
                "vocab_size": 100000,
                "torch_dtype": "bfloat16",
            }

            import json

            with open(sft_dir / "config.json", "w") as f:
                json.dump(config_content, f)

        qat_output_dir = tmp_path / "gpt-oss-20b-qat"

        # Command for QAT training (Step 2)
        cmd_parts = [
            "accelerate",
            "launch",
            "--config_file",
            "configs/zero3.yaml",
            "sft.py",
            "--config",
            "configs/sft_full.yaml",
            "--model_name_or_path",
            str(sft_dir),
            "--quant_cfg",
            "MXFP4_MLP_WEIGHT_ONLY_CFG",
            "--output_dir",
            str(qat_output_dir),
        ]

        run_example_command(cmd_parts, "gpt-oss")

        # Verify QAT output directory exists
        assert qat_output_dir.exists(), "QAT output directory should exist after training"

    def gpt_oss_mxfp4_conversion(self, tmp_path):
        """Test conversion to MXFP4 weight-only format - Step 3."""
        # This test assumes test_gpt_oss_qat_training has been run first
        # Look for the QAT output directory from step 2
        qat_dir = tmp_path / "gpt-oss-20b-qat"

        # If QAT directory doesn't exist, create a mock one for standalone testing
        if not qat_dir.exists():
            qat_dir.mkdir()

            # Create minimal config.json for the mock model
            config_content = {
                "model_type": "gpt_oss",
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_hidden_layers": 44,
                "vocab_size": 100000,
                "torch_dtype": "bfloat16",
            }

            import json

            with open(qat_dir / "config.json", "w") as f:
                json.dump(config_content, f)

        conversion_output_dir = tmp_path / "gpt-oss-20b-qat-real-mxfp4"

        # Command for MXFP4 conversion (Step 3)
        cmd_parts = [
            "python",
            "convert_oai_mxfp4_weight_only.py",
            "--model_path",
            str(qat_dir),
            "--output_path",
            str(conversion_output_dir),
        ]

        run_example_command(cmd_parts, "gpt-oss")

        # Verify conversion output directory exists
        assert conversion_output_dir.exists(), "MXFP4 conversion output directory should exist"

    def deploy_gpt_oss_trtllm(self, tmp_path):
        """Deploy GPT-OSS model with TensorRT-LLM."""
        # Prepare benchmark data
        tensorrt_llm_workspace = "/app/tensorrt_llm"
        script = os.path.join(tensorrt_llm_workspace, "benchmarks", "cpp", "prepare_dataset.py")
        model_name = self.model_path.split("/")[-1]
        benchmark_file = f"{model_name}_synthetic_128_128.txt"

        if not os.path.exists(benchmark_file) or os.path.getsize(benchmark_file) == 0:
            print(f"Creating dataset file '{benchmark_file}'...")
            with open(benchmark_file, "w") as fp:
                subprocess.run(
                    f"python {script} --stdout --tokenizer={self.model_path} token-norm-dist --input-mean 128 \
                    --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1400",
                    shell=True,
                    check=True,
                    stdout=fp,
                )
        else:
            print(f"Dataset file '{benchmark_file}' already exists.")

        assert os.path.isfile(benchmark_file), f"Benchmark file '{benchmark_file}' should exist"

        cmd_parts = [
            "trtllm-bench",
            "--model",
            self.model_path,
            "throughput",
            "--backend",
            "pytorch",
            "--dataset",
            benchmark_file,
            "--kv_cache_free_gpu_mem_fraction",
            "0.9",
            "--report_json",
            str(tmp_path / "low_latency_throughput.json"),
        ]
        run_example_command(cmd_parts, "gpt-oss")


@pytest.mark.parametrize(
    "model_path",
    [
        pytest.param("openai/gpt-oss-20b", id="gpt-oss-20b", marks=minimum_gpu(2)),
        pytest.param("openai/gpt-oss-120b", id="gpt-oss-120b", marks=minimum_gpu(8)),
    ],
)
def test_gpt_oss_complete_pipeline(model_path, tmp_path):
    """Test the complete GPT-OSS optimization pipeline by executing all 3 steps in sequence."""
    # Create GPTOSS instance with model path
    gpt_oss = GPTOSS(model_path)
    model_name = model_path.split("/")[-1]

    # Execute Step 1: SFT Training
    gpt_oss.gpt_oss_sft_training(tmp_path)

    # Execute Step 2: QAT Training
    gpt_oss.gpt_oss_qat_training(tmp_path)

    # Execute Step 3: MXFP4 Conversion
    gpt_oss.gpt_oss_mxfp4_conversion(tmp_path)

    # Verify all output directories exist
    sft_dir = tmp_path / f"{model_name}-sft"
    qat_dir = tmp_path / f"{model_name}-qat"
    conversion_dir = tmp_path / f"{model_name}-qat-real-mxfp4"

    assert sft_dir.exists(), "SFT output directory should exist after Step 1"
    assert qat_dir.exists(), "QAT output directory should exist after Step 2"
    assert conversion_dir.exists(), "MXFP4 conversion output directory should exist after Step 3"

    print(f"Complete pipeline executed successfully for {model_path}!")
    print(f"Step 1 output: {sft_dir}")
    print(f"Step 2 output: {qat_dir}")
    print(f"Step 3 output: {conversion_dir}")

    # Deploy with TensorRT-LLM
    gpt_oss.deploy_gpt_oss_trtllm(tmp_path)
