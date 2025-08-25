# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os

from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
from modelopt.torch.quantization.plugins.transformers_trainer import (
    convert_sharded_model_to_hf_format,
)

# Enable ModelOpt checkpointing for HuggingFace models
mto.enable_huggingface_checkpointing()


def main():
    parser = argparse.ArgumentParser(description="Convert sharded checkpoint to HuggingFace format")
    parser.add_argument(
        "--hf_model_path", type=str, required=True, help="Path to the original HuggingFace model"
    )
    parser.add_argument(
        "--sharded_ckpt_path",
        type=str,
        required=True,
        help="Path to the sharded checkpoint directory",
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Output path to save the converted model"
    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.hf_model_path)
    if os.path.exists(os.path.join(args.sharded_ckpt_path, "pytorch_model_fsdp_0")):
        convert_sharded_model_to_hf_format(
            model, args.sharded_ckpt_path, "modelopt_state_train.pth", args.output_path
        )


if __name__ == "__main__":
    main()
