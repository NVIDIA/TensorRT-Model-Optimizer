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

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_hf_checkpoint
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.quantization.utils import set_quantizer_state_dict
from modelopt.torch.utils import print_rank_0

RAND_SEED = 1234


def get_lora_model(
    ckpt_path: str,
    device="cuda",
):
    """
    Loads a QLoRA model that has been trained using modelopt trainer.
    """
    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Load model with adapters
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device_map)

    # Restore modelopt state
    modelopt_state = torch.load(f"{ckpt_path}/modelopt_state_calib.pth", weights_only=False)
    restore_from_modelopt_state(model, modelopt_state)
    print_rank_0("Restored modelopt state")

    # Restore modelopt quantizer state dict
    modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
    if modelopt_weights is not None:
        set_quantizer_state_dict(model, modelopt_weights)
        print_rank_0("Restored modelopt quantizer state dict")

    return model


def main(args):
    # Load model
    model = get_lora_model(args.pyt_ckpt_path, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pyt_ckpt_path)

    # Export HF checkpoint
    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    base_model_dir = export_dir / "base_model"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    try:
        post_state_dict, hf_quant_config = _export_hf_checkpoint(model, is_modelopt_qlora=True)

        with open(f"{export_dir}/base_model/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save base model
        model.base_model.save_pretrained(f"{export_dir}/base_model", state_dict=post_state_dict)
        # Save adapters
        model.save_pretrained(export_dir)

        config_path = f"{export_dir}/base_model/config.json"

        # In the case of LoRA model.save_pretrained does not save the correct config.json
        config_data = model.config.to_dict()

        config_data["quantization_config"] = hf_quant_config

        with open(config_path, "w") as file:
            json.dump(config_data, file, indent=4)

        # Save tokenizer
        tokenizer.save_pretrained(export_dir)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        help="Specify where the PyTorch checkpoint path is",
        required=True,
    )

    parser.add_argument("--device", default="cuda")

    parser.add_argument("--export_path", default="exported_model")

    args = parser.parse_args()

    main(args)
