# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import re

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import load_image

import modelopt.torch.quantization as mtq

USE_PEFT = True
try:
    from peft.tuners.lora.layer import Conv2d as PEFTLoRAConv2d
    from peft.tuners.lora.layer import Linear as PEFTLoRALinear
except ModuleNotFoundError:
    USE_PEFT = False


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(model_id, backbone, quant_level=2.5, enable_conv_3d=True):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ) and name != "proj_out":  # Disable the final output layer from flux model
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, torch.nn.Conv3d) and not enable_conv_3d:
            """
                Error: Torch bug, ONNX export failed due to unknown kernel shape in QuantConv3d.
                TRT_FP8QuantizeLinear and TRT_FP8DequantizeLinear operations in UNetSpatioTemporalConditionModel for svd
                cause issues. Inputs on different devices (CUDA vs CPU) may contribute to the problem.
            """
            module.input_quantizer.disable()
            module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            # TRT only supports FP8 MHA with head_size % 16 == 0.
            head_size = int(module.inner_dim / module.heads)
            if quant_level >= 4 and head_size % 16 == 0:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
                if model_id == "flux-dev":
                    if name.startswith("transformer_blocks"):
                        module.bmm2_output_quantizer.enable()
                    else:
                        module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", False)
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
                module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", True)


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    with open(calib_data_path, "r", encoding="utf8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def load_calib_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = load_image(img_path)
            if image is not None:
                images.append(image)
    return images


def set_fmha(unet):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor())


def check_lora(unet):
    for name, module in unet.named_modules():
        if isinstance(module, (LoRACompatibleConv, LoRACompatibleLinear)):
            assert (
                module.lora_layer is None
            ), f"To quantize {name}, LoRA layer should be fused/merged. Please fuse the LoRA layer before quantization."
        elif USE_PEFT and isinstance(module, (PEFTLoRAConv2d, PEFTLoRALinear)):
            assert module.merged, (
                f"To quantize {name}, LoRA layer should be fused/merged. Please"
                " fuse the LoRA layer before quantization."
            )


def fp8_mha_disable(backbone, quantized_mha_output: bool = True):
    def mha_filter_func(name):
        pattern = re.compile(
            r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer).*"
            if quantized_mha_output
            else r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
        )
        return pattern.match(name) is not None

    if hasattr(F, "scaled_dot_product_attention"):
        mtq.disable_quantizer(backbone, mha_filter_func)
