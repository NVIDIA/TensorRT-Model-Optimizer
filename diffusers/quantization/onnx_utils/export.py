# Adapted https://github.com/huggingface/optimum/blob/15a162824d0c5d8aa7a3d14ab6e9bb07e5732fb6/optimum/exporters/onnx/convert.py#L573-L614

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from diffusers.models.attention_processor import Attention
from onnxmltools.utils.float16_converter import convert_float_to_float16
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from torch.onnx import export as onnx_export

from .fp8_onnx_graphsurgeon import cast_fp8_mha_io, cast_resize_io, convert_fp16_io, convert_zp_fp8

AXES_NAME = {
    "sdxl-1.0": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sdxl-turbo": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    # SD 1.5 has been removed from HF, but we’re keeping the ONNX export
    # logic in case anyone is looking to quantize a similar model.
    "sd1.5": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sd2.1": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sd2.1-base": {
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "sd3-medium": {
        "hidden_states": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "pooled_projections": {0: "batch_size"},
        "sample": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    },
    "flux-dev": {
        "hidden_states": {0: "batch_size", 1: "sequence"},
        "encoder_hidden_states": {0: "batch_size"},
        "pooled_projections": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "img_ids": {0: "batch_size", 1: "sequence"},
        "txt_ids": {0: "batch_size"},
        "guidance": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
}

# Per-tensor for INT8, we will convert it to FP8 later in onnxgraphsurgeon
SDXL_FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": None},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*lm_head*": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}


def flux_convert_rope_weight_type(onnx_graph):
    graph = gs.import_onnx(onnx_graph)
    for node in graph.nodes:
        if node.op == "Einsum":
            node.inputs[1].dtype == "float32"
            print(node.name)
    return gs.export_onnx(graph)


def generate_fp8_scales(backbone):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)
        elif isinstance(module, Attention) and (
            hasattr(module.q_bmm_quantizer, "_amax") and module.q_bmm_quantizer is not None
        ):
            module.q_bmm_quantizer._num_bits = 8
            module.q_bmm_quantizer._amax = module.q_bmm_quantizer._amax * (127 / 448.0)
            module.k_bmm_quantizer._num_bits = 8
            module.k_bmm_quantizer._amax = module.k_bmm_quantizer._amax * (127 / 448.0)
            module.v_bmm_quantizer._num_bits = 8
            module.v_bmm_quantizer._amax = module.v_bmm_quantizer._amax * (127 / 448.0)
            module.softmax_quantizer._num_bits = 8
            module.softmax_quantizer._amax = module.softmax_quantizer._amax * (127 / 448.0)


def generate_dummy_inputs(sd_version, device, is_infer=False):
    dummy_input = {}
    if sd_version == "sdxl-1.0" or sd_version == "sdxl-turbo":
        dummy_input["sample"] = torch.ones(2, 4, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 2048).to(device).half()
        dummy_input["added_cond_kwargs"] = {}
        dummy_input["added_cond_kwargs"]["text_embeds"] = torch.ones(2, 1280).to(device).half()
        dummy_input["added_cond_kwargs"]["time_ids"] = torch.ones(2, 6).to(device).half()
    elif sd_version == "sd3-medium":
        dummy_input["hidden_states"] = torch.ones(2, 16, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(2).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 333, 4096).to(device).half()
        dummy_input["pooled_projections"] = torch.ones(2, 2048).to(device).half()
    elif sd_version == "sd1.5":
        dummy_input["sample"] = torch.ones(2, 4, 64, 64).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 768).to(device).half()
    elif sd_version == "sd2.1" or sd_version == "sd2.1-base":
        dummy_input["sample"] = torch.ones(2, 4, 96, 96).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 1024).to(device).half()
    elif sd_version == "flux-dev":
        dummy_input["hidden_states"] = torch.randn(
            1, 1024 if not is_infer else 4096, 64, dtype=torch.bfloat16, device=device
        )
        dummy_input["encoder_hidden_states"] = torch.randn(
            1, 512, 4096, dtype=torch.bfloat16, device=device
        )
        dummy_input["pooled_projections"] = torch.randn(1, 768, dtype=torch.bfloat16, device=device)
        dummy_input["timestep"] = torch.randn(1, dtype=torch.bfloat16, device=device)
        dummy_input["img_ids"] = torch.randn(
            1, 1024 if not is_infer else 4096, 3, dtype=torch.float32, device=device
        )
        dummy_input["txt_ids"] = torch.randn(1, 512, 3, dtype=torch.float32, device=device)
        dummy_input["guidance"] = torch.randn(1, dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(f"Unsupported sd_version: {sd_version}")

    return dummy_input


def modelopt_export_sd(backbone, onnx_dir, model_name, precision):
    os.makedirs(f"{onnx_dir}", exist_ok=True)
    dummy_inputs = generate_dummy_inputs(model_name, device=backbone.device)

    output = Path(f"{onnx_dir}/backbone.onnx")
    if model_name == "sdxl-1.0" or model_name == "sdxl-turbo":
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["latent"]
    elif model_name == "sd1.5" or model_name == "sd2.1" or model_name == "sd2.1-base":
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        output_names = ["latent"]
    elif model_name == "sd3-medium":
        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
        output_names = ["sample"]
    elif model_name == "flux-dev":
        input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ]
        output_names = ["output"]
    else:
        raise NotImplementedError(f"Unsupported sd_version: {model_name}")

    dynamic_axes = AXES_NAME[model_name]
    do_constant_folding = True
    opset_version = 17

    # Copied from Huggingface's Optimum
    onnx_export(
        backbone,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)

    if precision == "fp8":
        onnx_model = onnx.load(str(output), load_external_data=True)
        # Iterate over all files in the folder and delete them
        output_folder = output.parent
        for file in output_folder.iterdir():
            if file.is_file():
                file.unlink()
        onnx_model = convert_zp_fp8(onnx_model)
        if model_name != "flux-dev":
            onnx_model = convert_float_to_float16(
                onnx_model, keep_io_types=True, disable_shape_infer=True
            )
            graph = gs.import_onnx(onnx_model)
            cast_resize_io(graph)
            convert_fp16_io(graph)
            cast_fp8_mha_io(graph)
            onnx_model = gs.export_onnx(graph)
        else:
            flux_convert_rope_weight_type(onnx_model)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
