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
import shutil
import tempfile
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from onnxmltools.utils.float16_converter import convert_float_to_float16
from torch.onnx import export as onnx_export

from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

from .fp8_onnx_graphsurgeon import cast_resize_io, convert_fp16_io, convert_zp_fp8

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
        "hidden_states": {0: "batch_size", 1: "latent_dim"},
        "encoder_hidden_states": {0: "batch_size"},
        "pooled_projections": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "img_ids": {0: "latent_dim"},
        "guidance": {0: "batch_size"},
        "latent": {0: "batch_size"},
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
            node.inputs[1].dtype = "float32"
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


def generate_dummy_inputs(model_id, device, model_dtype="BFloat16"):
    dummy_input = {}
    if model_id == "sdxl-1.0" or model_id == "sdxl-turbo":
        dummy_input["sample"] = torch.ones(2, 4, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 2048).to(device).half()
        dummy_input["added_cond_kwargs"] = {}
        dummy_input["added_cond_kwargs"]["text_embeds"] = torch.ones(2, 1280).to(device).half()
        dummy_input["added_cond_kwargs"]["time_ids"] = torch.ones(2, 6).to(device).half()
        dummy_input["return_dict"] = False
    elif model_id == "sd3-medium":
        dummy_input["hidden_states"] = torch.ones(2, 16, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(2).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 333, 4096).to(device).half()
        dummy_input["pooled_projections"] = torch.ones(2, 2048).to(device).half()
        dummy_input["return_dict"] = False
    elif model_id == "sd2.1":
        dummy_input["sample"] = torch.ones(2, 4, 96, 96).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 1024).to(device).half()
        dummy_input["return_dict"] = False
    elif model_id == "sd2.1-base":
        dummy_input["sample"] = torch.ones(2, 4, 64, 64).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 1024).to(device).half()
        dummy_input["return_dict"] = False
    elif model_id == "flux-dev":
        text_maxlen = 512
        torch_dtype = torch.bfloat16 if model_dtype == "BFloat16" else torch.float16
        dummy_input["hidden_states"] = torch.randn(1, 1024, 64, dtype=torch_dtype, device=device)
        dummy_input["encoder_hidden_states"] = torch.randn(
            1, text_maxlen, 4096, dtype=torch_dtype, device=device
        )
        dummy_input["pooled_projections"] = torch.randn(1, 768, dtype=torch_dtype, device=device)
        dummy_input["timestep"] = torch.tensor(data=[1.0] * 1, dtype=torch_dtype, device=device)
        dummy_input["img_ids"] = torch.randn(1024, 3, dtype=torch.float32, device=device)
        dummy_input["txt_ids"] = torch.randn(text_maxlen, 3, dtype=torch.float32, device=device)
        dummy_input["guidance"] = torch.full((1,), 3.5, dtype=torch.float32, device=device)
        dummy_input["return_dict"] = False
    else:
        raise NotImplementedError(f"Unsupported model_id: {model_id}")

    return dummy_input


def save_onnx(onnx_model, output):
    onnx.save(
        onnx_model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + "_data",
        size_threshold=1024,
    )
    print(f"ONNX model saved to {output}")


def set_onnx_export_attr(model):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"


def modelopt_export_sd(backbone, onnx_dir, model_name, precision, model_dtype="BFloat16"):
    model_file_name = "model.onnx"
    os.makedirs(f"{onnx_dir}", exist_ok=True)
    tmp_subfolder = tempfile.mkdtemp(prefix="myapp_")
    tmp_output = Path(f"{tmp_subfolder}/{model_file_name}")
    q_output = Path(f"{onnx_dir}/{model_file_name}")

    if precision == "fp4":
        set_onnx_export_attr(backbone)

    dummy_inputs = generate_dummy_inputs(
        model_name, device=backbone.device, model_dtype=model_dtype
    )

    if model_name == "sdxl-1.0" or model_name == "sdxl-turbo":
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["latent"]
    elif model_name == "sd3-medium":
        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
        output_names = ["sample"]
    elif model_name in ["flux-dev"]:
        input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ]
        output_names = ["latent"]
    elif model_name == "sd2.1" or model_name == "sd2.1-base":
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        output_names = ["latent"]
    else:
        raise NotImplementedError(f"Unsupported model_id: {model_name}")

    dynamic_axes = AXES_NAME[model_name]
    do_constant_folding = True
    opset_version = 20

    onnx_export(
        backbone,
        (dummy_inputs,),
        f=tmp_output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )
    print(f"Saved at {tmp_output}")
    onnx_model = onnx.load(str(tmp_output), load_external_data=True)
    if precision == "fp8":
        if not model_name.startswith("flux"):
            graph = gs.import_onnx(onnx_model)
            graph.cleanup().toposort()
            convert_fp16_io(graph)
            onnx_model = gs.export_onnx(graph)
            onnx_model = convert_zp_fp8(onnx_model)
            onnx_model = convert_float_to_float16(
                onnx_model, keep_io_types=True, disable_shape_infer=True
            )
            graph = gs.import_onnx(onnx_model)
            cast_resize_io(graph)
            onnx_model = gs.export_onnx(graph.cleanup())
        else:
            flux_convert_rope_weight_type(onnx_model)

    if precision == "fp4":
        onnx_model = fp4qdq_to_2dq(onnx_model)
    save_onnx(onnx_model, q_output)
    shutil.rmtree(tmp_subfolder, ignore_errors=True)
