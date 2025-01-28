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

import torch
from calib.plugin_calib import PercentileCalibrator
from utils import filter_func

from modelopt.core.torch.quantization.config import NVFP4_FP8_MHA_CONFIG  # noqa: F401

FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="global_min",
):
    quant_config = {
        "quant_cfg": {
            "*output_quantizer": {"enable": False},
            "default": {"enable": False},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {
                    "num_bits": 8,
                    "axis": 0,
                }
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": -1,
                }
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {
                "num_bits": 8,
                "axis": 0,
            }
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config


def get_fp4_config(model, fp4_linear_only=False):
    """fp4 for linear, optionally fp8 for conv"""

    quant_config = {
        "quant_cfg": {},
        "algorithm": "max",
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if (
            w_name in quant_config["quant_cfg"].keys()  # type: ignore
            or i_name in quant_config["quant_cfg"].keys()  # type: ignore
        ):
            continue
        if isinstance(module, torch.nn.Linear):
            quant_config["quant_cfg"][w_name] = {  # type: ignore
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            }
            quant_config["quant_cfg"][i_name] = {  # type: ignore
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            }
        elif isinstance(module, torch.nn.Conv2d):
            if fp4_linear_only:
                quant_config["quant_cfg"][w_name] = {"enable": False}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"enable": False}  # type: ignore
            else:
                # fp8 for conv
                quant_config["quant_cfg"][w_name] = {"num_bits": (4, 3), "axis": None}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"num_bits": (4, 3), "axis": None}  # type: ignore
    return quant_config


def set_quant_config_attr(quant_config, trt_high_precision_dtype):
    for _, p in quant_config["quant_cfg"].items():
        if "num_bits" in p.keys() and "trt_high_precision_dtype" not in p.keys():
            p["trt_high_precision_dtype"] = trt_high_precision_dtype


def update_dynamic_axes(model, dynamic_axes):
    if model in ["flux-dev", "flux-schnell"]:
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model in ["sdxl-1.0", "sdxl-turbo"]:
        dynamic_axes["added_cond_kwargs.text_embeds"] = dynamic_axes.pop("text_embeds")
        dynamic_axes["added_cond_kwargs.time_ids"] = dynamic_axes.pop("time_ids")
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model in ["sd2.1", "sd2.1-base"]:
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model == "sd3-medium":
        dynamic_axes["out.0"] = dynamic_axes.pop("sample")


def remove_nesting(compilation_args):
    compilation_args["dynamic_shapes"]["minShapes"]["text_embeds"] = compilation_args[
        "dynamic_shapes"
    ]["minShapes"].pop("added_cond_kwargs.text_embeds")
    compilation_args["dynamic_shapes"]["minShapes"]["time_ids"] = compilation_args[
        "dynamic_shapes"
    ]["minShapes"].pop("added_cond_kwargs.time_ids")
    compilation_args["dynamic_shapes"]["optShapes"]["text_embeds"] = compilation_args[
        "dynamic_shapes"
    ]["optShapes"].pop("added_cond_kwargs.text_embeds")
    compilation_args["dynamic_shapes"]["optShapes"]["time_ids"] = compilation_args[
        "dynamic_shapes"
    ]["optShapes"].pop("added_cond_kwargs.time_ids")


SDXL_DYNAMIC_SHAPES = {
    "sample": {"min": [2, 4, 128, 128], "opt": [16, 4, 128, 128]},
    "timestep": {"min": [1], "opt": [1]},
    "encoder_hidden_states": {"min": [2, 77, 2048], "opt": [16, 77, 2048]},
    "added_cond_kwargs.text_embeds": {"min": [2, 1280], "opt": [16, 1280]},
    "added_cond_kwargs.time_ids": {"min": [2, 6], "opt": [16, 6]},
}

SD2_DYNAMIC_SHAPES = {
    "sample": {"min": [2, 4, 96, 96], "opt": [16, 4, 96, 96]},
    "timestep": {"min": [1], "opt": [1]},
    "encoder_hidden_states": {"min": [2, 77, 1024], "opt": [16, 77, 1024]},
}

SD2_BASE_DYNAMIC_SHAPES = {
    "sample": {"min": [2, 4, 64, 64], "opt": [16, 4, 64, 64]},
    "timestep": {"min": [1], "opt": [1]},
    "encoder_hidden_states": {"min": [2, 77, 1024], "opt": [16, 77, 1024]},
}

SD3_DYNAMIC_SHAPES = {
    "hidden_states": {"min": [2, 16, 128, 128], "opt": [16, 16, 128, 128]},
    "timestep": {"min": [2], "opt": [16]},
    "encoder_hidden_states": {"min": [2, 333, 4096], "opt": [16, 333, 4096]},
    "pooled_projections": {"min": [2, 2048], "opt": [16, 2048]},
}

FLUX_DEV_DYNAMIC_SHAPES = {
    "hidden_states": {"min": [1, 4096, 64], "opt": [1, 4096, 64]},
    "timestep": {"min": [1], "opt": [1]},
    "guidance": {"min": [1], "opt": [1]},
    "pooled_projections": {"min": [1, 768], "opt": [1, 768]},
    "encoder_hidden_states": {"min": [1, 512, 4096], "opt": [1, 512, 4096]},
    "txt_ids": {"min": [512, 3], "opt": [512, 3]},
    "img_ids": {"min": [4096, 3], "opt": [4096, 3]},
}

FLUX_SCHNELL_DYNAMIC_SHAPES = FLUX_DEV_DYNAMIC_SHAPES.copy()
FLUX_SCHNELL_DYNAMIC_SHAPES.pop("guidance")


def create_dynamic_shapes(dynamic_shapes):
    min_shapes = {}
    opt_shapes = {}
    for key, value in dynamic_shapes.items():
        min_shapes[key] = value["min"]
        opt_shapes[key] = value["opt"]
    return {
        "dynamic_shapes": {
            "minShapes": min_shapes,
            "optShapes": opt_shapes,
            "maxShapes": opt_shapes,
        }
    }


DYNAMIC_SHAPES = {
    "sdxl-1.0": create_dynamic_shapes(SDXL_DYNAMIC_SHAPES),
    "sdxl-turbo": create_dynamic_shapes(SDXL_DYNAMIC_SHAPES),
    "sd2.1": create_dynamic_shapes(SD2_DYNAMIC_SHAPES),
    "sd2.1-base": create_dynamic_shapes(SD2_BASE_DYNAMIC_SHAPES),
    "sd3-medium": create_dynamic_shapes(SD3_DYNAMIC_SHAPES),
    "flux-dev": create_dynamic_shapes(FLUX_DEV_DYNAMIC_SHAPES),
    "flux-schnell": create_dynamic_shapes(FLUX_SCHNELL_DYNAMIC_SHAPES),
}

IO_SHAPES = {
    "sdxl-1.0": {"out.0": [2, 4, 128, 128]},
    "sdxl-turbo": {"out.0": [2, 4, 64, 64]},
    "sd2.1": {"out.0": [2, 4, 96, 96]},
    "sd2.1-base": {"out.0": [2, 4, 64, 64]},
    "sd3-medium": {"out.0": [2, 16, 128, 128]},
    "flux-dev": {},
    "flux-schnell": {},
}


def get_io_shapes(model, onnx_load_path):
    output_name = ""
    if onnx_load_path != "":
        if model in ["sdxl-1.0", "sdxl-turbo", "sd2.1", "sd2.1-base"]:
            output_name = "latent"
        elif model in ["flux-dev", "flux-schnell"]:
            output_name = "output"
        elif model in ["sd3-medium"]:
            output_name = "sample"
    else:
        output_name = "out.0"
    io_shapes = IO_SHAPES[model]
    # For models that are loaded from the output name will not be "out.0"
    # so we need to update the dictionary key to match the output name
    if "out.0" in io_shapes.keys():
        io_shapes[output_name] = io_shapes.pop("out.0")
    return io_shapes
