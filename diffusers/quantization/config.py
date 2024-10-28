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

import torch
from calib.plugin_calib import PercentileCalibrator
from utils import filter_func

FP8_FP16_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Half",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_FP32_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Float",
        },
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Float",
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
                    "trt_high_precision_dtype": "Half",
                }
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": -1,
                    "trt_high_precision_dtype": "Half",
                }
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {
                "num_bits": 8,
                "axis": 0,
                "trt_high_precision_dtype": "Half",
            }
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "trt_high_precision_dtype": "Half",
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


def set_stronglytyped_precision(quant_config, precision: str = "Half"):
    for key in quant_config["quant_cfg"].keys():
        if "trt_high_precision_dtype" in quant_config["quant_cfg"][key].keys():
            quant_config["quant_cfg"][key]["trt_high_precision_dtype"] = precision


def update_dynamic_axes(model, dynamic_axes):
    if model in ["flux-dev", "flux-schnell"]:
        dynamic_axes["out.0"] = dynamic_axes.pop("output")
    elif model in ["sdxl-1.0", "sdxl-turbo"]:
        dynamic_axes["added_cond_kwargs.text_embeds"] = dynamic_axes.pop("text_embeds")
        dynamic_axes["added_cond_kwargs.time_ids"] = dynamic_axes.pop("time_ids")
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model in ["sd2.1", "sd2.1-base"]:
        dynamic_axes["out.0"] = dynamic_axes.pop("latent")
    elif model == "sd3-medium":
        dynamic_axes["out.0"] = dynamic_axes.pop("sample")


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
    "txt_ids": {"min": [1, 512, 3], "opt": [1, 512, 3]},
    "img_ids": {"min": [1, 4096, 3], "opt": [1, 4096, 3]},
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
