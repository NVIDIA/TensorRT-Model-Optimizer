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

import types
from pathlib import Path

import tensorrt as trt
import torch
from cuda import cudart
from pipeline.config import SDXL_ONNX_CONFIG
from pipeline.models import (
    cachecrossattnupblock2d_forward,
    cacheunet_forward,
    cacheupblock2d_forward,
)
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from torch.onnx import export as onnx_export

from .utils import Engine


def replace_new_forward(unet):
    unet.forward = types.MethodType(cacheunet_forward, unet)
    for upsample_block in unet.up_blocks:
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            upsample_block.forward = types.MethodType(
                cachecrossattnupblock2d_forward, upsample_block
            )
        else:
            upsample_block.forward = types.MethodType(cacheupblock2d_forward, upsample_block)


def get_input_info(dummy_dict, info=None):
    return_val = [] if info == "profile_shapes" or info == "input_names" else {}

    def collect_leaf_keys(d):
        for key, value in d.items():
            if isinstance(value, dict):
                collect_leaf_keys(value)
            else:
                if info == "profile_shapes":
                    return_val.append((key, value))  # type: ignore
                elif info == "profile_shapes_dict":
                    return_val[key] = value  # type: ignore
                elif info == "dummy_input":
                    return_val[key] = torch.ones(value).half().cuda()  # type: ignore
                elif info == "input_names":
                    return_val.append(key)  # type: ignore

    collect_leaf_keys(dummy_dict)
    return return_val


def complie2trt(onnx_path: Path, engine_path: Path):
    subdirs = [f for f in onnx_path.iterdir() if f.is_dir()]
    for subdir in subdirs:
        if subdir.name not in SDXL_ONNX_CONFIG.keys():
            continue
        model_path = subdir / "model.onnx"
        plan_path = engine_path / f"{subdir.name}.plan"
        if not plan_path.exists():
            print(f"Building {str(model_path)}")
            build_profile = Profile()
            profile_shapes = get_input_info(
                SDXL_ONNX_CONFIG[subdir.name]["dummy_input"], "profile_shapes"
            )
            for input_name, input_shape in profile_shapes:
                build_profile.add(input_name, input_shape, input_shape, input_shape)
            block_network = network_from_onnx_path(
                str(model_path), flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
            build_config = CreateConfig(
                fp16=True,
                profiles=[build_profile],
            )
            engine = engine_from_network(
                block_network,
                config=build_config,
            )
            save_engine(engine, path=plan_path)
        else:
            print(f"{str(model_path)} already exists!")


def get_total_device_memory(unet):
    max_device_memory = 0
    for _, engine in unet.engines.items():
        max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
    return max_device_memory


def load_engines(unet, engine_path: Path):
    unet.engines = {}
    for f in engine_path.iterdir():
        if f.is_file():
            eng = Engine()
            eng.load(str(f))
            unet.engines[f"{f.stem}"] = eng
    _, shared_device_memory = cudart.cudaMalloc(get_total_device_memory(unet))
    for engine in unet.engines.values():
        engine.activate(shared_device_memory)
    unet.cuda_stream = cudart.cudaStreamCreate()[1]
    for block_name in unet.engines.keys():
        unet.engines[block_name].allocate_buffers(
            shape_dict=get_input_info(
                SDXL_ONNX_CONFIG[block_name]["dummy_input"], "profile_shapes_dict"
            ),
            device=unet.device,
        )
    # TODO: Free and clean up the origin pytorch cuda memory


def export_onnx(unet, onnx_path: Path):

    # In order to export these models into ONNX, we need modify some of the forward function
    # 1, Down_block
    for i, downsample_block in enumerate(unet.down_blocks):
        _onnx_dir = onnx_path.joinpath(f"down_blocks.{i}")
        _onnx_file = _onnx_dir.joinpath("model.onnx")
        if not _onnx_file.exists():
            _onnx_dir.mkdir(parents=True, exist_ok=True)
            dummy_input = get_input_info(
                SDXL_ONNX_CONFIG[f"down_blocks.{i}"]["dummy_input"], "dummy_input"
            )
            input_names = get_input_info(
                SDXL_ONNX_CONFIG[f"down_blocks.{i}"]["dummy_input"], "input_names"
            )
            output_names = SDXL_ONNX_CONFIG[f"down_blocks.{i}"]["output_names"]
            onnx_export(
                downsample_block,
                args=dummy_input,
                f=_onnx_file.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=SDXL_ONNX_CONFIG[f"down_blocks.{i}"]["dynamic_axes"],
                do_constant_folding=True,
                opset_version=17,
            )
        else:
            print(f"{str(_onnx_file)} alread exists!")

    # Mid block
    if unet.mid_block is not None:
        _onnx_dir = onnx_path.joinpath("mid_block")
        _onnx_file = _onnx_dir.joinpath("model.onnx")
        if not _onnx_file.exists():
            _onnx_dir.mkdir(parents=True, exist_ok=True)
            dummy_input = get_input_info(
                SDXL_ONNX_CONFIG["mid_block"]["dummy_input"], "dummy_input"
            )
            input_names = get_input_info(
                SDXL_ONNX_CONFIG["mid_block"]["dummy_input"], "input_names"
            )
            output_names = SDXL_ONNX_CONFIG["mid_block"]["output_names"]
            onnx_export(
                unet.mid_block,
                args=dummy_input,
                f=_onnx_file.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=SDXL_ONNX_CONFIG["mid_block"]["dynamic_axes"],
                do_constant_folding=True,
                opset_version=17,
            )
        else:
            print(f"{str(_onnx_file)} alread exists!")

    # Up_block
    for i, up_block in enumerate(unet.up_blocks):
        _onnx_dir = onnx_path.joinpath(f"up_blocks.{i}")
        _onnx_file = _onnx_dir.joinpath("model.onnx")
        if not _onnx_file.exists():
            _onnx_dir.mkdir(parents=True, exist_ok=True)
            dummy_input = get_input_info(
                SDXL_ONNX_CONFIG[f"up_blocks.{i}"]["dummy_input"], "dummy_input"
            )
            input_names = get_input_info(
                SDXL_ONNX_CONFIG[f"up_blocks.{i}"]["dummy_input"], "input_names"
            )
            output_names = SDXL_ONNX_CONFIG[f"up_blocks.{i}"]["output_names"]
            onnx_export(
                up_block,
                args=dummy_input,
                f=_onnx_file.as_posix(),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=SDXL_ONNX_CONFIG[f"up_blocks.{i}"]["dynamic_axes"],
                do_constant_folding=True,
                opset_version=17,
            )
        else:
            print(f"{str(_onnx_file)} alread exists!")


def warm_up(unet):
    print("Warming-up TensorRT engines...")
    for name, engine in unet.engines.items():
        dummy_input = get_input_info(SDXL_ONNX_CONFIG[name]["dummy_input"], "dummy_input")
        _ = engine(dummy_input, unet.cuda_stream)


def teardown(unet):
    for engine in unet.engines.values():
        del engine

    cudart.cudaStreamDestroy(unet.cuda_stream)
    del unet.cuda_stream


def compile(unet, onnx_path: Path, engine_path: Path):
    onnx_path.mkdir(parents=True, exist_ok=True)
    engine_path.mkdir(parents=True, exist_ok=True)

    replace_new_forward(unet)
    export_onnx(unet, onnx_path)
    complie2trt(onnx_path, engine_path)
    load_engines(unet, engine_path)
    warm_up(unet)
    unet.use_trt_infer = True
