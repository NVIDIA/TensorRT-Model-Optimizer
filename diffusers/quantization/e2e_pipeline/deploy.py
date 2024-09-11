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

from pathlib import Path
from types import SimpleNamespace

import torch
from cuda import cudart
from e2e_pipeline.utlis import Engine
from onnx_utils.export import generate_dummy_inputs

from .models.trt_module import TrTBackBone

ADDITIONA_ATTRS = {
    "sdxl-1.0": ["add_embedding.linear_1.in_features"],
    "sd3-medium": [],
    "flux-dev": [],
}


def get_backbone_name(pipe):
    for attr in ["unet", "transformer"]:
        if hasattr(pipe, attr):
            return attr
    raise KeyError


def replace_backbone(pipe, pipe_name, backbone_name, eng_obj, cuda_stream, use_cuda_graph=False):
    trt_backbone = TrTBackBone(
        config=getattr(pipe, backbone_name, None).config,  # type: ignore
        engine=eng_obj,
        stream=cuda_stream,
        use_cuda_graph=use_cuda_graph,
        pipe_name=pipe_name,
    )

    other_config = ADDITIONA_ATTRS[pipe_name]
    backbone = getattr(pipe, backbone_name, None)
    assert backbone is not None
    for attr_path in other_config:
        attr_parts = attr_path.split(".")
        current_attr_old = backbone
        for part in attr_parts[:-1]:
            current_attr_old = getattr(current_attr_old, part)
        old_value = getattr(current_attr_old, attr_parts[-1], None)
        current_attr_new = trt_backbone
        for part in attr_parts[:-1]:
            if not hasattr(current_attr_new, part):
                setattr(current_attr_new, part, SimpleNamespace())
            current_attr_new = getattr(current_attr_new, part)
        setattr(current_attr_new, attr_parts[-1], old_value)

    del backbone
    setattr(pipe, backbone_name, None)
    torch.cuda.empty_cache()
    setattr(pipe, backbone_name, trt_backbone)


def get_nested_keys_values(input_dict):
    keys_values = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # Directly add the nested keys without appending the parent key
            nested_dict = get_nested_keys_values(value)
            keys_values.update(nested_dict)
        else:
            keys_values[key] = value
    return keys_values


def load_engine(engine_path, dummy_inputs, batch_size: int = 2, pipe_name: str = "sdxl-1.0"):
    eng_obj = Engine()
    eng_obj.load(str(engine_path))
    _, shared_device_memory = cudart.cudaMalloc(eng_obj.engine.device_memory_size)
    eng_obj.activate(shared_device_memory)
    shape_dict = get_nested_keys_values(dummy_inputs)
    for key, value in shape_dict.items():
        shape_tuple = tuple(value.shape)
        if key != "timestep" or pipe_name == "sd3-medium" or pipe_name == "flux-dev":
            _value = (batch_size,) + shape_tuple[1:]
            shape_dict[key] = _value
        else:
            shape_dict[key] = shape_tuple
    if pipe_name == "flux-dev":
        # The value 4096 is for an image size of 1024x1024. If you want to generate
        # an image with a different size, please adjust the value accordingly.
        shape_dict["output"] = (batch_size,) + (4096, 64)
    eng_obj.allocate_buffers(shape_dict=shape_dict, device="cuda", infer_batch_size=batch_size)
    return eng_obj


def load(
    pipe,
    pipe_name: str,
    engine_path: Path,
    infer_batchsize: int,
):
    backbone_name = get_backbone_name(pipe)
    dummy_inputs = generate_dummy_inputs(pipe_name, "cuda", True)
    if not engine_path.exists():
        raise NotImplementedError

    eng_obj = load_engine(
        engine_path, get_nested_keys_values(dummy_inputs), infer_batchsize, pipe_name
    )
    cuda_stream = cudart.cudaStreamCreate()[1]
    replace_backbone(pipe, pipe_name, backbone_name, eng_obj, cuda_stream, False)
