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

import fnmatch

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import ResnetBlock2D, TemporalResnetBlock
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import PixArtAlphaPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline,
)

from .module import CachedModule
from .utils import replace_module

SUPPORTED_METHODS = {PixArtAlphaPipeline, StableDiffusionXLPipeline, StableVideoDiffusionPipeline}


def cachify(model, num_inference_steps, config_list):
    for name, module in model.named_modules():
        for config in config_list:
            if _pass(name, config["wildcard_or_filter_func"]) and isinstance(
                module, (Attention, ResnetBlock2D, TemporalResnetBlock, FeedForward)
            ):
                replace_module(
                    model,
                    name,
                    CachedModule(module, num_inference_steps, config["select_cache_step_func"]),
                )


def disable(pipe):
    model = get_model(pipe)
    for _, module in model.named_modules():
        if isinstance(module, CachedModule):
            module.disable_cache()


def enable(pipe):
    model = get_model(pipe)
    for _, module in model.named_modules():
        if isinstance(module, CachedModule):
            module.enable_cache()


def _pass(name, wildcard_or_filter_func):
    if isinstance(wildcard_or_filter_func, str):
        return fnmatch.fnmatch(name, wildcard_or_filter_func)
    elif callable(wildcard_or_filter_func):
        return wildcard_or_filter_func(name)
    else:
        raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")


def get_model(pipe):
    if hasattr(pipe, "unet"):
        model = pipe.unet
    elif hasattr(pipe, "transformer"):
        model = pipe.transformer
    else:
        raise KeyError

    return model


def prepare(pipe, num_inference_steps, config_list):
    assert pipe.__class__ in SUPPORTED_METHODS, f"{pipe.__class__} is not supported!"

    model = get_model(pipe)

    cachify(model, num_inference_steps, config_list)
