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

"""A collection of common models for testing."""

from typing import Any, Callable
from urllib.error import HTTPError

import pytest
import timm
import torch
import torch.nn as nn
import torchvision

from modelopt.torch.utils import torch_to

__all__ = ["get_benchmark_models"]


def _process_model_and_inputs(model, inputs, on_gpu):
    try:
        model.eval()

        # move to GPU if desired
        model = model.cuda() if on_gpu else model.cpu()
        inputs = torch_to(inputs, "cuda" if on_gpu else "cpu")
        args, kwargs = inputs["args"], inputs["kwargs"]

        return model, args, kwargs
    except HTTPError:
        pytest.xfail("Rate limit exceeded")


def _create_model_fn_for_torchhub(repo_or_dir, name, get_input, **kwargs):
    def get_model_and_input(on_gpu: bool = False):
        # skip_validation to avoid `HTTP Error 403: rate limit exceeded
        model = torch.hub.load(repo_or_dir, name, skip_validation=True, **kwargs)
        return _process_model_and_inputs(model, get_input(), on_gpu)

    return get_model_and_input


def _create_rand_input_fn(*shape):
    def get_input():
        return {"args": (torch.randn(*shape),), "kwargs": {}}

    return get_input


def _create_facebook_fn(name):
    return _create_model_fn_for_torchhub(
        "facebookresearch/pytorchvideo:0.1.3", name, _create_rand_input_fn(1, 3, 20, 224, 224)
    )


def _create_intel_fn(name):
    return _create_model_fn_for_torchhub(
        "intel-isl/MiDaS:f28885af", name, _create_rand_input_fn(1, 3, 320, 640)
    )


def _create_ultralytics_fn(name):
    return _create_model_fn_for_torchhub(
        "ultralytics/yolov5:v6.2",
        name,
        _create_rand_input_fn(1, 3, 320, 320),
        autoshape=False,
        verbose=False,
    )


def _create_timm_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        model = timm.create_model(name)
        return _process_model_and_inputs(model, _create_rand_input_fn(1, 3, 224, 224)(), on_gpu)

    return get_model_and_input


def _create_torchvision_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        model = getattr(torchvision.models, name)(progress=False)
        return _process_model_and_inputs(model, _create_rand_input_fn(1, 3, 224, 224)(), on_gpu)

    return get_model_and_input


def _create_torchvision_segmentation_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        model = getattr(torchvision.models.segmentation, name)(progress=False)
        return _process_model_and_inputs(model, _create_rand_input_fn(1, 3, 512, 512)(), on_gpu)

    return get_model_and_input


def _create_unet_fn(name):
    return _create_model_fn_for_torchhub(
        "milesial/Pytorch-UNet", name, _create_rand_input_fn(1, 3, 320, 320)
    )


MODELS = {
    # "facebookresearch": (
    #     [
    #         "x3d_s",  # TODO: hub seems to be broken...
    #     ],
    #     _create_facebook_fn,
    # ),
    # numerically unstable, and often facing HTTPError, Removing it!
    # "intel-isl": (["MiDaS_small"], _create_intel_fn),
    "timm": (
        [
            # "dla34", # numerically unstable, removing it!
            "hrnet_w18_small",
            "cspresnet50",
            "dpn68",
            "vit_tiny_patch16_224",
            "vovnet39a",
            "dm_nfnet_f0",
            "efficientnet_b0",
        ],
        _create_timm_fn,
    ),
    "torchvision": (
        [
            "resnet18",
            "mobilenet_v2",
            "mobilenet_v3_small",
        ],
        _create_torchvision_fn,
    ),
    "torchvision_segmentation": (
        [
            "deeplabv3_resnet50",
            "fcn_resnet50",
            "lraspp_mobilenet_v3_large",
        ],
        _create_torchvision_segmentation_fn,
    ),
    # Additional dependencies for ultralytics/yolov5: opencv-python-headless, pandas, matplotlib, seaborn
    # Disabled since test often fails during download of checkpoint
    # "ultralytics": (
    #     [
    #         "yolov5s",
    #     ],
    #     _create_ultralytics_fn,
    # ),
    "unet": (
        [
            "unet_carvana",
        ],
        _create_unet_fn,
    ),
}


def get_benchmark_models() -> dict[str, Callable[[bool], tuple[nn.Module, Any, Any]]]:
    """Returns a dict of benchmark model name to a function that returns the model, args, kwargs."""
    return {f"{repo}/{name}": fn(name) for repo, (names, fn) in MODELS.items() for name in names}
