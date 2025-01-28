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

"""Default configurations for NAS modes."""

from pydantic import create_model

from modelopt.torch.opt.config import (
    ModeloptBaseConfig,
    ModeloptField,
    get_kwargs_for_create_model_with_rules,
)

from .registry import DMRegistry

__all__ = ["AutoNASConfig", "ExportConfig"]


def _get_ratio_list():
    return (0.5, 0.67, 1.0)


def _conv_config():
    return {
        "channels_ratio": _get_ratio_list(),
        "kernel_size": (),
        "channel_divisor": 32,
    }


def _norm_lin_config():
    return {
        "features_ratio": _get_ratio_list(),
        "feature_divisor": 32,
    }


AutoNASConfig: type[ModeloptBaseConfig] = create_model(
    "AutoNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={
            "nn.Conv1d": _conv_config(),
            "nn.Conv2d": _conv_config(),
            "nn.Conv3d": _conv_config(),
            "nn.ConvTranspose1d": _conv_config(),
            "nn.ConvTranspose2d": _conv_config(),
            "nn.ConvTranspose3d": _conv_config(),
            "nn.Linear": _norm_lin_config(),
            "nn.BatchNorm1d": _norm_lin_config(),
            "nn.BatchNorm2d": _norm_lin_config(),
            "nn.BatchNorm3d": _norm_lin_config(),
            "nn.SyncBatchNorm": _norm_lin_config(),
            "nn.InstanceNorm1d": _norm_lin_config(),
            "nn.InstanceNorm2d": _norm_lin_config(),
            "nn.InstanceNorm3d": _norm_lin_config(),
            "nn.LayerNorm": _norm_lin_config(),
            "nn.GroupNorm": {k: v for k, v in _conv_config().items() if k != "kernel_size"},
            "nn.Sequential": {"min_depth": 0},
        },
        doc='Configuration for the ``"autonas"`` mode.',
    ),
)


class ExportConfig(ModeloptBaseConfig):
    """Configuration for the export mode.

    This mode is used to export a model after NAS search.
    """

    strict: bool = ModeloptField(
        default=True,
        title="Strict export",
        description="Enforces that the subnet configuration must exactly match during export.",
    )

    calib: bool = ModeloptField(
        default=False,
        title="Calibration",
        description="Whether to calibrate the subnet before exporting.",
    )
