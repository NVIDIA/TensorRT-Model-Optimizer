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

"""Default configurations for prune modes."""

from pydantic import create_model

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, get_kwargs_for_create_model_with_rules


def _conv_config():
    return {
        "channels_ratio": tuple(0.05 * i for i in range(1, 21)),
        "kernel_size": (),
        "channel_divisor": 32,
    }


def _norm_lin_config():
    return {
        "features_ratio": tuple(0.05 * i for i in range(1, 21)),
        "feature_divisor": 32,
    }


def _get_fastnas_default_rules():
    return {
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
    }


FastNASConfig: type[ModeloptBaseConfig] = create_model(
    "FastNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"fastnas"`` mode.',
    ),
)


GradNASConfig: type[ModeloptBaseConfig] = create_model(
    "GradNASConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules=_get_fastnas_default_rules(),
        doc='Configuration for the ``"gradnas"`` mode.',
    ),
)

MCoreGPTMinitronConfig: type[ModeloptBaseConfig] = create_model(
    "MCoreGPTMinitronConfig",
    **get_kwargs_for_create_model_with_rules(
        registry=DMRegistry,
        default_rules={},  # Dynamically generated rules if Megatron-core is available
        doc='Configuration for the ``"mcore_gpt_minitron"`` mode.',
    ),
)
