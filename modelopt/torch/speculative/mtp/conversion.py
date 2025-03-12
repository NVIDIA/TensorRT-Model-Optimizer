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

"""MTP conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import MTPConfig

MTPDMRegistry = _DMRegistryCls(prefix="MTP")  # global instance for the registry


def convert_to_mtp_model(model: nn.Module, config: MTPConfig) -> ConvertReturnType:
    """Convert the model to a mtp model as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in MTPDMRegistry:
        for cls in MTPDMRegistry._registry:
            if issubclass(original_cls, cls):
                MTPDMRegistry.register({original_cls: "base_model_class"})(MTPDMRegistry[cls])
                break

    mtp_model = MTPDMRegistry.convert(model)
    mtp_model.modify(
        mtp_num_layers=config.mtp_num_layers,
        mtp_num_module=config.mtp_num_module,
        mtp_freeze_list=config.mtp_freeze_list,
        use_last_layernorm=config.use_last_layernorm,
    )

    # no metadata, all specifed via config.
    metadata = {}

    return mtp_model, metadata


def restore_mtp_model(model: nn.Module, config: MTPConfig, metadata: MetadataDict) -> nn.Module:
    """Function for restoring a previously convert model to a mtp model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_mtp_model(model, config)[0]
