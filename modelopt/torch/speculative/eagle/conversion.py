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

"""Eagle conversion/restore utilities."""

from torch import nn

from modelopt.torch.opt.conversion import ModelLikeModule
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from ..config import EagleConfig

EagleDMRegistry = _DMRegistryCls(prefix="Eagle")  # global instance for the registry


def convert_to_eagle_model(model: nn.Module, config: EagleConfig) -> ConvertReturnType:
    """Convert the model to a eagle model as per `config`."""
    # initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    original_cls = type(model)
    if original_cls not in EagleDMRegistry:
        for cls in EagleDMRegistry._registry:
            if issubclass(original_cls, cls):
                EagleDMRegistry.register({original_cls: "base_model_class"})(EagleDMRegistry[cls])
                break

    eagle_model = EagleDMRegistry.convert(model)
    eagle_model.modify(
        eagle_num_layers=config.eagle_num_layers,
        use_input_layernorm_in_first_layer=config.use_input_layernorm_in_first_layer,
        use_last_layernorm=config.use_last_layernorm,
        eagle_hidden_state_distillation=config.eagle_hidden_state_distillation,
        use_aux_hidden_state=config.use_aux_hidden_state,
        eagle_aux_hidden_state_layer_ids=config.eagle_aux_hidden_state_layer_ids,
        eagle_disable_moe=config.eagle_disable_moe,
        draft_vocab_size=config.draft_vocab_size,
        use_mtp_layernorm=config.use_mtp_layernorm,
        ffn_hidden_size=config.ffn_hidden_size,
    )

    # no metadata, all specifed via config.
    metadata = {}

    return eagle_model, metadata


def restore_eagle_model(model: nn.Module, config: EagleConfig, metadata: MetadataDict) -> nn.Module:
    """Function for restoring a previously convert model to a eagle model."""
    # the metadata should be empty
    assert not metadata, "No metadata expected!"

    return convert_to_eagle_model(model, config)[0]
