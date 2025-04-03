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

"""Module implementing and describing modes that can be used during the NAS convert process.

Check out :meth:`mtp.prune <modelopt.torch.prune.pruning.prune>` to learn more about modes.
"""

from typing import Optional

from modelopt.torch.nas.mode import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeDescriptor,
    _ModeRegistryCls,
)
from modelopt.torch.opt.searcher import BaseSearcher

from .config import FastNASConfig, GradNASConfig, MCoreGPTMinitronConfig
from .fastnas import (
    BinarySearcher,
    convert_fastnas_searchspace,
    restore_fastnas_searchspace,
    update_fastnas_metadata,
)
from .gradnas import GradientBinarySearcher
from .mcore_gpt_minitron import MCoreGPTMinitronSearcher

PruneModeRegistry = _ModeRegistryCls("prune")


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class FastNASModeDescriptor(_ModeDescriptor):
    """Class to describe the ``"fastnas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "fastnas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return FastNASConfig

    @property
    def next_modes(self) -> Optional[set[str]]:
        """Modes that must immediately follow this mode."""
        return {"export", "kd_loss", "quantize", "sparse_magnitude", "sparse_gpt"}

    @property
    def export_mode(self) -> Optional[str]:
        """The mode that corresponds to the export mode of this mode."""
        return "export"

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        return BinarySearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_fastnas_searchspace

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_fastnas_searchspace

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_fastnas_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_fastnas_metadata


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class GradNASModeDescriptor(FastNASModeDescriptor):
    """Class to describe the ``"gradnas"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "gradnas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return GradNASConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        return GradientBinarySearcher


@NASModeRegistry.register_mode
@PruneModeRegistry.register_mode
class MCoreGPTMinitronModeDescriptor(FastNASModeDescriptor):
    """Class to describe the ``"mcore_gpt_minitron"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "mcore_gpt_minitron"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return MCoreGPTMinitronConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Specifies the search algorithm to use for this mode (if any)."""
        return MCoreGPTMinitronSearcher
