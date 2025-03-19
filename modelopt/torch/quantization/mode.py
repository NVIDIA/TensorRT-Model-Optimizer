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

"""This module contains the mode descriptor for the quantization mode."""

from typing import Optional

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeDescriptor,
    _ModeRegistryCls,
)

from .config import QuantizeConfig, _QuantizeExportConfig
from .conversion import (
    convert_to_quantized_model,
    export_quantized_model,
    restore_export_quantized_model,
    restore_quantized_model,
    update_quantize_metadata,
)

QuantizeModeRegistry = _ModeRegistryCls("quantization")


# TODO: OMNIML-717 Reuse search infra for quantization calibration algorithms
@QuantizeModeRegistry.register_mode
class QuantizeModeDescriptor(_ModeDescriptor):
    """Class to describe the ``"quant"`` mode.

    The properties of this mode can be inspected via the source code.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "quantize"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return QuantizeConfig

    @property
    def next_modes(self) -> Optional[set[str]]:
        """Modes that must immediately follow this mode."""
        return {"kd_loss"}

    @property
    def export_mode(self) -> Optional[str]:
        """The mode that corresponds to the export mode of this mode."""
        return "export_quantize"

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return convert_to_quantized_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_quantized_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before saving."""
        return update_quantize_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_quantize_metadata


@QuantizeModeRegistry.register_mode
class QuantizeExportModeDescriptor(_ModeDescriptor):
    """Class to describe the export of quantization mode.

    Note that this mode is just a placeholder to throw an error since we don't support exporting
    quantized models right now. It is used to properly indicate that the ``quantize`` mode does
    require an export mode if we ever wanted to do chaining/stacking of modes with it.
    """

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "quantize_export"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return _QuantizeExportConfig

    @property
    def is_export_mode(self) -> bool:
        """Specifies whether the mode is an export mode."""
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return export_quantized_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_export_quantized_model
