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

from pathlib import Path

from scripts.convert_llama3_to_decilm import convert_llama3_to_decilm
from torch import nn

from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher

# TODO Move initialize_hydra_config_for_dir from tests to main
from tests.utils.test_utils import initialize_hydra_config_for_dir


class CompressModel(nn.Module):
    pass


class CompressConfig(ModeloptBaseConfig):
    """Configuration for Compress NAS algorithm."""

    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    puzzle_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    dataset_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )


# TOD: Why is it called SuperNetMLP?
class SuperNetMLP(CompressModel):
    """Marker subclass indicating converted/search-space state for CompressConfig.
    TODO: Provide better description
    """

    hydra_config_dir: str
    puzzle_dir: str
    dataset_path: str


def convert_compress_model(model: nn.Module, config: CompressConfig) -> ConvertReturnType:
    """Convert the model to a search space model."""
    print("=" * 80)
    print(f"[convert] before convert:\n{model}")
    model.__class__ = SuperNetMLP
    model.hydra_config_dir = config.hydra_config_dir
    model.puzzle_dir = config.puzzle_dir
    model.dataset_path = config.dataset_path

    # Load hydra config
    initialize_hydra_config_for_dir(
        config_dir=config.hydra_config_dir,
        config_name="Llama-3_1-8B",  # TODO: Make it configurable
        overrides=[
            f"puzzle_dir={config.puzzle_dir}",
            f"dataset_path={config.dataset_path}",
        ],
    )

    # Convert Llama3 model to DeciLM model
    hf_ckpt_teacher_dir = "ckpts/teacher"
    convert_llama3_to_decilm(
        input_dir=Path(config.puzzle_dir) / hf_ckpt_teacher_dir,  # TODO this should be configurable
        output_dir=Path(config.puzzle_dir) / hf_ckpt_teacher_dir,
    )

    print(f"[convert] after convert:\n{model}")
    return model, {}


def restore_compress_model(
    model: nn.Module, config: CompressConfig, metadata: MetadataDict
) -> nn.Module:
    """Reuse convert to produce the same behavior on restore."""
    return convert_compress_model(model, config)[0]


@NASModeRegistry.register_mode
class CompressDescriptor(ModeDescriptor):
    """Descriptor for the Compress mode."""

    @property
    def name(self) -> str:
        """String identifier for this mode."""
        return "compress"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Configuration class for this mode."""
        return CompressConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Return the associated searcher implementation."""
        raise NotImplementedError("Compress mode does not have a search algorithm.")

    @property
    def convert(self) -> ConvertEntrypoint:
        """Entrypoint to convert a model."""
        return convert_compress_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """Entrypoint to restore a model."""
        return restore_compress_model

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export"
