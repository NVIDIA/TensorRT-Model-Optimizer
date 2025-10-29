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

import datetime
from pathlib import Path

import pruning_ckpts
import score_pruning_activations
import torch
from scripts.convert_llama3_to_decilm import convert_llama3_to_decilm
from torch import nn

from modelopt.torch._compress.runtime import NativeDdpRuntime
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

    input_model_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    hydra_config_name: str = ModeloptField(
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


def convert_compress_model(model: nn.Module, config: CompressConfig) -> ConvertReturnType:
    """Convert the model to a search space model."""
    print("=" * 80)
    print(f"[convert] before convert:\n{model}")

    runtime = NativeDdpRuntime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    )

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=config.hydra_config_dir,
        config_name=config.hydra_config_name,
        overrides=[
            f"puzzle_dir={config.puzzle_dir}",
            f"dataset_path={config.dataset_path}",
        ],
    )

    # Convert Llama3 model to DeciLM model
    hf_ckpt_teacher_dir = "ckpts/teacher"
    convert_llama3_to_decilm(
        input_dir=config.input_model_path,
        output_dir=Path(config.puzzle_dir) / hf_ckpt_teacher_dir,
    )

    #  Score_pruning_activations (distributed processing)
    score_pruning_activations.launch_score_activations(hydra_cfg, runtime)

    if runtime.global_rank == 0:
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    runtime.wait_for_everyone()

    print(f"[convert] after convert:\n{model}")
    return model, {}


def restore_compress_model(
    model: nn.Module, config: CompressConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore is not needed for the compress mode as we are not saving any model state"""
    return model


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
