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

"""
Compress NAS plugin for the Modelopt framework (based on Puzzle algorithm: https://arxiv.org/abs/2411.19146).

It is used by mtn.convert() to convert a model from HF format to DeciLM format + do pruning scoring
and save pruned checkpoints, and by mtn.search() to perform the MIP-based NAS search.
"""

import datetime
from pathlib import Path

import build_library_and_stats
import mip_and_realize_models
import pruning_ckpts
import scoring
import torch
from torch import nn

from modelopt.torch._compress.activation_scoring import score_pruning_activations
from modelopt.torch._compress.decilm.converters.convert_llama3_to_decilm import (
    convert_llama3_to_decilm,
)
from modelopt.torch._compress.tools.hydra_utils import initialize_hydra_config_for_dir
from modelopt.torch._compress.tools.logger import mprint
from modelopt.torch._compress.tools.runtime import NativeDdpRuntime
from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict


class CompressModel(nn.Module):
    pass  # No model implementation is needed for the compress mode


class CompressConfig(ModeloptBaseConfig):
    """Configuration for Compress NAS algorithm."""

    # Input model path to compress in the HF format
    input_model_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config directory containing the search space definition
    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config name containing the search space definition
    hydra_config_name: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Directory to save the compressed model and intermediate results
    puzzle_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Dataset path to use for scoring in prunining and NAS search
    dataset_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )


def convert_compress_model(model: nn.Module, config: CompressConfig) -> ConvertReturnType:
    """1. Convert the model from HF format to DeciLM format.
    2. Score the pruning activations.
    3. Prune the model and save pruned checkpoints

    The output of this step will be used by mnt.search() to perform the NAS search.
    """

    # NativeDdpRuntime must be initialized/closed from outside of this function, so we are
    # NOT calling runtime.cleanup() here. TODO: Not optimal - redesign it.
    runtime = NativeDdpRuntime(
        dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
    )

    # Required for mtn.search() to read NAS configuration
    model.hydra_config_dir = config.hydra_config_dir
    model.hydra_config_name = config.hydra_config_name
    model.puzzle_dir = config.puzzle_dir
    model.dataset_path = config.dataset_path

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
    if runtime.global_rank == 0:
        mprint("Compress Progress 2/8: converting model from HF to DeciLM (single-gpu)")
        hf_ckpt_teacher_dir = "ckpts/teacher"  # TODO: make it configurable
        convert_llama3_to_decilm(
            input_dir=config.input_model_path,
            output_dir=Path(config.puzzle_dir) / hf_ckpt_teacher_dir,
        )
    runtime.wait_for_everyone()

    # Score_pruning_activations (distributed processing)
    mprint("Compress Progress 3/8: scoring pruning activations (multi-gpu)")
    score_pruning_activations.launch_score_activations(hydra_cfg, runtime)

    # Prune the model and save pruned checkpoints
    if runtime.global_rank == 0:
        mprint(
            "Compress Progress 4/8: pruning the model and saving pruned checkpoints (single-gpu)"
        )
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    runtime.wait_for_everyone()

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

        return CompressSearcher

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
        """The mode that corresponds to the export mode.
        For now, this will be a no-op as there is no modelopt's concept of search space defined
        for the compress algorithm.
        """
        return "export_nas"


class CompressSearcher(BaseSearcher):
    """Runs NAS search for the Compress mode."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Not needed for the compress mode as we are not saving any model state"""
        return {}

    def run_search(self) -> None:
        # NativeDdpRuntime must be initialized/closed from outside of this function, so we are
        # NOT calling runtime.cleanup() here. TODO: Not optimal - redesign it.
        runtime = NativeDdpRuntime(
            dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10)
        )

        # Load hydra config
        hydra_cfg = initialize_hydra_config_for_dir(
            config_dir=self.model.hydra_config_dir,
            config_name=self.model.hydra_config_name,
            overrides=[
                f"puzzle_dir={self.model.puzzle_dir}",
                f"dataset_path={self.model.dataset_path}",
            ],
        )

        # Build_library_and_stats (single process)
        if runtime.global_rank == 0:
            mprint(
                "Compress Progress 5/8: building replacement library and subblock statistics (single-gpu)"
            )
            build_library_and_stats.launch_build_library_and_stats(hydra_cfg)
        runtime.wait_for_everyone()

        # Calc_one_block_scores (distributed processing)
        mprint("Compress Progress 6/8: calculating one block scores (multi-gpu)")
        scoring.launch_scoring(hydra_cfg, runtime)

        # mip_and_realize_models (distributed processing)
        mprint("Compress Progress 7/8: running MIP and realizing models (multi-gpu)")
        mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg, runtime)
