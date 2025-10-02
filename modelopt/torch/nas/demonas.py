# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Bare-bones DemoNAS template mode and searcher.

This file demonstrates how to add a new NAS algorithm to ModelOpt with the minimal
plumbing needed to integrate with the `mtn.convert` and `mtn.search` APIs.
"""

import torch
import torch.nn as nn

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


class DemoNASConfig(ModeloptBaseConfig):
    """Configuration for DemoNAS Search Space Generation."""

    mlp_inner_choices: list[int] = ModeloptField(
        default=[],
        title="MLP inner choices",
        description="List of absolute mlp_inner values to consider in search space.",
    )


class MLP(nn.Module):
    """A simple MLP with two Linear layers (hidden_size, inner) and (inner, hidden_size).

    Exposes attribute `mlp_inner` representing the inner dimension.
    """

    def __init__(self, mlp_inner: int = 40, hidden_size: int = 10):
        """Initialize the MLP with given inner width and hidden size."""
        super().__init__()
        self.mlp_inner = mlp_inner
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, mlp_inner, bias=False)
        self.fc2 = nn.Linear(mlp_inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class SuperNetMLP(MLP):
    """Marker subclass indicating converted/search-space state for DemoNAS."""

    mlp_inner_choices: list[int]


def convert_demonas(model: nn.Module, config: DemoNASConfig) -> ConvertReturnType:
    """Convert the model to a search space model."""
    print("=" * 80)
    print(f"[convert] before convert:\n{model}")
    # NOTE: Class overwrite for demonstration that convert can change the model to something else like a supernet
    model.__class__ = SuperNetMLP
    model.mlp_inner_choices = config.mlp_inner_choices
    print(f"[convert] after convert:\n{model}")
    return model, {}


def restore_demonas(model: nn.Module, config: DemoNASConfig, metadata: MetadataDict) -> nn.Module:
    """Reuse convert to produce the same behavior on restore."""
    return convert_demonas(model, config)[0]


class DemoNASSearcher(BaseSearcher):
    """A no-op searcher that accepts constraints and leaves the model unchanged."""

    computed_1block_scores: dict[int, float]

    @property
    def default_state_dict(self) -> SearchStateDict:
        """State to be saved to a checkpoint for resuming search."""
        return {"computed_1block_scores": {}}

    def run_search(self) -> None:
        """Pick an inner width under params constraint and trim model's Linear layers.

        Assumes model is an instance of MLP with attributes `mlp_inner`, `fc1`, and `fc2`.
        """
        print("=" * 80)
        print(f"[search] start:\n{self.model}")
        assert isinstance(self.model, SuperNetMLP), "DemoNAS searcher expects a SuperNetMLP model"
        choices = self.model.mlp_inner_choices
        print(f"[search] mlp_inner_choices={choices}")

        l1, l2 = self.model.fc1, self.model.fc2

        original_inner = self.model.mlp_inner
        original_params = 2 * self.model.hidden_size * original_inner
        params_limit: int = self.constraints["params"]
        print(f"[search] {original_inner=}, {original_params=}, {params_limit=}")

        # Simple scores per choice as a proxy of validation score: larger inner -> higher score
        self.computed_1block_scores = {c: float(c) for c in choices}
        print(f"[search] scores={self.computed_1block_scores}")
        valid_inners = (
            [k for k in choices if 2 * self.model.hidden_size * k <= params_limit]
            if params_limit is not None
            else choices
        )
        print(f"[search] valid_inners={valid_inners}")
        # Select inner with highest score but under constraint
        new_inner = max(valid_inners, key=lambda k: self.computed_1block_scores[int(k)])
        print(f"[search] selected new_inner after search = {new_inner}")

        # Create trimmed Linear layers and copy sliced weights
        new_l1 = nn.Linear(l1.in_features, new_inner, bias=False)
        new_l1.weight.data.copy_(l1.weight.data[:new_inner, :])

        new_l2 = nn.Linear(new_inner, l2.out_features, bias=False)
        new_l2.weight.data.copy_(l2.weight.data[:, :new_inner])

        # Assign new layers back and update attribute
        self.model.fc1 = new_l1
        self.model.fc2 = new_l2
        self.model.mlp_inner = new_inner
        print(
            f"[search] trimmed: fc1.out_features={self.model.fc1.out_features}, "
            f"fc2.in_features={self.model.fc2.in_features}, mlp_inner={self.model.mlp_inner}"
        )

        # Restore class name to MLP after search
        # NOTE: Actual logic may be different, but this is just a demonstration that
        # the converted model is now exported to the original model type.
        if isinstance(self.model, SuperNetMLP):
            self.model.__class__ = MLP
            print(f"[search] after search:\n{self.model}")


@NASModeRegistry.register_mode
class DemoNASModeDescriptor(ModeDescriptor):
    """Descriptor for the "demonas" mode."""

    @property
    def name(self) -> str:
        """String identifier for this mode."""
        return "demonas"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Configuration class for this mode."""
        return DemoNASConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Return the associated searcher implementation."""
        return DemoNASSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """Entrypoint to convert a model."""
        return convert_demonas

    @property
    def restore(self) -> RestoreEntrypoint:
        """Entrypoint to restore a model."""
        return restore_demonas

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export_nas"
