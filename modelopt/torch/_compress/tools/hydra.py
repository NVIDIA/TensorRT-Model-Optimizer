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

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

"""
Utilities for hydra config initialization.
"""


def initialize_hydra_config_for_dir(
    config_dir: str, config_name: str, overrides: list[str]
) -> DictConfig:
    """Initialize a hydra config from an absolute path for a config directory

    Args:
        config_dir (str):
        config_name (str):
        overrides (List[str]):

    Returns:
        DictConfig:
    """

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        args = compose(config_name, overrides)
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args


def initialize_hydra_config(config_path: str, config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        args = compose(config_name, overrides)
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args
