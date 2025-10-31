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
Main script for running the compress algorithm on large language models (based on Puzzle paper https://arxiv.org/abs/2411.19146).

This script provides two modes:
1. Default mode: Runs the full compression pipeline
2. MIP-only mode: Runs only the MIP search and realize models phase

Usage:
    # Full compression pipeline
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml

    # Only MIP search and realize models phase
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only
"""

import argparse
from pathlib import Path

from puzzle_tools.hydra_utils import register_hydra_resolvers

from tests.utils.test_utils import initialize_hydra_config_for_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress large language models using the Compress algorithm (based on Puzzle paper https://arxiv.org/abs/2411.19146)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the main config YAML file (e.g., ./configs/llama_3.2_1B_pruneffn_memory.yaml)",
    )
    parser.add_argument(
        "--mip-only",
        action="store_true",
        help="Run only the MIP search and realize models phase (skip pruning and NAS scoring)",
    )

    return parser.parse_args()


def run_full_compress(hydra_config_path: str):
    """Run the full compression pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )


def run_mip_only(hydra_config_path: str):
    """Run only the MIP search and realize models phase.

    This assumes that pruning, replacement library building, NAS scoring, and subblock stats calculation
    have already been completed.

    Args:
        config_path: Path to the YAML configuration file
    """
    hydra_config_path = Path(hydra_config_path).resolve()
    # config_dir = str(hydra_config_path.parent)
    # config_name = hydra_config_path.stem


def main():
    args = parse_args()

    if args.mip_only:
        run_mip_only(hydra_config_path=args.config)
    else:
        run_full_compress(hydra_config_path=args.config)


if __name__ == "__main__":
    main()
