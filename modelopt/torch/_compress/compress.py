"""

This module provides the main compression function for a model
using MIP-based NAS search algorithm.

"""

import build_library_and_stats
import mip_and_realize_models
import pruning_ckpts
import score_pruning_activations
import scoring
from omegaconf import DictConfig
from puzzle_tools.runtime import IRuntime

# TODO Move initialize_hydra_config_for_dir from tests to main
from tests.utils.test_utils import initialize_hydra_config_for_dir


def compress(
    hydra_config_dir: str, hydra_config: str, puzzle_dir: str, dataset_path: str, runtime: IRuntime
) -> DictConfig:
    """Compress a puzzletron model using the MIP-based NAS search algorithm.

    Args:
        hydra_config_dir (str): path to a hydra_config_dir that defines the search space
        hydra_config (str): the corresponding hydra config file
        puzzle_dir (str): directory with a puzzletron model to compress
        dataset_path (str): dataset used for scoring and distillation
        runtime: distributed runtime to use to run the compression steps, e.g.,
                 NativeDDP_Runtime(dtype=torch.bfloat16, torch_distributed_timeout=datetime.timedelta(10))

    Returns:
        Hydra config object after compressing the model.
        The same hydra configuration object is used across all compression steps.
        @TODO: Investigate if this config object is immutable across steps and clarify
    """
    # Step 0: Load puzzletron hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config,
        overrides=[
            f"puzzle_dir={puzzle_dir}",
            f"dataset_path={dataset_path}",
        ],
    )

    # Step 1: score_pruning_activations (distributed processing)
    score_pruning_activations.launch_score_activations(hydra_cfg, runtime)

    # Step 2: pruning_ckpts (single process)
    if runtime.global_rank == 0:
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    runtime.wait_for_everyone()

    # # Step 3: bypass distillation (distributed processing)
    # # TODO: Add bypass distillation step
    # #run_bypassed_training(hydra_cfg, runtime)

    # Step 4: build_library_and_stats (single process)
    if runtime.global_rank == 0:
        build_library_and_stats.launch_build_library_and_stats(hydra_cfg)
    runtime.wait_for_everyone()

    # Step 5: calc_one_block_scores (distributed processing)
    scoring.launch_scoring(hydra_cfg, runtime)

    # Step 6: mip_and_realize_models (distributed processing)
    mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg, runtime)

    return hydra_cfg
