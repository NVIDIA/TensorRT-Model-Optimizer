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
# mypy: ignore-errors

"""Provides hooks for capturing the inputs and the outputs of pytorch modules that are used for
activation scoring for pruning.
"""

import argparse
import gc
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as safe_load_file
from torch import nn

# BlockConfig used at runtime, not just type hints (lines 680, 790)
from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import BlockConfig  # noqa: TC001
from modelopt.torch._compress.decilm.deci_lm_hf_code.configuration_decilm import (
    DeciLMConfig,  # noqa: TC001
)
from modelopt.torch._compress.decilm.deci_lm_hf_code.modeling_decilm import (
    DeciLMDecoderLayer,
    DeciLMRMSNorm,
)
from modelopt.torch._compress.tools.logger import aprint
from modelopt.torch._compress.tools.robust_json import json_dump
from modelopt.torch._compress.tools.runtime import IRuntime


def clear_gpu_memory(clear: bool) -> None:
    if clear:
        gc.collect()
        torch.cuda.empty_cache()


class ActivationsHook(ABC):
    @abstractmethod
    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        A hook to be registered in pytorch modules: torch.nn.Module.register_forward_hook()

        Args:
            module (nn.Module):
            args (tuple[torch.Tensor]): Input of the pytorch module
            output (torch.Tensor): Output of the pytorch module
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, torch.Tensor]: ...

    def save_state(self) -> dict:
        """
        Save the internal state of the hook for checkpointing.

        Returns:
            dict: State dictionary that can be used to restore the hook's state
        """
        # Default implementation - hooks should override this if they have state to save
        return {}

    def load_state(self, state_dict: dict) -> None:
        """
        Load the internal state of the hook from a checkpoint.

        Args:
            state_dict: State dictionary previously returned by save_state()
        """
        # Default implementation - hooks should override this if they have state to load

    def get_progress_info(self) -> dict:
        """
        Get progress information for this hook (e.g., current iteration, samples processed).

        Returns:
            dict: Progress information
        """
        # Default implementation - hooks can override to provide progress info
        return {}

    @classmethod
    def dump_activations_logs(
        cls: type["ActivationsHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
        args: argparse.Namespace,
        runtime: IRuntime | None,
    ):
        """
        Default implementation for dumping final activation scores logs to disk.
        This is called only at the end of scoring to save final results.
        """

        activations_log_dir = Path(activations_log_dir)
        activations_log_dir.mkdir(exist_ok=True, parents=True)
        rank = runtime.global_rank if runtime is not None else 0
        activations_log_path = activations_log_dir / f"rank_{rank}.pth"
        activations_log = {
            module_name: hook.to_dict() for module_name, hook in activation_hooks.items()
        }
        torch.save(activations_log, activations_log_path)

        if rank == 0:
            args.activation_hooks_kwargs.pop("model")
            json_dump(
                OmegaConf.to_container(args, resolve=True)
                if isinstance(args, DictConfig)
                else vars(args),
                activations_log_dir / "args.json",
            )
        if runtime is not None:
            runtime.wait_for_everyone()  # rank 0 will not wait before dumping args.json

        aprint(f"Dumped final activations log to {activations_log_path}")

    @classmethod
    def save_hook_states(
        cls: type["ActivationsHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
        runtime: IRuntime | None,
    ):
        """
        Save hook states for checkpointing (separate from final results).
        This can be called periodically during scoring.
        Note: Synchronization should be handled at a higher level to avoid deadlocks.
        """
        activations_log_dir = Path(activations_log_dir)
        activations_log_dir.mkdir(exist_ok=True, parents=True)
        rank = runtime.global_rank if runtime is not None else 0

        hook_states_path = activations_log_dir / f"hook_states_rank_{rank}.pth"
        hook_states = {
            module_name: hook.save_state() for module_name, hook in activation_hooks.items()
        }
        torch.save(hook_states, hook_states_path)

        return hook_states_path


class MinitronHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        weight_matrix = linear_layer.weight.float()
        num_channels = linear_layer.in_features
        self.agg_channel_activations = torch.zeros(
            size=(num_channels,), dtype=torch.float32, device=weight_matrix.device
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        mean_abs_channel_activations = activations.float().mean(
            dim=list(range(activations.ndim - 1))
        )
        self.agg_channel_activations[:] += mean_abs_channel_activations  # shape [I]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_channel_activations.cpu(),
        }


class MlpHook(ActivationsHook):
    def __init__(self, mlp: nn.Module, activation_hooks_kwargs: dict):
        features = ["l2_norm", "max_norm", "ffn_to_skip_ratio"]
        self.features = {f: [] for f in features}

        self.num_batches = 0

    def __call__(
        self, module: nn.Module, mlp_input: tuple[torch.Tensor], mlp_out: torch.Tensor
    ) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        # print instance and shape of input and output
        print(f"MLP input shape: {mlp_input[0].shape}, output shape: {mlp_out.shape}")
        print(f"MLP input type: {type(mlp_input)=}, {type(mlp_input[0])=}")
        # return

        mlp_input = mlp_input[1]  # unnormed input
        l2_norm_skip = torch.linalg.vector_norm(mlp_input, dim=-1)  # shape should be (B, T)
        l2_norm_ffn_out = torch.linalg.vector_norm(mlp_out, dim=-1)  # shape should be (B, T)
        max_norm = torch.linalg.vector_norm(
            mlp_out, ord=float("inf"), dim=-1
        )  # shape should be (B, T)

        ffn_ratio = l2_norm_ffn_out / (l2_norm_skip + l2_norm_ffn_out)
        # difference_norm = torch.linalg.vector_norm(mlp_out - mlp_input, dim=-1)

        self.features["l2_norm"].append(l2_norm_ffn_out.cpu())
        self.features["max_norm"].append(max_norm.cpu())
        self.features["ffn_to_skip_ratio"].append(ffn_ratio.cpu())

        self.num_batches += 1

        # calculate_moe_ness

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            # f"{k}_score": v.cpu() / self.num_batches for k, v in self.features.items()
            f"{k}_score": torch.cat(v)
            for k, v in self.features.items()
        }


class BlockHook(ActivationsHook):
    def __init__(self, block: DeciLMDecoderLayer, activation_hooks_kwargs: dict):
        features = ["l2_norm", "max_norm", "ffn_to_skip_ratio"]
        self.features = {f: [] for f in features}

        self.num_batches = 0

    def __call__(
        self,
        module: nn.Module,
        block_input: tuple[torch.Tensor],
        block_out: torch.Tensor,
    ) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        # # print instance and shape of input and output
        print(f"Block input shape: {block_input[0].shape}, output shape: {block_out.shape}")
        print(f"Block input type: {type(block_input)=}, {type(block_input[0])=}")
        print(f"Block output type: {type(block_out)=}")
        print(f"{type(module)=}")
        print(f"{block_input[0].device=}")

        block_input = block_input[0]  # unnormed input
        block_out_before_skip = block_out - block_input

        l2_norm_input = torch.linalg.vector_norm(block_input, dim=-1)  # shape should be (B, T)
        l2_norm = torch.linalg.vector_norm(block_out_before_skip, dim=-1)  # shape should be (B, T)
        max_norm = torch.linalg.vector_norm(
            block_out_before_skip, ord=float("inf"), dim=-1
        )  # shape should be (B, T)

        diff_ratio = l2_norm / (l2_norm_input + l2_norm)
        # difference_norm = torch.linalg.vector_norm(mlp_out - mlp_input, dim=-1)

        self.features["l2_norm"].append(l2_norm.cpu())
        self.features["max_norm"].append(max_norm.cpu())
        self.features["ffn_to_skip_ratio"].append(diff_ratio.cpu())

        self.num_batches += 1

        # calculate_moe_ness

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            # f"{k}_score": v.cpu() / self.num_batches for k, v in self.features.items()
            f"{k}_score": torch.cat(v)
            for k, v in self.features.items()
        }


class IOCorrelationBlockHook(ActivationsHook):
    def __init__(self, block: DeciLMDecoderLayer, activation_hooks_kwargs: dict):
        layer_input_descriptors_path = activation_hooks_kwargs.get("layer_input_descriptors_path")
        assert layer_input_descriptors_path is not None, (
            "layer_input_descriptors_path must be provided"
        )
        assert Path(layer_input_descriptors_path).exists(), (
            f"layer_input_descriptors_path {layer_input_descriptors_path} does not exist"
        )

        self.layer_input_descriptors = safe_load_file(layer_input_descriptors_path)[
            "layer_descriptors"
        ]

        features = [
            "spearman_corr",
            "cosine_dist",
            "pearson_corr",
            "data_dep_in_spearman_corr",
            "data_dep_in_cosine_dist",
            "data_dep_in_pearson_corr",
            "data_dep_out_spearman_corr",
            "data_dep_out_cosine_dist",
            "data_dep_out_pearson_corr",
        ]
        self.features = {f: [] for f in features}

        self.num_batches = 0

    @staticmethod
    def calculate_metrics(block_delta, layer_input_descriptors):
        # Ensure tensors are on the same device
        device = block_delta.device
        layer_input_descriptors = layer_input_descriptors.to(device)

        B, T, E = block_delta.shape  # noqa: N806
        L, _ = layer_input_descriptors.shape  # noqa: N806

        # Normalize for cosine similarity and Pearson correlation in advance
        block_delta_norm = (block_delta - block_delta.mean(dim=-1, keepdim=True)) / block_delta.std(
            dim=-1, keepdim=True
        )
        layer_input_descriptors_norm = (
            layer_input_descriptors - layer_input_descriptors.mean(dim=-1, keepdim=True)
        ) / layer_input_descriptors.std(dim=-1, keepdim=True)

        # Precompute cosine similarity (equivalent to Pearson correlation for standardized vectors)
        pearson_results = (
            torch.einsum("bte,le->btl", block_delta_norm, layer_input_descriptors_norm) / E
        )

        # Precompute cosine similarity for raw normalized vectors
        block_delta_cosine_norm = torch.nn.functional.normalize(block_delta, dim=-1)
        layer_input_descriptors_cosine_norm = torch.nn.functional.normalize(
            layer_input_descriptors, dim=-1
        )
        cosine_results = torch.einsum(
            "bte,le->btl", block_delta_cosine_norm, layer_input_descriptors_cosine_norm
        )

        # Compute Spearman correlation using rank-based operations
        block_delta_ranks = block_delta.argsort(dim=-1).float()
        layer_input_descriptors_ranks = layer_input_descriptors.argsort(dim=-1).float()

        block_delta_ranks = (
            block_delta_ranks - block_delta_ranks.mean(dim=-1, keepdim=True)
        ) / block_delta_ranks.std(dim=-1, keepdim=True)
        layer_input_descriptors_ranks = (
            layer_input_descriptors_ranks - layer_input_descriptors_ranks.mean(dim=-1, keepdim=True)
        ) / layer_input_descriptors_ranks.std(dim=-1, keepdim=True)

        spearman_results = (
            torch.einsum("bte,le->btl", block_delta_ranks, layer_input_descriptors_ranks) / E
        )

        return spearman_results.cpu(), pearson_results.cpu(), cosine_results.cpu()

    @staticmethod
    def calculate_metrics_data_dependent(block_delta, layer_input_descriptors, block_input):
        # Ensure tensors are on the same device
        device = block_delta.device
        layer_input_descriptors = layer_input_descriptors.to(device)
        block_input = block_input.to(device)

        B, T, E = block_delta.shape  # noqa: N806
        L, _ = layer_input_descriptors.shape  # noqa: N806

        # Compute data-dependent combination
        data_dependent_results = torch.einsum("bte,le->btle", block_input, layer_input_descriptors)

        # Normalize the data-dependent descriptors
        data_dependent_norm = (
            data_dependent_results - data_dependent_results.mean(dim=-1, keepdim=True)
        ) / data_dependent_results.std(dim=-1, keepdim=True)

        # Normalize block_delta for comparison
        block_delta_norm = (block_delta - block_delta.mean(dim=-1, keepdim=True)) / block_delta.std(
            dim=-1, keepdim=True
        )

        # Compute Pearson correlation
        pearson_results = torch.einsum("bte,btle->btl", block_delta_norm, data_dependent_norm) / E

        # Compute cosine similarity
        block_delta_cosine_norm = torch.nn.functional.normalize(block_delta, dim=-1)
        data_dependent_cosine_norm = torch.nn.functional.normalize(data_dependent_results, dim=-1)
        cosine_results = torch.einsum(
            "bte,btle->btl", block_delta_cosine_norm, data_dependent_cosine_norm
        )

        # Compute Spearman correlation using rank-based operations
        block_delta_ranks = block_delta.argsort(dim=-1).float()
        data_dependent_ranks = data_dependent_results.argsort(dim=-1).float()

        block_delta_ranks = (
            block_delta_ranks - block_delta_ranks.mean(dim=-1, keepdim=True)
        ) / block_delta_ranks.std(dim=-1, keepdim=True)
        data_dependent_ranks = (
            data_dependent_ranks - data_dependent_ranks.mean(dim=-1, keepdim=True)
        ) / data_dependent_ranks.std(dim=-1, keepdim=True)

        spearman_results = (
            torch.einsum("bte,btle->btl", block_delta_ranks, data_dependent_ranks) / E
        )

        return spearman_results.cpu(), pearson_results.cpu(), cosine_results.cpu()

    def __call__(
        self,
        module: nn.Module,
        block_input: tuple[torch.Tensor],
        block_out: torch.Tensor,
    ) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        # # print instance and shape of input and output
        print(f"Block input shape: {block_input[0].shape}, output shape: {block_out.shape}")
        print(f"Block input type: {type(block_input)=}, {type(block_input[0])=}")
        print(f"Block output type: {type(block_out)=}")
        print(f"{type(module)=}")
        print(f"{block_input[0].device=}")

        block_input = block_input[0]  # unnormed input shape B T E
        block_delta = block_out - block_input

        spearman_results, pearson_results, cosine_results = self.calculate_metrics(
            block_delta, self.layer_input_descriptors
        )

        self.features["spearman_corr"].append(spearman_results)
        self.features["cosine_dist"].append(cosine_results)
        self.features["pearson_corr"].append(pearson_results)

        # spearman_results, pearson_results, cosine_results = (
        #     self.calculate_metrics_data_dependent(
        #         block_delta, self.layer_input_descriptors, block_input
        #     )
        # )
        # self.features['data_dep_in_spearman_corr'].append(spearman_results)
        # self.features['data_dep_in_cosine_dist'].append(cosine_results)
        # self.features['data_dep_in_pearson_corr'].append(pearson_results)

        # spearman_results, pearson_results, cosine_results = (
        #     self.calculate_metrics_data_dependent(
        #         block_delta, self.layer_input_descriptors, block_out
        #     )
        # )
        # self.features['data_dep_out_spearman_corr'].append(spearman_results)
        # self.features['data_dep_out_cosine_dist'].append(cosine_results)
        # self.features['data_dep_out_pearson_corr'].append(pearson_results)

        self.num_batches += 1

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            # f"{k}_score": v.cpu() / self.num_batches for k, v in self.features.items()
            f"{k}_score": torch.cat(v)
            for k, v in self.features.items()
        }


class MinitronAbsHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        weight_matrix = linear_layer.weight.float()
        num_channels = linear_layer.in_features
        self.agg_channel_activations = torch.zeros(
            size=(num_channels,), dtype=torch.float32, device=weight_matrix.device
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        mean_abs_channel_activations = (
            activations.abs().float().mean(dim=list(range(activations.ndim - 1)))
        )
        self.agg_channel_activations[:] += mean_abs_channel_activations  # shape [I]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_channel_activations.cpu(),
        }


class IndependentChannelContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        weight_matrix = linear_layer.weight.float()
        self.weight_norm = torch.linalg.vector_norm(weight_matrix, dim=0)
        num_channels = linear_layer.in_features
        self.agg_channel_activations = torch.zeros(
            size=(num_channels,), dtype=torch.float32, device=weight_matrix.device
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        mean_abs_channel_activations = (
            activations.abs().float().mean(dim=list(range(activations.ndim - 1)))
        )
        self.agg_channel_activations[:] += mean_abs_channel_activations  # shape [I]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": (self.weight_norm * self.agg_channel_activations).cpu(),
            "weight_norm": self.weight_norm.cpu(),
            "agg_channel_activations": self.agg_channel_activations.cpu(),
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "agg_channel_activations": self.agg_channel_activations.cpu().clone(),
            "weight_norm": self.weight_norm.cpu().clone(),
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.agg_channel_activations = state_dict["agg_channel_activations"].to(
            self.agg_channel_activations.device
        )
        # weight_norm should be the same as it's derived from the model weights
        # but we can verify it matches
        expected_weight_norm = state_dict["weight_norm"].to(self.weight_norm.device)
        if not torch.allclose(self.weight_norm, expected_weight_norm, rtol=1e-5):
            print(
                "Warning: weight_norm mismatch during state loading - model weights may have changed"
            )


def get_pruning_schedule(num_channels, pruning_iters):
    """
    Spending decreases monotonically when num_channels >= pruning_iters.
    Intervals between spends increase monotonically when pruning_iters > num_channels.
    The budget is fully utilized, and there's spending in the last iteration.
    num_channels = 10, pruning_iters = 4 ==> [3, 3, 2, 2]
    num_channels = 4, pruning_iters = 10 ==> [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    """
    if num_channels >= pruning_iters:
        # Case when budget is greater than or equal to iterations
        q = num_channels // pruning_iters  # Base spend per iteration
        r = num_channels % pruning_iters  # Remainder to distribute

        schedule = []
        for i in range(pruning_iters):
            if i < r:
                # Assign higher spend to earlier iterations
                schedule.append(q + 1)
            else:
                schedule.append(q)
    else:
        # Case when iterations are greater than budget
        schedule = [0] * pruning_iters
        for i in range(1, num_channels + 1):
            # Distribute spends at positions where intervals increase monotonically
            pos = ((i * pruning_iters) // num_channels) - 1
            schedule[pos] = 1
    return schedule


class IterativeChannelContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        self.weight_matrix = linear_layer.weight
        self.num_channels = linear_layer.in_features
        self.pruning_iters = activation_hooks_kwargs["validation_full_iters"]
        self.clear_gpu_memory = activation_hooks_kwargs.get("clear_gpu_memory", False)
        self.curr_iter = 0
        self.pruning_schedule = get_pruning_schedule(
            num_channels=self.num_channels, pruning_iters=self.pruning_iters
        )

        self.agg_cont_per_channel = torch.zeros(
            size=(self.num_channels,),
            dtype=torch.float32,
            device=self.weight_matrix.device,
        )
        self.pruned_channels = []
        self.calibration_method = activation_hooks_kwargs.get("calibration_method")
        self.epsilon = 1e-8

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        activations = args[0]
        n_channels_to_prune = self.pruning_schedule[self.curr_iter]

        curr_activations = activations.clone()  # Shape B,T,I
        curr_activations[..., self.pruned_channels] = 0
        output_curr = F.linear(input=curr_activations, weight=self.weight_matrix)  # Shape B,T,E

        if self.calibration_method is None:
            scaling_factor_per_token = torch.ones_like(output[..., 0])  # Shape B,T
        elif self.calibration_method == "scale_by_magnitude":
            output_norms = torch.linalg.vector_norm(output, dim=-1)  # Shape B,T
            output_curr_norms = torch.linalg.vector_norm(output_curr, dim=-1)  # Shape B,T
            scaling_factor_per_token = output_curr_norms / (output_norms + self.epsilon)
            del output_curr_norms, output_norms
        else:
            raise NotImplementedError
        del curr_activations
        clear_gpu_memory(clear=self.clear_gpu_memory)

        s = scaling_factor_per_token.unsqueeze(-1) * output - output_curr  # Shape: (B, T, E)
        s_squared_per_token = torch.sum(s**2, dim=-1)  # Shape: (B, T)
        b = s @ self.weight_matrix  # Shape: (B, T, I)
        c = torch.sum(self.weight_matrix**2, dim=0)  # Shape: (I)
        del s, output_curr
        clear_gpu_memory(clear=self.clear_gpu_memory)

        contribution_squared = (
            s_squared_per_token.unsqueeze(2) + 2 * activations * b + (activations**2) * c
        )  # Shape: (B, T, I)
        del s_squared_per_token, b, c, activations
        clear_gpu_memory(clear=self.clear_gpu_memory)

        contribution = torch.sqrt(contribution_squared + self.epsilon)  # Shape: (B, T, I)
        mean_cont_per_channel = torch.mean(contribution, dim=(0, 1))  # Shape: (I)
        mean_cont_per_channel[self.pruned_channels] = torch.inf
        del contribution, contribution_squared
        clear_gpu_memory(clear=self.clear_gpu_memory)

        if n_channels_to_prune == 0:
            self.agg_cont_per_channel += mean_cont_per_channel
        else:
            _, worst_indices = torch.topk(mean_cont_per_channel, n_channels_to_prune, largest=False)
            worst_indices_list = worst_indices.tolist()
            assert not set(self.pruned_channels).intersection(set(worst_indices_list))
            self.pruned_channels.extend(worst_indices_list)
            self.agg_cont_per_channel.zero_()
        self.curr_iter += 1

    def to_dict(self) -> dict[str, torch.Tensor]:
        assert self.num_channels == len(self.pruned_channels)
        channels_importance_ascending = torch.tensor(self.pruned_channels, dtype=torch.long)
        score = torch.empty(self.num_channels, dtype=torch.long)
        score[channels_importance_ascending] = torch.arange(self.num_channels, dtype=torch.long)

        return {
            "score": score.cpu(),
            "channels_importance_ascending": channels_importance_ascending.cpu(),
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "curr_iter": self.curr_iter,
            "pruned_channels": self.pruned_channels.copy(),
            "agg_cont_per_channel": self.agg_cont_per_channel.cpu().clone(),
            "num_channels": self.num_channels,
            "pruning_iters": self.pruning_iters,
            "pruning_schedule": self.pruning_schedule.copy(),
            "calibration_method": self.calibration_method,
            "epsilon": self.epsilon,
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.curr_iter = state_dict["curr_iter"]
        self.pruned_channels = state_dict["pruned_channels"].copy()
        self.agg_cont_per_channel = state_dict["agg_cont_per_channel"].to(self.weight_matrix.device)
        # Verify other parameters match
        assert self.num_channels == state_dict["num_channels"], "Channel count mismatch"
        assert self.pruning_iters == state_dict["pruning_iters"], "Iteration count mismatch"
        assert self.pruning_schedule == state_dict["pruning_schedule"], "Pruning schedule mismatch"

    def get_progress_info(self) -> dict:
        """Get progress information."""
        progress = self.curr_iter / self.pruning_iters if self.pruning_iters > 0 else 0.0
        return {
            "curr_iter": self.curr_iter,
            "total_iters": self.pruning_iters,
            "progress": progress,
            "pruned_channels_count": len(self.pruned_channels),
            "total_channels": self.num_channels,
        }


class IndependentKvHeadContributionHook(ActivationsHook):
    def __init__(self, linear_layer: nn.Linear, activation_hooks_kwargs: dict):
        model_config: DeciLMConfig = activation_hooks_kwargs["model"].config
        block_config: BlockConfig = activation_hooks_kwargs["block_config"]

        self.optimize_for = activation_hooks_kwargs.get("optimize_for", "memory")
        assert self.optimize_for in ["latency", "memory"]

        self.hidden_size = model_config.hidden_size
        self.n_heads_in_group = block_config.attention.n_heads_in_group
        self.num_q_heads = model_config.num_attention_heads
        self.num_kv_heads = self.num_q_heads // self.n_heads_in_group
        self.head_dim = getattr(model_config, "head_dim", self.hidden_size // self.num_q_heads)

        self.agg_kv_head_contributions = torch.zeros(
            size=(self.num_kv_heads,),
            dtype=torch.float32,
            device=linear_layer.weight.device,
        )

        # Reshape weight matrix to group by KV heads
        self.weight_grouped = linear_layer.weight.view(
            self.hidden_size, self.num_kv_heads, self.head_dim * self.n_heads_in_group
        ).permute((1, 0, 2))
        # weight_grouped.shape: (kv_heads, hidden_dim, head_dim * n_heads_in_group)

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module: The linear projection layer
        :param args: tuple containing attention output tensor (B, T, num_q_heads * head_dim)
        :param output: The projected output (B, T, hidden_dim)
        """
        attn_out = args[0]  # Shape: (B, T, num_q_heads * head_dim)
        batch_size, seq_len, _ = attn_out.shape

        # Reshape attention output to group by KV heads
        attn_out_grouped = attn_out.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim * self.n_heads_in_group,
        ).unsqueeze(-2)
        # attn_out_grouped.shape: (B, T, kv_heads, 1, head_dim * n_heads_in_group)

        if self.optimize_for == "latency":
            # Compute contribution per KV head group
            # First compute the projection for each KV head group
            layer_out_grouped = attn_out_grouped @ self.weight_grouped.transpose(-1, -2)
            layer_out_grouped = layer_out_grouped.squeeze(-2)
            # layer_out_grouped.shape: (B, T, kv_heads, hidden_dim)

        else:
            layer_out_grouped = []
            for i in range(self.num_kv_heads):
                _layer_out = attn_out_grouped[:, :, i] @ self.weight_grouped[i].transpose(-1, -2)
                layer_out_grouped.append(_layer_out)
            layer_out_grouped = torch.cat(layer_out_grouped, dim=2)

        # Compute L2 norm of each group's contribution
        contrib_per_kv_head = torch.linalg.vector_norm(layer_out_grouped, dim=-1)
        # contrib_per_kv_head.shape: (B, T, kv_heads)

        contrib_per_kv_head = contrib_per_kv_head.mean(dim=(0, 1))
        # contrib_per_kv_head.shape: (kv_heads,)

        # Accumulate contributions
        self.agg_kv_head_contributions += contrib_per_kv_head

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_kv_head_contributions.cpu(),
        }


class RouterStatsHook(ActivationsHook):
    def __init__(self, router: nn.Linear, activation_hooks_kwargs: dict):
        self.r_stats = torch.zeros(
            size=(router.out_features,), dtype=torch.int64, device=router.weight.device
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        router_logits = output
        top_k = 1
        _, router_indices = torch.topk(router_logits, top_k, dim=-1)
        ids, counts = router_indices.unique(return_counts=True)
        self.r_stats[ids] += counts
        # for router_id in router_indices.flatten().cpu().tolist():
        #     self.r_stats[router_id] += 1

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "r_stats": self.r_stats.cpu(),
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "r_stats": self.r_stats.cpu().clone(),
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        self.r_stats = state_dict["r_stats"].to(self.r_stats.device)


class RankedChoiceVotingHook(ActivationsHook):
    def __init__(self, router: nn.Linear, activation_hooks_kwargs: dict):
        self.router_argsort: list[torch.Tensor] = []
        block_config: BlockConfig = activation_hooks_kwargs["block_config"]
        self.top_k = block_config.ffn.moe.num_experts_per_tok

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        router_logits = output
        num_experts = router_logits.shape[-1]
        router_argsort = torch.argsort(router_logits, dim=-1, descending=True)
        router_argsort = router_argsort.view(-1, num_experts).to(torch.int16).cpu()
        self.router_argsort.append(router_argsort)

    def to_dict(self) -> dict[str, torch.Tensor]:
        router_argsort = torch.concat(self.router_argsort, dim=0)
        num_tokens, num_experts = router_argsort.shape

        expert_ranks = torch.full((num_experts,), -1)
        expert_counts_at_pruning_time = {}

        expert_kept_per_iteration: list[list[int]] = []
        expert_counts_per_iteration: list[dict[int, int]] = []

        for rank in range(num_experts):
            ids, counts = router_argsort[:, : self.top_k].unique(return_counts=True)
            ids = ids.tolist()
            counts = counts.tolist()
            expert_counts = dict(zip(ids, counts))

            expert_kept_per_iteration.append(ids)
            expert_counts_per_iteration.append(expert_counts)

            least_popular_expert, min_count = min(expert_counts.items(), key=lambda tup: tup[1])

            expert_ranks[least_popular_expert] = rank
            expert_counts_at_pruning_time[least_popular_expert] = min_count

            router_argsort = router_argsort[router_argsort != least_popular_expert].view(
                num_tokens, -1
            )

        zero_shot_expert_counts = torch.zeros((num_experts,), dtype=torch.long)
        for expert_id, expert_counts in expert_counts_per_iteration[0].items():
            zero_shot_expert_counts[expert_id] = expert_counts
        zero_shot_expert_ranks = torch.argsort(torch.argsort(zero_shot_expert_counts))

        return {
            # who's more important with Ranked Choice Voting? bigger is better.
            "expert_ranks": expert_ranks,
            # who's more important with zero-shot voting? bigger is better.
            "zero_shot_expert_ranks": zero_shot_expert_ranks,
            # how many tokens chose each expert in the iteration where it was removed.
            "expert_counts_at_pruning_time": expert_counts_at_pruning_time,
            # full expert distribution per pruning iteration.
            "expert_counts_per_iteration": expert_counts_per_iteration,
            "top_k": self.top_k,  # top_k experts per token.
        }

    def save_state(self) -> dict:
        """Save the internal state for checkpointing."""
        return {
            "router_argsort": [tensor.cpu().clone() for tensor in self.router_argsort],
            "top_k": self.top_k,
        }

    def load_state(self, state_dict: dict) -> None:
        """Load the internal state from a checkpoint."""
        # Move tensors back to appropriate device (will be determined when hook is called)
        self.router_argsort = [tensor.cpu() for tensor in state_dict["router_argsort"]]
        self.top_k = state_dict["top_k"]

    def get_progress_info(self) -> dict:
        """Get progress information."""
        return {
            "num_batches_processed": len(self.router_argsort),
            "total_tokens_processed": sum(tensor.shape[0] for tensor in self.router_argsort)
            if self.router_argsort
            else 0,
        }


class RouterNumActiveExpertsStatsHook(ActivationsHook):
    def __init__(self, router: nn.Linear, activation_hooks_kwargs: dict):
        self.batch_sizes = [8, 16, 32, 64, 128, 256]
        self.r_stats = {batch_size: [] for batch_size in self.batch_sizes}

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        router_logits = output
        assert len(router_logits.shape) == 3, f"router_logits.shape: {router_logits.shape}"
        top_k = 1
        _, router_indices = torch.topk(router_logits, top_k, dim=-1)

        # shuffle router_indices on dim=1
        num_samples = 5
        rand_perm = torch.randperm(router_indices.size(1))
        router_indices_shuffled = router_indices[:, rand_perm]

        for batch_size in self.batch_sizes:
            btsz = router_indices.shape[0]
            seq_length = router_indices.shape[1]
            seq_to_take = batch_size // btsz
            # sample a random starting place (random comes from randperm)
            starting_place = torch.arange(0, seq_length - seq_to_take + 1, num_samples)

            for start in starting_place[:num_samples]:
                num_uniques = (
                    router_indices_shuffled[:, start : start + seq_to_take]
                    .flatten()
                    .unique()
                    .numel()
                )
                self.r_stats[batch_size].append(num_uniques)

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "num_active_experts": {
                batch_size: torch.tensor(self.r_stats[batch_size])
                for batch_size in self.batch_sizes
            },
        }


class RouterNumActiveExpertsStatsHookUnshuffled(RouterNumActiveExpertsStatsHook):
    def __init__(self, router: nn.Linear, activation_hooks_kwargs: dict):
        super().__init__(router, activation_hooks_kwargs)

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        router_logits = output
        assert len(router_logits.shape) == 3, f"router_logits.shape: {router_logits.shape}"
        top_k = 1
        _, router_indices = torch.topk(router_logits, top_k, dim=-1)

        for batch_size in self.batch_sizes:
            seq_to_take = batch_size
            num_uniques = router_indices[:, -seq_to_take:].flatten().unique().numel()
            self.r_stats[batch_size].append(num_uniques)


class RouterEntropyHook(ActivationsHook):
    def __init__(self, router: nn.Linear, activation_hooks_kwargs: dict):
        self.entropy = []

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        :param module:
        :param args: tuple with one tensor entry (B,T,I)
        :param output: B,T,E
        """
        router_logits = output
        assert len(router_logits.shape) == 3, f"router_logits.shape: {router_logits.shape}"
        probs = F.softmax(router_logits, dim=-1)
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        self.entropy.append(entropy.cpu())

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "entropy": torch.stack(self.entropy),
            "entropy_mean": torch.stack(self.entropy).mean(dim=0),
            "entropy_std": torch.stack(self.entropy).std(dim=0),
        }


class LayerNormlContributionHook(ActivationsHook):
    def __init__(self, layernorm_layer: DeciLMRMSNorm, activation_hooks_kwargs: dict):
        self.agg_embedding_activations = torch.zeros(
            size=(layernorm_layer.weight.shape[0],),
            dtype=torch.float32,
            device=layernorm_layer.weight.device,
        )

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.agg_embedding_activations += (
            output.abs().float().mean(dim=list(range(output.ndim - 1)))
        )

    @classmethod
    def dump_activations_logs(
        cls: type["LayerNormlContributionHook"],
        activation_hooks: dict[str, "ActivationsHook"],
        activations_log_dir: Path | str,
        args: argparse.Namespace,
        runtime: IRuntime | None,
    ):
        """
        At the end of the default implementation of dumping activation scores to disc,
        save aggregated channel importance results.
        """

        super().dump_activations_logs(activation_hooks, activations_log_dir, args, runtime)

        rank = runtime.global_rank if runtime is not None else 0
        if rank == 0:
            LayerNormlContributionHook._save_channel_importance_results(
                activation_hooks, activations_log_dir, args
            )

        runtime.wait_for_everyone()

    @staticmethod
    def _save_channel_importance_results(
        activation_hooks: dict[str, ActivationsHook],
        activations_log_dir: Path,
        args: argparse.Namespace,
    ) -> None:
        """
        Save channel importance results from activation hooks.
        """

        # Find all activation files (for multi-rank scenarios)
        activations_log_dir = Path(activations_log_dir)
        activation_files = list(activations_log_dir.glob("rank_*.pth"))
        if not activation_files:
            aprint(f"Warning: No activation files found in {activations_log_dir}")
            return

        # Load and aggregate activation data from all ranks
        all_scores = []
        for activation_file in activation_files:
            aprint(f"Loading activations from {activation_file}")
            activation_data = torch.load(activation_file, map_location="cpu")

            # Extract scores from the activation data
            for module_name, hook_data in activation_data.items():
                if "score" in hook_data:
                    scores = hook_data["score"]
                    all_scores.append(scores)
                    aprint(f"Loaded {len(scores)} channel scores from {module_name}")

        if not all_scores:
            aprint("Warning: No valid activation data found")
            return

        # Average scores across all ranks and modules
        avg_scores = torch.stack(all_scores).mean(dim=0)
        aprint(f"Averaged {len(all_scores)} score sets into {len(avg_scores)} channels")

        # Create channel importance ranking (descending order)
        ranked_channels = torch.argsort(avg_scores, descending=True).tolist()

        # Create output data structure
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        output_data = {
            "model_path": getattr(args, "model_name_or_path", "unknown"),
            "dataset_path": getattr(args, "dataset_path", "unknown"),
            "experiment_id": getattr(args, "experiment_id", f"experiment_{timestamp}"),
            "eval_samples": getattr(args, "eval_samples", 0),
            "micro_batch_size": getattr(args, "micro_batch_size", 0),
            "timestamp": timestamp,
            "total_channels": len(ranked_channels),
            "channel_importance_ranking": ranked_channels,
            "channel_scores": avg_scores.tolist(),
            "score_statistics": {
                "min": float(avg_scores.min()),
                "max": float(avg_scores.max()),
                "mean": float(avg_scores.mean()),
                "std": float(avg_scores.std()),
            },
        }

        # Save the output
        output_path = activations_log_dir / "channel_importance_results.json"
        aprint(f"Saving channel importance data to {output_path}")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        # Print summary statistics
        aprint("=== Channel Importance Summary ===")
        aprint(f"Total channels: {len(ranked_channels)}")
        aprint(f"Top 10 most important channels: {ranked_channels[:10]}")
        aprint(f"Bottom 10 least important channels: {ranked_channels[-10:]}")
        aprint(f"Score range: {avg_scores.min():.4f} to {avg_scores.max():.4f}")
        aprint(f"Score mean: {avg_scores.mean():.4f}")
        aprint(f"Score std: {avg_scores.std():.4f}")

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "score": self.agg_embedding_activations.cpu(),
            "channels_importance_ascending": self.agg_embedding_activations.sort()[1].cpu(),
        }
