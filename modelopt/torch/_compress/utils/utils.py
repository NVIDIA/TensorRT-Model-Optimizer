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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# mypy: ignore-errors

import dataclasses
import functools
import json
import os
import pathlib
import pickle  # nosec B403 - pickle used for PyTorch model serialization
import re
import warnings
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from fire import Fire
from puzzle_tools.deci_lm_hf_code.block_config import AttentionConfig, BlockConfig, FFNConfig
from tqdm import tqdm


def calculate_kv_dim(n_heads_in_group: int, n_head: int, n_embd: int) -> int:
    if n_heads_in_group is None:
        return 0
    n_kv_heads = n_head // n_heads_in_group
    head_size = n_embd // n_head
    kv_dim = 2 * n_kv_heads * head_size
    return kv_dim


def raise_unknown_subblock_config_error(subblock_config: Any) -> None:
    raise ValueError(
        f"subblock_config should be an instance of FFNConfig or AttentionConfig, instead got {type(subblock_config)}"
    )


def sizeof_dtype(dtype: torch.dtype | str) -> int | float:
    """returns the size in bytes of the given dtype"""
    if dtype == "nvfp4":
        return 1 / 1.7
    return torch.tensor([], dtype=dtype).element_size()


def _sort_by_prefix(strings, prefix_order):
    """Sorts a list of strings based on a given order of their prefix."""
    prefix_index = {prefix: i for i, prefix in enumerate(prefix_order)}
    sorted_strings = sorted(
        strings, key=lambda s: (prefix_index.get(s.split(".")[0], len(prefix_order)), s)
    )
    return sorted_strings


def load_puzzle_solutions(
    puzzle_dir: str | pathlib.Path,
    include_patterns: list[str] = (),
    exclude_patterns: list[str] = (),
    drop_empty_columns: bool = True,
    default_generation_seq_len: int | None = None,
) -> pd.DataFrame:
    """Loads all puzzle solutions from a puzzle directory into a DataFrame.

    Supports `include_patterns` and `exclude_patterns` which are lists of expressions that will be
    evaluated with `re.findall()`. Can be used together.
    """
    puzzle_dir = pathlib.Path(puzzle_dir)

    # Find all matching solutions files.
    solution_files = list(puzzle_dir.glob("**/solutions.json"))
    if exclude_patterns:
        solution_files = [
            f
            for f in solution_files
            if not any(re.findall(pat, str(f)) for pat in exclude_patterns)
        ]
    if include_patterns:
        solution_files = [
            f for f in solution_files if any(re.findall(pat, str(f)) for pat in include_patterns)
        ]

    solution_records = []
    for solution_file in solution_files:
        with open(solution_file) as f:
            solutions = json.load(f)

        for sol_idx, sol in enumerate(solutions):
            mip_constraints = (
                sol["puzzle_args"].get("mip_constraints") or sol["puzzle_args"]["constraints"]
            )  # For backward compatibility with old puzzle formats.
            human_constraints = sol["puzzle_args"].get("human_constraints")
            args = (
                sol.get("subblock_stats") or sol["puzzle_args"]["subblock_stats_args"]
            )  # For partial backward compatibility - if its an old run without "subblock_stats",
            # we at least take the args used to filter the subblock_stats.
            total_costs = sol["total_costs"]
            total_value = sol["total_value"]
            chosen_items = sol["chosen_items"]

            if "generation_seq_len" not in args and default_generation_seq_len is None:
                raise ValueError(
                    "Trying to parse an old puzzle dir (without the full `subblock_stats` in "
                    "solution.json) without explicitly providing `default_generation_seq_len`. "
                    "For clarity of the calculated results we don't assume this is 1000 by default. "
                    "Please provide the argument."
                )

            gen_seq_len = args.get("generation_seq_len", default_generation_seq_len)
            calculated_cost_throughput = (
                gen_seq_len * args["batch_size"] / (total_costs["stats.runtime_ms"] / 1000)
                if "batch_size" in args and "stats.runtime_ms" in total_costs
                else np.nan
            )
            total_costs["throughput"] = calculated_cost_throughput

            calculated_constraint_throughput = (
                gen_seq_len * args["batch_size"] / (mip_constraints["stats.runtime_ms"] / 1000)
                if "batch_size" in args and "stats.runtime_ms" in mip_constraints
                else np.nan
            )

            record = (
                {
                    f"mip_constraint.{k.removeprefix('stats.')}": v
                    for k, v in mip_constraints.items()
                }
                | {f"human_constraint.{k}": v for k, v in human_constraints.items()}
                | {
                    f"args.{k}": v
                    for k, v in args.items()
                    if k in ("batch_size", "gpu", "prefill_seq_len", "generation_seq_len")
                }
                | {f"costs.{k.removeprefix('stats.')}": v for k, v in total_costs.items()}
                | total_value  # keys in `total_value` already includes the `total_value.` prefix
                | {
                    "calculated_constraint_throughput": calculated_constraint_throughput,
                    "solution_idx": sol_idx,
                    "solution_file": str(solution_file),
                    "chosen_items": chosen_items,
                }
            )
            solution_records.append(record)

    if not solution_records:
        raise ValueError(
            f"No solutions were found in {puzzle_dir} ({include_patterns=}, {exclude_patterns=})"
        )

    df = pd.DataFrame.from_dict(solution_records)
    df = df.reindex(
        _sort_by_prefix(
            df.columns, ["human_constraint", "mip_constraint", "args", "costs", "metrics"]
        ),
        axis="columns",
    )

    if drop_empty_columns:
        df = df.dropna(axis="columns", how="all")

    return df


def load_json(file_path: str):
    if not os.path.exists(file_path):
        print("file does not exist {file_path}")
        return None

    with open(file=file_path) as f:
        return json.load(f)


def save_json(obj: object, file_path: str):
    with open(file=file_path, mode="w") as f:
        return json.dump(obj, f)


def print_solution(solution_path: str, solution_id=0):
    solution = load_json(solution_path)
    if solution is not None:
        sol = solution
        if isinstance(solution, list):
            sol = solution[solution_id]
        elif isinstance(solution, dict) and solution.get("puzzle_solution") is not None:
            sol = solution.get("puzzle_solution")
        print(sol["solution_repr"])

        sol["total_costs"]["stats.num_params"] = (
            f"{sol['total_costs']['stats.num_params'] / 1e9:.2f}B"
        )

        print("costs are: ", sol["total_costs"])
        print("sum kl_div is : ", sol["total_value"])
        sum_kl_div = sol["total_value"]["metrics.kl_div_loss"]
        # actual kl_div
        validation_path = solution_path.replace(
            ".json", f"--validation/solution_{solution_id}.json"
        )
        if os.path.exists(validation_path):
            validation = load_json(validation_path)
            kl_div_loss = validation["kl_div_loss"]["avg"]
            lm_loss = validation["lm_loss"]["avg"]
            print(f"actual {kl_div_loss=}")
            print(f"actual {lm_loss=}")
            print(f"{sum_kl_div:.3f}, {kl_div_loss:.3f}, {lm_loss:.3f}")


def get_block_repr(parent_layer_indices, single_sequence_replacement):
    block_variant_name = ""
    if isinstance(single_sequence_replacement, list):
        for block_config in single_sequence_replacement:
            block_variant_name += block_config_to_str(deepcopy(block_config))

    else:
        block_variant_name = block_config_to_str(deepcopy(single_sequence_replacement))
    return f"block(s) {parent_layer_indices}: " + block_variant_name


def load_scores(validation_dir: str) -> pd.DataFrame:
    rows = []
    for solution_path in tqdm(list(pathlib.Path(validation_dir).glob("solution*.json"))):
        solution_info = json.loads(solution_path.read_text())
        solution_id = re.search(r"solution_(\d+)", solution_path.stem).group(1)
        # return solution_info
        # print(solution_path)
        # print(solution_info["puzzle_solution"]["single_sequence_replacement"].keys())
        replacement_info = solution_info["puzzle_solution"]["single_sequence_replacement"]
        parent_layer_indices = replacement_info["parent_layer_indices"]
        # kl_div_loss = solution_info['kl_div_loss']
        scores = {
            k: v["avg"] for k, v in solution_info.items() if isinstance(v, dict) and v.get("avg")
        }

        child_block_configs = deepcopy(replacement_info["child_block_configs"].copy())
        if isinstance(child_block_configs, list):
            block_variant_name = ""
            for block_config in child_block_configs:
                block_variant_name += block_config_to_str(deepcopy(block_config))
        else:
            block_variant_name = block_config_to_str(deepcopy(block_config))
        rows.append(
            {
                "solution_id": solution_id,
                "parent_layer_indices": parent_layer_indices,
                "block_variant_name": block_variant_name,
                "block_repr": get_block_repr(parent_layer_indices, block_config),
                "block_config": replacement_info["child_block_configs"],
                **scores,
            }
        )

    return pd.DataFrame(rows)


def load_val_results(validation_dir: str) -> pd.DataFrame:
    rows = []
    validation_dir = pathlib.Path(validation_dir)
    sol_paths = list(validation_dir.rglob("solution_0.json"))
    teacher_paths = list(validation_dir.rglob("teacher.json"))
    for solution_path in tqdm(sol_paths + teacher_paths):
        solution_info = json.loads(solution_path.read_text())
        solution_id = (
            str(solution_path.parent.parent.relative_to(validation_dir))
            if solution_path.name != "teacher.json"
            else "teacher"
        )
        scores = {
            k: v["avg"] for k, v in solution_info.items() if isinstance(v, dict) and v.get("avg")
        }
        rows.append({"solution_id": solution_id, **scores})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    return df


def validate_scores_with_solutions(validation_dir: str, solutions_path: str) -> pd.DataFrame:
    scores_df = load_scores(validation_dir=validation_dir)
    solutions_list = pd.read_json(solutions_path)

    def add_sol_num(idx: int, block_repr):
        return f"{idx}  {block_repr}"

    solutions_repr = [
        add_sol_num(
            idx,
            get_block_repr(
                s["single_block_replacement"]["block_idx"],
                s["single_block_replacement"]["block_config"],
            ),
        )
        for idx, s in enumerate(solutions_list)
    ]
    scores_df["sol_repr"] = scores_df["solution_id"] + " " + scores_df["block_repr"]

    assert len(np.setdiff(scores_df.sol_repr, solutions_repr)) == 0


def delete_scores(validation_dir: str, blocks_regex: str):
    scores_df = load_scores(validation_dir=validation_dir)

    score_ids_to_delete = scores_df.query(f'block_repr.str.contains("{blocks_regex}")').solution_id
    for score in score_ids_to_delete:
        scores_path = Path(validation_dir) / f"solution_{score}.json"
        print(f"about to delete: {scores_path}")
        os.remove(scores_path)


def solution_to_str(block_configs: list[dict[str, Any] | BlockConfig]) -> str:
    block_configs = deepcopy(block_configs)
    reps = []
    for block_idx, block_config in enumerate(block_configs):
        rep = f"block_{block_idx}:".ljust(9)
        rep += block_config_to_str(block_config)
        reps.append(rep)
    rep = "\n".join(reps) + "\n"
    return rep


def block_config_to_str(block_config: BlockConfig | dict[str, Any] | None) -> str | None:
    if block_config is None:
        return None
    rep = ""
    if dataclasses.is_dataclass(block_config):
        block_config = dataclasses.asdict(block_config)
    for subblock_name in ["attention", "ffn"]:
        subblock_config = block_config[subblock_name]
        rep += subblock_config_to_str(subblock_config, subblock_name)
    return rep


def subblock_config_to_str(
    subblock_config: FFNConfig | AttentionConfig | dict[str, Any] | None,
    subblock_name: None | str = None,
) -> str | None:
    if subblock_config is None:
        return None
    subblock_name = (
        "ffn"
        if isinstance(subblock_config, FFNConfig)
        else "mamba"
        if isinstance(subblock_config, AttentionConfig) and subblock_config.is_mamba
        else "attention"
        if isinstance(subblock_config, AttentionConfig)
        else subblock_name
    )
    assert subblock_name is not None, "Must provide subblock_name if subblock_config is a dict."

    if dataclasses.is_dataclass(subblock_config):
        subblock_config = dataclasses.asdict(subblock_config)

    if subblock_name == "attention" and subblock_config.get("mamba") is not None:
        subblock_name = "mamba"

    if subblock_name == "ffn" and subblock_config.get("moe") is not None:
        subblock_name = "moe"

    rep = f"  {subblock_name}"
    if subblock_config.get("no_op"):
        rep += "  no_op".ljust(8)
    elif subblock_config.get("replace_with_linear"):
        rep += "  linear".ljust(8)
    elif subblock_name == "ffn":
        intermediate_size = subblock_config["intermediate_size"]
        rep += f"  intermediate_{intermediate_size}".ljust(8)
    elif subblock_name == "attention":
        n_heads_in_group = subblock_config["n_heads_in_group"]
        rep += f"  gqa_{n_heads_in_group}".ljust(8)
    elif subblock_name == "mamba":
        mamba_num_heads = subblock_config["mamba"]["num_heads"]
        mamba_head_dim = subblock_config["mamba"]["head_dim"]
        rep += f"  num_heads_{mamba_num_heads}  head_dim_{mamba_head_dim}".ljust(8)
    elif subblock_name == "moe":
        moe_num_local_experts = subblock_config["moe"]["num_local_experts"]
        moe_expert_intermediate_dim = subblock_config["moe"]["expert_intermediate_dim"]
        shared_expert_intermediate_dim = subblock_config["moe"]["shared_expert_intermediate_dim"]
        num_experts_per_tok = subblock_config["moe"]["num_experts_per_tok"]
        rep += (
            f"  num_experts_{moe_num_local_experts}  "
            f"expert_intermediate_dim_{moe_expert_intermediate_dim}  "
            f"shared_expert_intermediate_dim_{shared_expert_intermediate_dim}  "
            f"num_experts_per_tok_{num_experts_per_tok}"
        ).ljust(8)
    else:
        raise ValueError(f"subblock_config_to_str: unrecognized subblock_name: {subblock_name}.")

    return rep


def pareto_frontier(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_bigger_is_better: bool = False,  # default: smaller x is better
    y_bigger_is_better: bool = True,  # default: bigger y is better
) -> pd.DataFrame:
    """
    Returns the Pareto frontier (non-dominated points) from df based on the criteria:
    - For the x-axis, if x_bigger_is_better is True, then higher x values are preferred;
      if False, lower x values are preferred.
    - For the y-axis, if y_bigger_is_better is True, then higher y values are preferred;
      if False, lower y values are preferred.

    A point is considered dominated if there exists another point that is strictly better
    in both dimensions.
    """
    # Extract the columns as numpy arrays.
    X = df[x].to_numpy().copy()  # noqa: N806
    Y = df[y].to_numpy().copy()  # noqa: N806

    # Transform the coordinates so that "higher is always better" in both dimensions.
    if not x_bigger_is_better:
        X = -X  # noqa: N806
    if not y_bigger_is_better:
        Y = -Y  # noqa: N806

    n_points = len(df)
    is_dominated = np.zeros(n_points, dtype=bool)

    # For each point, check if any other point is strictly better in both dimensions.
    for i in range(n_points):
        domination_mask = (X[i] < X) & (Y[i] < Y)
        if np.any(domination_mask):
            is_dominated[i] = True

    # Return the DataFrame filtered to only include non-dominated points.
    return df[~is_dominated]


def non_max_suppression(
    df: pd.DataFrame,
    x: str,
    y: str,
    max_x_diff: float,
    y_bigger_is_better: bool,
) -> pd.DataFrame:
    """
    Filter rows in the DataFrame such that if two rows are within `diff` along the x-axis,
    only the one with the preferred y value is kept.

    Parameters:
    - df: pandas DataFrame.
    - x: Column name for the x-axis values.
    - y: Column name for the y-axis values.
    - max_x_diff: Distance threshold on the x-axis.
    - y_bigger_is_better:
          If True, keeps the row with the larger y value (default).
          If False, keeps the row with the smaller y value.

    Returns:
    - A DataFrame with only the selected rows.
    """
    # Sort by y: descending if higher y is preferred, ascending if lower y is preferred.
    df_sorted = df.sort_values(y, ascending=not y_bigger_is_better)
    kept_indices = []

    # Iterate over rows in the sorted DataFrame.
    for idx, row in df_sorted.iterrows():
        x_val = row[x]
        # Skip the row if its x value is within diff of any already-kept row.
        if any(abs(x_val - df.loc[kept_idx, x]) < max_x_diff for kept_idx in kept_indices):
            continue
        kept_indices.append(idx)

    # Return the filtered DataFrame (optionally sorted in the original order).
    return df.loc[kept_indices]


def soft_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_bigger_is_better: bool,
    window: int | Literal["auto"] = "auto",
    median_diff_factor: float = 2.5,
) -> pd.DataFrame:
    """
    Removes low-valued outliers in a sliding window fashion.
    Good for getting a soft-pareto frontier, that keeps more than just the very best values.
    The auto window is len(df) // 5.
    """
    y = df.sort_values(x_col)[y_col]
    if not y_bigger_is_better:
        y = -y
    y_to_keep = rolling_low_values_filter(y, window, median_diff_factor)
    indices_to_keep = y_to_keep.index
    return df.loc[indices_to_keep]


def rolling_low_values_filter(
    s: pd.Series,
    window: int | Literal["auto"] = "auto",
    median_diff_factor: float = 2.5,
) -> pd.Series:
    """
    Implements a rolling function that does this given a window:
    1. calculates the max in the window
    2. calculates diff=(max-y) for each y in the window
    3. calculates the median diff
    4. marks values that are smaller than max - 2 * median_diff
    5. removes from the series entries that were marked at least once
    """
    s = s[s.notna()]
    if window == "auto":
        window = len(s) // 5

    # Create a boolean mask with the same index as the series.
    marks = pd.Series(False, index=s.index)

    # Iterate over every possible window
    for start in range(len(s) - window + 1):
        # Select the current window
        window_slice = s.iloc[start : start + window]

        # Step 1: Compute the maximum in the window
        max_val = window_slice.max()

        # Step 2: Compute the difference between max and each value in the window
        diff = max_val - window_slice

        # Step 3: Compute the median of these differences
        median_diff = diff.median()

        # Step 4: Identify values that are smaller than (max - median_diff_factor * median_diff)
        threshold = max_val - median_diff_factor * median_diff
        mark_window = window_slice < threshold

        # Update the mask: mark an index if it is marked in any window
        marks.iloc[start : start + window] |= mark_window

    # Step 5: Remove all entries that were marked at least once
    return s[~marks]


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with


        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
                model = LLaMA(model_config)
            model.load_state_dict(torch.load("llama-lit/7B/lit-llama.pth"))"""

        self.device = device
        self.dtype = dtype

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)


# this is taken from torchhacks https://github.com/lernapparat/torchhacks


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func, new_type, args, state, *, archiveinfo=None):
        ret = func(*args)
        if isinstance(ret, NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)

            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(cls, data, requires_grad, backward_hooks, *, archiveinfo=None):
        if isinstance(data, NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls,
        storage,
        storage_offset,
        size,
        stride,
        requires_grad,
        backward_hooks,
        metadata=None,
        *,
        archiveinfo=None,
    ):
        rebuild_args = (
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        metatensor = torch._utils._rebuild_tensor_v2(
            storage,
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        storageinfo = storage.archiveinfo
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        name, storage_cls, fn, device, size = self.storageinfo
        dtype = self.metatensor.dtype

        uts = (
            self.archiveinfo.zipfile_context.zf.get_storage_from_record(
                f"data/{fn}",
                size * torch._utils._element_size(dtype),
                torch.UntypedStorage,
            )
            ._typed_storage()
            ._untyped_storage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = torch.storage.TypedStorage(
                wrap_storage=uts, dtype=self.metatensor.dtype, _internal=True
            )
        tensor = torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [(a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args]
        res = func(*loaded_args, **kwargs)
        # gc.collect would be costly here, maybe do it optionally
        return res

    def __getattr__(self, name):
        # properties
        ## TODO: device, is_...??
        ## TODO: mH, mT, H, T, data, imag, real
        ## name ???
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        # materializing with contiguous is needed for quantization
        if name in {"contiguous"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({self.metatensor!r})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile_context):
        super().__init__(file)
        self.zipfile_context = zipfile_context

    def find_class(self, module, name):
        res = super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return functools.partial(NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        elif module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return functools.partial(NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        elif module == "torch._utils" and name == "_rebuild_parameter":
            return functools.partial(NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return res

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        s.archiveinfo = pid
        return s


class LazyLoad:
    def __init__(self, fn):
        self.zf = torch._C.PyTorchFileReader(str(fn))
        with BytesIO(self.zf.get_record("data.pkl")) as pkl:
            mup = LazyLoadingUnpickler(pkl, self)
            self.sd = mup.load()

    def __enter__(self):
        return self.sd

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.zf  # I don't think there is a way to force closing...
        self.zf = None


# TODO normalize_storage_type not defined
# class SavingProxyForStorage:
#     def __init__(self, obj, saver, protocol_version=5):
#         self.protocol_version = protocol_version
#         self.saver = saver
#         if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
#             raise TypeError(f"expected storage, not {type(obj)}")

#         # this logic is taken from PyTorch 2.0+ torch/serialization.py
#         if isinstance(obj, torch.storage.TypedStorage):
#             # PT upstream wants to deprecate this eventually...
#             storage = obj._untyped_storage
#             storage_type_str = obj._pickle_storage_type()
#             storage_type = getattr(torch, storage_type_str)
#             storage_numel = obj._size()
#         else:
#             storage = obj
#             storage_type = normalize_storage_type(type(obj))
#             storage_numel = storage.nbytes()

#         storage_key = saver._write_storage_and_return_key(storage)
#         location = torch.serialization.location_tag(storage)

#         self.storage_info = (
#             "storage",
#             storage_type,
#             storage_key,
#             location,
#             storage_numel,
#         )

#     def __reduce_ex__(self, protocol_version):
#         assert False, "this should be handled with out of band"

# TODO: SavingProxyForStorage not defined
# class SavingProxyForTensor:
#     def __init__(self, tensor, saver, protocol_version=5):
#         self.protocol_version = protocol_version
#         self.reduce_ret_fn, (storage, *other_reduce_args) = tensor.__reduce_ex__(protocol_version)
#         assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
#         storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
#         self.reduce_args = (storage_proxy, *other_reduce_args)

#     def __reduce_ex__(self, protocol_version):
#         if protocol_version != self.protocol_version:
#             raise RuntimeError(
#                 f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}"
#             )
#         return self.reduce_ret_fn, self.reduce_args


# TODO normalize_storage_type not defined
# class IncrementalPyTorchPickler(pickle.Pickler):
#     def __init__(self, saver, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.storage_dtypes = {}
#         self.saver = saver
#         self.id_map = {}

#     # this logic is taken from PyTorch 2.0+ torch/serialization.py
#     def persistent_id(self, obj):
#         # FIXME: the docs say that persistent_id should only return a string
#         # but torch store returns tuples. This works only in the binary protocol
#         # see
#         # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
#         # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
#         if isinstance(obj, SavingProxyForStorage):
#             return obj.storage_info

#         if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
#             if isinstance(obj, torch.storage.TypedStorage):
#                 # TODO: Once we decide to break serialization FC, this case
#                 # can be deleted
#                 storage = obj._untyped_storage
#                 storage_dtype = obj.dtype
#                 storage_type_str = obj._pickle_storage_type()
#                 storage_type = getattr(torch, storage_type_str)
#                 storage_numel = obj._size()

#             else:
#                 storage = obj
#                 storage_dtype = torch.uint8
#                 storage_type = normalize_storage_type(type(obj))
#                 storage_numel = storage.nbytes()

#             # If storage is allocated, ensure that any other saved storages
#             # pointing to the same data all have the same dtype. If storage is
#             # not allocated, don't perform this check
#             if storage.data_ptr() != 0:
#                 if storage.data_ptr() in self.storage_dtypes:
#                     if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
#                         raise RuntimeError(
#                             "Cannot save multiple tensors or storages that "
#                             "view the same data as different types"
#                         )
#                 else:
#                     self.storage_dtypes[storage.data_ptr()] = storage_dtype

#             storage_key = self.id_map.get(storage._cdata)
#             if storage_key is None:
#                 storage_key = self.saver._write_storage_and_return_key(storage)
#                 self.id_map[storage._cdata] = storage_key
#             location = torch.serialization.location_tag(storage)

#             return ("storage", storage_type, storage_key, location, storage_numel)

#         return None


# TODO IncrementalPyTorchPickler not defined
# class IncrementalSave:
#     def __init__(self, name):
#         self.name = name
#         self.zipfile = torch._C.PyTorchFileWriter(str(name))
#         self.has_saved = False
#         self.next_key = 0

#     def __enter__(self):
#         return self

#     def store_early(self, tensor):
#         if isinstance(tensor, torch.Tensor):
#             return SavingProxyForTensor(tensor, self)
#         raise TypeError(f"can only store tensors early, not {type(tensor)}")

#     def save(self, obj):
#         if self.has_saved:
#             raise RuntimeError("have already saved")
#         # Write the pickle data for `obj`
#         data_buf = BytesIO()
#         pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
#         pickler.dump(obj)
#         data_value = data_buf.getvalue()
#         self.zipfile.write_record("data.pkl", data_value, len(data_value))
#         self.has_saved = True

#     def _write_storage_and_return_key(self, storage):
#         if self.has_saved:
#             raise RuntimeError("have already saved")
#         key = self.next_key
#         self.next_key += 1
#         name = f"data/{key}"
#         if storage.device.type != "cpu":
#             storage = storage.cpu()
#         num_bytes = storage.nbytes()
#         self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
#         return key

#     def __exit__(self, type, value, traceback):
#         self.zipfile.write_end_of_file()


if __name__ == "__main__":
    Fire()
