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

"""Performs greedy search to find optimal multi-layer replacements under resource constraints."""

# mypy: ignore-errors
import math
from copy import deepcopy
from random import random
from typing import Any, Hashable, TypeAlias

from .utils import InfeasibleError, consecutive_ngrams, get_nested_key, sort_replacements

ReplacementID: TypeAlias = Hashable
Replacement: TypeAlias = dict[str, Any]
ChosenReplacements: TypeAlias = list[Replacement]


def run_greedy_search(
    teacher_replacements: list[Replacement],
    student_replacements: list[Replacement],
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
) -> tuple[ChosenReplacements, float, dict[str, float]]:
    print("#######  running greedy search  #######")
    teacher_replacements = deepcopy(teacher_replacements)
    student_replacements = deepcopy(student_replacements)
    chosen_replacements: ChosenReplacements = []

    teacher_replacements = {
        replacement["parent_layer_indices"][0]: replacement for replacement in teacher_replacements
    }

    all_parent_layers = set(teacher_replacements.keys())
    uncovered_parent_layers = set(all_parent_layers)

    while True:
        if len(student_replacements) == 0:
            raise InfeasibleError()

        choice_func = max if bigger_is_better else min
        best_replacement = choice_func(
            student_replacements, key=lambda replacement: get_nested_key(replacement, objective)
        )
        chosen_replacements.append(best_replacement)
        uncovered_parent_layers -= set(best_replacement["parent_layer_indices"])
        student_replacements = _filter_overlapping_replacements(
            student_replacements, uncovered_parent_layers
        )

        padded_chosen_replacements = list(chosen_replacements)
        for uncovered_block_idx in uncovered_parent_layers:
            padded_chosen_replacements.append(teacher_replacements[uncovered_block_idx])

        all_constraints_satisfied = True
        for constraint_key, max_cost in constraints.items():
            total_cost = sum(
                get_nested_key(replacement, constraint_key)
                for replacement in padded_chosen_replacements
            )
            is_constraint_satisfied = total_cost < max_cost or math.isclose(
                total_cost, max_cost, rel_tol=1e-9
            )
            if not is_constraint_satisfied:
                all_constraints_satisfied = False

        if all_constraints_satisfied:
            chosen_replacements = padded_chosen_replacements
            break

    # Trust But Verify: calculate total value and costs, and check that all the constraints are filled
    total_value = 0.0
    total_costs = {constraint_key: 0 for constraint_key in constraints.keys()}
    chosen_layers = set()
    for replacement in chosen_replacements:
        total_value += get_nested_key(replacement, objective)
        for constraint_key in constraints.keys():
            total_costs[constraint_key] += get_nested_key(replacement, constraint_key)
        for parent_layer_idx in replacement["parent_layer_indices"]:
            assert parent_layer_idx not in chosen_layers, (
                f"Found duplicate chosen layer {parent_layer_idx}"
            )
            chosen_layers.add(parent_layer_idx)

    missing_layers = all_parent_layers - set(chosen_layers)
    assert len(missing_layers) == 0, (
        f"The following layers were not chosen by any replacement:\n{missing_layers=}\n{chosen_replacements}"
    )

    for constraint_key, max_cost in constraints.items():
        assert total_costs[constraint_key] < max_cost or math.isclose(
            total_costs[constraint_key], max_cost, rel_tol=1e-9
        ), (
            f"this constraint was violated {constraint_key} in the solution, sol val={total_costs[constraint_key]} <= {max_cost=}"
        )

    chosen_replacements = sort_replacements(chosen_replacements)
    for cr in chosen_replacements:
        if "block_config" in cr:
            cr["child_block_configs"] = cr["block_config"]

    return [
        {
            "chosen_replacements": chosen_replacements,
            "total_value": total_value,
            "total_costs": total_costs,
        }
    ]


def _filter_overlapping_replacements(
    replacements: list[Replacement],
    uncovered_parent_layers: set[int],
) -> list[Replacement]:
    return [
        replacement
        for replacement in replacements
        if set(replacement["parent_layer_indices"]).issubset(uncovered_parent_layers)
    ]


def usage_example():
    num_layers = 32
    num_options_per_parent_replacement = 5

    teacher_replacements = []
    student_replacements = []
    for num_layers_in_replacement in (1, 2, 3):
        for i_option in range(num_options_per_parent_replacement):
            for parent_layer_indices in consecutive_ngrams(num_layers, num_layers_in_replacement):
                is_teacher = num_layers_in_replacement == 1 and i_option == 0
                replacement_id = f"parent layers {parent_layer_indices}  child config {i_option}"
                replacement = {
                    "parent_layer_indices": parent_layer_indices,
                    "metrics": {"loss": random() if not is_teacher else 0.0},
                    "stats": {"cost": 1},
                    "replacement_id": replacement_id,
                }
                if is_teacher:
                    teacher_replacements.append(replacement)
                else:
                    student_replacements.append(replacement)

    constraints = {"stats.cost": num_layers - 8}
    (result,) = run_greedy_search(
        teacher_replacements,
        student_replacements,
        objective="metrics.loss",
        constraints=constraints,
        bigger_is_better=False,
    )
    chosen_replacements = result["chosen_replacements"]
    total_value = result["total_value"]
    total_costs = result["total_costs"]

    print()
    print()
    print(f"{total_value=}")
    print(f"{total_costs=}")
    print(f"{constraints=}")
    print("chosen_replacements=")
    print(chosen_replacements)
    print("\n".join([rep["replacement_id"] for rep in chosen_replacements]))


if __name__ == "__main__":
    usage_example()
