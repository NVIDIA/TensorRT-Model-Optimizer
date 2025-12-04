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

"""Solves the grouped knapsack problem using Mixed Integer Programming to find optimal item selections."""

# mypy: ignore-errors
import math
import warnings
from copy import deepcopy
from random import random
from typing import Any, Hashable, Iterable, Optional, TypeAlias, Union

from mip import BINARY, Model, maximize, minimize, xsum
from tqdm import tqdm

from .utils import InfeasibleError, get_nested_key

Item: TypeAlias = dict[str, float | dict[str, float]]
Group: TypeAlias = dict[Hashable, Item]
ChosenItems: TypeAlias = dict[Hashable, Hashable]


def multi_solution_grouped_knapsack(
    groups: dict[Hashable, Group],
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
    num_solutions: int,
    minimal_diversity: int = 1,
    max_seconds_per_solution: Optional[float] = None,
) -> list[dict[str, Union[ChosenItems, float]]]:
    solutions = []
    previous_choices = []
    for i_run in tqdm(range(num_solutions), desc="multi_solution_grouped_knapsack"):
        try:
            chosen_items, total_value, total_costs = grouped_knapsack(
                groups,
                objective,
                constraints,
                bigger_is_better,
                previous_choices,
                minimal_diversity,
                max_seconds_per_solution,
            )
        except InfeasibleError:
            warnings.warn(f"Found only {i_run} feasible solutions (requested {num_solutions})")
            break
        previous_choices.append(chosen_items)
        solutions.append(
            {"chosen_items": chosen_items, "total_value": total_value, "total_costs": total_costs}
        )
    return solutions


def grouped_knapsack(
    groups: dict[Hashable, Group],
    objective: str,
    constraints: dict[str, float | tuple[float, float]],
    bigger_is_better: bool,
    previous_choices: Optional[list[ChosenItems]] = None,
    minimal_diversity: int = 1,
    max_seconds_per_solution: Optional[float] = None,
) -> tuple[ChosenItems, float, dict[str, float]]:
    groups = deepcopy(groups)
    mip_model = Model()

    objective_vars = []
    constraint_vars = {constraint_key: [] for constraint_key in constraints.keys()}
    for group_name, group_items in groups.items():
        group_vars = []
        for item_name, item in group_items.items():
            is_chosen = mip_model.add_var(var_type=BINARY)
            item["is_chosen"] = is_chosen
            group_vars.append(is_chosen)
            objective_vars.append(is_chosen * get_nested_objective(item, objective))
            for constraint_key in constraints.keys():
                constraint_vars[constraint_key].append(
                    is_chosen * get_nested_key(item, constraint_key)
                )

        mip_model += xsum(group_vars) == 1

    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost

        if max_cost is not None:
            mip_model += xsum(constraint_vars[constraint_key]) <= max_cost
        if min_cost is not None:
            mip_model += xsum(constraint_vars[constraint_key]) >= min_cost

    if previous_choices is not None:
        for previous_chosen_items in previous_choices:
            corresponding_vars = [
                groups[group_name][item_name]["is_chosen"]
                for group_name, item_name in previous_chosen_items.items()
            ]
            mip_model += xsum(corresponding_vars) <= len(groups) - minimal_diversity

    mip_model.objective = (
        maximize(xsum(objective_vars)) if bigger_is_better else minimize(xsum(objective_vars))
    )

    if max_seconds_per_solution is not None:
        mip_model.max_seconds = max_seconds_per_solution

    mip_model.optimize()

    if is_chosen.x is None:
        raise InfeasibleError()

    total_value = 0.0
    total_costs = {constraint_key: 0 for constraint_key in constraints.keys()}
    chosen_items: ChosenItems = dict()
    for group_name, group_items in groups.items():
        for item_name, item in group_items.items():
            is_chosen = item["is_chosen"].x >= 0.99
            if is_chosen:
                assert group_name not in chosen_items
                chosen_items[group_name] = item_name
                total_value += get_nested_objective(item, objective)
                for constraint_key in constraints.keys():
                    total_costs[constraint_key] += get_nested_key(item, constraint_key)

    if len(chosen_items) != len(groups):
        in_groups_and_not_in_chosen_items = set(groups.keys()) - set(chosen_items.keys())
        in_chosen_items_and_not_in_groups = set(chosen_items.keys()) - set(groups.keys())
        missing_groups = [groups[key] for key in in_groups_and_not_in_chosen_items]
        raise RuntimeError(f"""
        Different number of 'chosen_items' and 'groups': {len(chosen_items)=}  {len(groups)=}
        {in_groups_and_not_in_chosen_items=}
        {in_chosen_items_and_not_in_groups=}
        {missing_groups=}
        """)

    for constraint_key, max_cost in constraints.items():
        min_cost = None
        if isinstance(max_cost, Iterable):
            min_cost, max_cost = max_cost

        if max_cost is not None:
            assert total_costs[constraint_key] < max_cost or math.isclose(
                total_costs[constraint_key], max_cost, rel_tol=1e-9
            ), (
                f"This max_cost was violated {constraint_key} in the solution, sol val={total_costs[constraint_key]} > {max_cost=}"
            )
        if min_cost is not None:
            assert total_costs[constraint_key] > min_cost or math.isclose(
                total_costs[constraint_key], min_cost, rel_tol=1e-9
            ), (
                f"This min_cost was violated {constraint_key} in the solution, sol val={total_costs[constraint_key]} < {min_cost=}"
            )

    for previous_chosen_items in previous_choices:
        num_differences = 0
        for group_name in groups.keys():
            num_differences += previous_chosen_items[group_name] != chosen_items[group_name]
        assert num_differences >= minimal_diversity

    return chosen_items, total_value, total_costs


def get_nested_objective(dictionary: dict[str, Any], nested_key: str) -> Any:
    if nested_key.startswith("metrics."):
        # handle metrics that have '.' in their name
        metric = nested_key.split("metrics.")[1]
        return dictionary["metrics"][metric]
    else:
        return get_nested_key(dictionary, nested_key)


def usage_example():
    num_layers = 32
    num_configs_per_block = 100
    groups = {
        f"layer_{i_layer}": {
            f"config_{i_config}": {
                "metrics": {"accuracy": random()},
                "stats": {"memory_mib": random() * 100, "runtime_ms": random() * 10},
            }
            for i_config in range(num_configs_per_block)
        }
        for i_layer in range(num_layers)
    }

    minimal_diversity = 10
    constraints = {"stats.memory_mib": num_layers * 50.0, "stats.runtime_ms": num_layers * 5.0}
    solutions = multi_solution_grouped_knapsack(
        groups,
        objective="metrics.accuracy",
        constraints=constraints,
        bigger_is_better=True,
        num_solutions=10,
        minimal_diversity=minimal_diversity,
    )

    print()
    print(constraints)

    for i_run, solution in enumerate(solutions):
        print()
        print(f"run {i_run}")
        print(solution)

    print(f"Checking differences, should be at least {minimal_diversity}:")
    for a in range(len(solutions)):
        for b in range(a + 1, len(solutions)):
            num_differences = 0
            for group_name in groups.keys():
                num_differences += (
                    solutions[a]["chosen_items"][group_name]
                    != solutions[b]["chosen_items"][group_name]
                )
            print(a, "<>", b, "=", num_differences)


if __name__ == "__main__":
    usage_example()
