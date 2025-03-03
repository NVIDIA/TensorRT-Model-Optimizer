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

"""High-level search and model design algorithms to help you optimize your model."""

import copy
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import QhullError
from torch import nn

from modelopt.torch.opt.searcher import ConstraintsDict, Deployment, LimitsTuple
from modelopt.torch.opt.utils import is_configurable
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import param_num_from_forward, random, unwrap_model

from .search_space import SampleFunc
from .utils import (
    _SearchSpaceUnwrapped,
    get_subnet_config,
    inference_flops,
    sample,
    sample_and_reset,
)

ConstraintsRes = dict[str, float]
ConstraintEvalFunc = Callable[[Optional[ConstraintsRes]], float]


class ConstraintInterpolator:
    """A piece-wise, linear, d-dim interpolator to interpolate some model property like latency.

    The interpolator is based on scipy.interpolate.LinearNDInterpolator, which is based on an
    d-dimensional linear interpolation from unstructured d-dimensional data points. The interpolator
    uses a Delaunay triangulation as intermediate data representation.

    Note that for a successful triangulation/interpolation, at least d+1 linearly independent points
    are needed that form a convex hull around the desired query point.

    Every time a new data point is queried, the interpolator will be dynamically re-generated as
    needed.
    """

    def __init__(
        self,
        model: nn.Module,
        points_funcs: dict[str, ConstraintEvalFunc],
        value_func: ConstraintEvalFunc,
    ):
        """Initialize interpolator with points_func and value_func to interpolate."""
        assert len(points_funcs) > 0, "Need at least one point function to interpolate from!"
        self.model = model
        self._points_funcs = points_funcs
        self._value_func = value_func
        self.points = np.zeros((0, len(points_funcs)))
        self._vals = np.zeros(0)
        self._interpolator = None

        # In collect mode, data points are added to the grid even if the interpolator could be used
        # in order to improve the interpolator's quality.
        self.collect_mode = False

    def _add_point(self, point: Union[list, NDArray]) -> float:
        """Add point to interpolation table and return value."""
        # compute new value
        val = self._value_func(self.model)

        # add point/va. to our database of points
        self.points = np.vstack((self.points, np.asarray(point)[None]))
        self._vals = np.hstack((self._vals, val))

        # dynamically re-generate the interpolator
        try:
            if len(self._points_funcs) > 1:
                self._interpolator = LinearNDInterpolator(self.points, self._vals)
            elif len(self._vals) > 1:  # can only use interp1d with 2+ points collected
                self._interpolator = interp1d(self.points[:, 0], self._vals, bounds_error=False)
            else:
                self._interpolator = None  # not enough points collected yet
        except QhullError:
            self._interpolator = None  # not enough linearly independent points collected yet

        # return new point
        return val

    def __call__(self, pre_computed: Optional[dict[str, float]] = None) -> float:
        """Interpolate based on the provided model query or compute a new, exact data point."""
        # compute new point but only query as necessary
        pre_computed = pre_computed or {}
        point = [
            pre_computed[key] if key in pre_computed else func(self.model)
            for key, func in self._points_funcs.items()
        ]

        # check if we can use the interpolator or need/want to query the value directly
        if self._interpolator is None or self.collect_mode:
            val = self._add_point(point)
        else:
            val = self._interpolator(point)
            # check if we landed outside the convex hull (val == NaN) and if so add it as new point
            if np.isnan(val):
                val = self._add_point(point)

        # return value
        return np.asarray(val).item()


class ConstraintsFunc:
    """A Functor class to check if sub-net satisfied all provided constraints.

    We intentionally expose some attributes like `limits` s.t. we can modify it manually.
    """

    _sample_points_dict: dict[tuple[str, ...], dict[str, SampleFunc]] = {
        ("flops",): {"min": min, "centroid": random.centroid, "max": max},
        ("flops", "flops_min_depth"): {
            "min": min,
            "min_regular_centroid_depth": {"*": min, "*.depth": random.centroid},
            "min_regular_max_depth": {"*": min, "*.depth": max},
            "centroid_regular_min_depth": {"*": random.centroid, "*.depth": min},
            "centroid": random.centroid,
            "centroid_regular_max_depth": {"*": random.centroid, "*.depth": max},
            "max_regular_min_depth": {"*": max, "*.depth": min},
            "max_regular_centroid_depth": {"*": max, "*.depth": random.centroid},
            "max": max,
        },
    }

    def __init__(
        self,
        model: nn.Module,
        constraints: ConstraintsDict,
        dummy_input: Union[Any, tuple],
        deployment: Optional[Deployment] = None,
        fast_eval: bool = True,
        dp_group=None,
    ):
        self.model = model
        self.dummy_input = dummy_input
        self.deployment = deployment
        self._fast_eval = fast_eval

        # Getting data parallel group for
        self.dp_group = dp_group

        # initialize latency interpolator
        keys_for_interpolation = ("flops",)
        if ConstraintsFunc.is_configurable(self.model, "depth"):
            keys_for_interpolation += ("flops_min_depth",)
        self._latency_interpolator = ConstraintInterpolator(
            self.model,
            points_funcs={k: self.constraint_eval_funcs[k] for k in keys_for_interpolation},
            value_func=self._get_true_latency,
        )
        # set fast/regular mode for latency interpolator
        self._latency_interpolator.collect_mode = not self.fast_eval

        # set limit at the end with setter to use sanity checks on constraints
        self._limits = {}
        self.limits = constraints

    @staticmethod
    def is_configurable(model: nn.Module, hparam_suffix: Optional[str] = None):
        return hparam_suffix is None or any(
            k.rpartition(".")[-1] == hparam_suffix for k in get_subnet_config(model, True).keys()
        )

    def get_sample_points(self) -> dict[str, SampleFunc]:
        """Get the sample points required for latency interpolator."""
        return self._sample_points_dict[tuple(self._latency_interpolator._points_funcs)]

    @property
    def fast_eval(self) -> bool:
        """Return boolean indicator whether we use faster interpolation for latency."""
        return self._fast_eval

    @fast_eval.setter
    def fast_eval(self, value: bool):
        """Set bool whether we should use faster interpolation for latency."""
        self._fast_eval = value
        self._latency_interpolator.collect_mode = not value

    @property
    def limits(self) -> dict[str, LimitsTuple]:
        """Get limits for all constraints."""
        return self._limits

    @limits.setter
    def limits(self, constraints: ConstraintsDict) -> None:
        # check that no unexpected keys are found in provided constraints dict.
        if not constraints.keys() <= self.constraint_eval_funcs.keys():
            raise KeyError(
                f"Found unexpected keys:\n{constraints.keys() - self.constraint_eval_funcs.keys()}",
                f"\nCurrently supported constraint keys:\n{self.constraint_eval_funcs.keys()}",
            )

        # check that deployment is provided when latency is used as a constraint.
        if "latency" in constraints and self.deployment is None:
            raise ValueError(
                "Deployment config must be provided when latency is used as a constraint."
            )

        # making sure we add both "flops" and "params" by default as fake constraints
        # since it's cheap and can provide useful insights during constraint evaluation.
        # Moreover, we need it for latency interpolation anyway.
        constraints_complete: ConstraintsDict = {
            k: None for k in ("flops", "params") if k in self.constraint_eval_funcs
        }
        constraints_complete.update(**constraints)

        if "latency" in constraints:
            fake_constraints = {k: None for k in self._latency_interpolator._points_funcs.keys()}
            constraints_complete.update(fake_constraints)

        self._limits = self.get_expanded_limits(constraints_complete)

    @property
    def effective_limits(self) -> dict[str, LimitsTuple]:
        return {k: v for k, v in self.limits.items() if v != self.trivial_limit}

    @property
    def trivial_limit(self) -> LimitsTuple:
        return (0, float("inf"))

    def get_expanded_limits(self, constraints: ConstraintsDict) -> dict[str, LimitsTuple]:
        """Extract and standardize limits from constraint dictionary."""
        # standardize limits to LimitsTuple: (lower_bound, upper_bound)
        expanded_limits = {}
        for k, lim in copy.deepcopy(constraints).items():
            if lim is None:
                expanded_limits[k] = self.trivial_limit
                continue

            if isinstance(lim, str):
                assert lim.endswith("%")
                fraction_value = float(lim[:-1]) / 100

                # Use _SearchSpaceUnwrapped to avoid reseting the calibration statistics
                ss = _SearchSpaceUnwrapped(self.model)
                config = ss.config()
                ss.sample(random.original)
                orig_value = self.constraint_eval_funcs[k]()  # type: ignore[call-arg]
                ss.select(config)

                lim = orig_value * fraction_value

            expanded_limits[k] = (0, lim)  # lower bound is always 0 by default

        return expanded_limits

    def set_rel_lower_bounds(self, rel_lower: float = 0.85) -> None:
        """Set relative lower bounds for all constraints."""
        for k, (lb, ub) in self._limits.items():
            assert lb == 0, "Lower bound has already been set!"
            if self._limits[k] != self.trivial_limit:
                self._limits[k] = (rel_lower * ub, ub)

    def _get_params(self, _: Optional[ConstraintsRes] = None) -> float:
        """Get number of model parameters from forward pass."""
        return param_num_from_forward(self.model, args=self.dummy_input, unit=1.0)

    def _get_flops(self, _: Optional[ConstraintsRes] = None) -> float:
        """Get inference FLOPs."""
        return float(inference_flops(self.model, dummy_input=self.dummy_input, unit=1.0))

    def _get_flops_min_depth(self, _: Optional[ConstraintsRes] = None) -> float:
        """Get inference FLOPs with depth set to minimum."""
        with sample_and_reset(self.model, {"*.depth": min}):
            flops = float(inference_flops(self.model, dummy_input=self.dummy_input, unit=1.0))
        return flops

    def _get_true_latency(self, _: Optional[ConstraintsRes] = None) -> float:
        """Get true inference latency."""
        from modelopt.torch._deploy import get_latency

        assert self.deployment is not None, "Deployment config must be provided for latency!"
        return get_latency(self.model, self.dummy_input, self.deployment)

    def _get_latency(self, precomputed: Optional[ConstraintsRes] = None) -> float:
        """Get inference latency from interpolator."""
        return self._latency_interpolator(precomputed)

    @property
    def constraint_eval_funcs(self) -> dict[str, ConstraintEvalFunc]:
        """Get constraint eval fns."""
        return {
            "params": self._get_params,
            "flops": self._get_flops,
            "flops_min_depth": self._get_flops_min_depth,
            "latency": self._get_latency,
        }

    def __call__(self) -> tuple[bool, ConstraintsRes]:
        return self.evaluate_constraints()

    def evaluate_constraints(self) -> tuple[bool, ConstraintsRes]:
        """Check if sub-net satisfied all provided constraints.

        Returns: A tuple (is_satisfied, vals) where
            is_satisfied is an indicator (bool) whether all constraints are satisfied,
            vals is a dictionary containing all evaluated constraints up to 1st unsatisfied constraint.

        The sub-net is profiled after exporting it in eval mode without BN.
        """

        def check_constraint(val: float, limits: LimitsTuple) -> bool:
            """Check if val falls within limits. `None` indicates open-ended intervals."""
            return val >= limits[0] and val <= limits[1]

        is_satisfied = True
        vals: ConstraintsRes = {}

        if dist.is_master(group=self.dp_group):
            # construct sub-net without DDP and BN for more accurate profiling
            model = unwrap_model(self.model)

            # put model into eval mode
            is_training = model.training
            model.eval()

            # check constraint (with early stopping if not satisfied)
            # follow the order of the constraint_eval_funcs
            for criterion, score_func in self.constraint_eval_funcs.items():
                if criterion not in self.limits:
                    continue
                val = score_func(vals)
                vals[criterion] = val
                is_satisfied &= check_constraint(val, self.limits[criterion])

            # put back into original mode
            model.train(is_training)

        is_satisfied, vals = dist.broadcast([is_satisfied, vals])
        return is_satisfied, vals


class StatisticsEstimator:
    """A class to randomly sample subnets & estimate statistics for the provided constraints.

    The class can:
        Estimate median and standard deviation of the constraints
        Plot the distribution of constraints

    Args:
        constraints_func: An object of class ``ConstraintsFunc`` used to define and evaluate the
            constraints.
        num_of_samples: An integer that specifies the number of random samples of model
            used to estimate the constraint statsitics.
    """

    def __init__(
        self,
        constraints_func: ConstraintsFunc,
        num_of_samples: int = 100,
    ) -> None:
        self.constraints_func = constraints_func
        self.num_of_samples = num_of_samples
        self._sample_dict = {}
        self.reset()

    def reset(self):
        """Resets statistics estimator history"""
        self._sample_dict = {k: [] for k in self.constraints_func.limits.keys()}

    @property
    def num_of_collected_samples(self):
        return len(next(iter(self._sample_dict.values())))

    def _sample_random_subnets(self, model) -> dict:
        with random._deterministic_seed():
            max_num_samples = self.num_of_samples if is_configurable(model) else 1
            for _ in range(self.num_of_collected_samples, max_num_samples):
                sample(model)
                info = self.constraints_func()[1]
                for constraint in self.constraints_func.limits.keys():
                    self._sample_dict[constraint].append(info[constraint])

        return self._sample_dict

    def median(self, model: nn.Module) -> dict[str, float]:
        """Estimates median constraint values for the model."""
        median_dict = {}
        sample_dict = self._sample_random_subnets(model)
        for constraint, constraint_samples in sample_dict.items():
            median_dict[constraint] = float(np.median(constraint_samples))
        return median_dict

    def std(self, model: nn.Module) -> dict[str, float]:
        """Estimates std constraint values for the model."""
        std_dict = {}
        sample_dict = self._sample_random_subnets(model)
        for constraint, constraint_samples in sample_dict.items():
            std_dict[constraint] = float(np.std(constraint_samples))
        return std_dict

    def median_subnet(self, model: nn.Module) -> dict:
        """Estimates a median/representative subnet for the model.

        Returns model nas.config dict
        """
        raise NotImplementedError


# Mapping module type to a specific constraint function. Plugins may change this mapping.
MODULE_TYPE_TO_CONSTRAINTS_FUNC = {}


def get_constraints_func(model, *args, **kwargs) -> ConstraintsFunc:
    """Getting the standard constraints function or a special one from the plugins."""
    # Initialize special constraints functor if MODULE_TYPE_TO_CONSTRAINTS_FUNC is not empty.
    for module_type, func in MODULE_TYPE_TO_CONSTRAINTS_FUNC.items():
        if isinstance(model, module_type):
            return func(model, *args, **kwargs)
    # Initialize the standard constraints functor
    return ConstraintsFunc(model, *args, **kwargs)
