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
from collections import OrderedDict
from typing import Any, Optional, Union

import torch
from rich.console import Console
from rich.table import Table
from torch import nn

from modelopt.torch.opt.conversion import get_mode
from modelopt.torch.opt.searcher import ConstraintsDict, Deployment, SearchConfig
from modelopt.torch.opt.utils import is_configurable, search_space_size
from modelopt.torch.utils import clear_cuda_cache, num2hrb

from ._algorithms import ConstraintsFunc, ConstraintsRes, StatisticsEstimator, get_constraints_func
from .conversion import export
from .search_space import SampleFunc
from .utils import (
    get_subnet_config,
    no_modelopt_patches,
    print_search_space_summary,
    sample_and_reset,
)

__all__ = ["profile", "search"]


def search(
    model: nn.Module,
    constraints: ConstraintsDict,
    dummy_input: Union[Any, tuple],
    config: Optional[SearchConfig] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Search a given prunable model for the best sub-net and return the search model.

    The best sub-net maximizes the score given by ``score_func`` while satisfying the
    ``constraints``.

    Args:
        model: The converted model to be searched.
        constraints: The dictionary from constraint name to upper bound the searched model has to satisfy.
            Currently, we support ``flops`` and ``params`` as constraints.
            The constraints dictionary generally takes the following form:

            .. code-block:: python

                constraints = {"params": 5.0e6, "flops": 4.5e8}

            We recommend to simply provide the most relevant constraint, e.g., flops:

            .. code-block:: python

                constraints = {"flops": 4.5e8}

            You can also provide a percentage value instead of absolute value, e.g.,

            .. code-block:: python

                # search for a model with <= 60% of the original model flops
                constraints = {"flops": "60%"}

        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        config: Additional optional arguments to configure the search. Currently, we support:

            * ``checkpoint``: Path to save/restore checkpoint with dictionary containing intermediate
              search state. If provided, the intermediate search state will be automatically
              restored before search (if exists) and stored/saved during search.
            * ``verbose``: Whether to print detailed search space profiling and search stats during search.
            * ``forward_loop``: A ``Callable`` that takes a model as input and runs a forward loop
              on it. It is recommended to choose the data loader used inside the forward loop
              carefully to reduce the runtime. Cannot be provided at the same time as
              ``data_loader`` and ``collect_func``.
            * ``data_loader``: An iterator yielding batches of data for calibrating the
              normalization layers in the model or compute gradient scores. It is recommended to use
              the same data loader as for training but with significantly fewer iterations. Cannot
              be provided at the same time as ``forward_loop``.
            * ``collect_func``: A ``Callable`` that takes a batch of data from the data loader as
              input and returns the input to ``model.forward()`` as described in
              :meth:`run_forward_loop <modelopt.torch.utils.network.run_forward_loop>`. Cannot
              be provided at the same time as ``forward_loop``.
            * ``max_iter_data_loader``: Maximum number of iterations to run the data loader.
            * ``score_func``: A callable taking the model as input and returning a single accuracy/score
              metric (float). This metric will be maximized during search.

                .. note::

                    The ``score_func`` is required for ``autonas`` and ``fastnas`` modes. It will be
                    evaluated on models in eval mode (``model.eval()``).
            * ``loss_func``: A ``Callable`` which takes the model output (i.e output of ``model.forward()``)
              and the batch of data as its inputs and returns a scalar loss.
              This is a required argument if the model is converted via ``gradnas`` mode.

              It should be possible to run a backward pass on the loss value returned by this method.

              ``collect_func`` will be used to gather the inputs to ``model.forward()``
              from a batch of data yielded by``data_loader``.

              ``loss_func`` should support the following usage:

                .. code-block:: python

                    for i, batch in enumerate(data_loader):
                        if i >= max_iter_data_loader:
                            break

                        # Assuming collect_func returns a tuple of arguments
                        output = model(*collect_func(batch))

                        loss = loss_func(output, batch)
                        loss.backward()

            .. note::

                Additional configuration options may be added by individual algorithms. Please
                refer to the documentation of the individual algorithms for more information.

    Returns: A tuple (subnet, state_dict) where
        subnet is the searched subnet (nn.Module), which can be used for subsequent tasks like
        fine-tuning, state_dict contains the history and detailed stats of the search procedure.

    .. note::

        The given model is modified (exported) in-place to match the best subnet found by the
        search algorithm. The returned subnet is thus a reference to the same model instance as the
        input model.
    """
    # retrieve searcher class
    mode = get_mode(model)
    assert mode is not None, "Model is not converted to a search space. Please convert first."
    searcher_cls = mode.search_algorithm

    # run search, select best, and get config, metric, constraint of search result
    searcher = searcher_cls()

    searcher.search(model, constraints, dummy_input, config)

    # export model in-place
    model = export(model)

    clear_cuda_cache()

    # return the optimized & exported model and searcher's state_dict (detailed stats)
    return model, searcher.state_dict()


@torch.no_grad()
@no_modelopt_patches()
def profile(
    model: nn.Module,
    dummy_input: Optional[Union[Any, tuple]] = None,
    constraints: Optional[Union[ConstraintsDict, ConstraintsFunc]] = None,
    deployment: Optional[Deployment] = None,
    strict: bool = False,
    verbose: bool = True,
    use_centroid: bool = False,
) -> tuple[bool, dict[str, dict]]:
    """Profile statistics of the search space of a converted model or a regular model.

    Args:
        model: The model to be profiled. Can be converted or not.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        constraints: The dictionary from constraint name to upper bound the searched model has to satisfy.
            Currently, we support ``flops`` and ``params`` as constraints.
            The constraints dictionary generally takes the following form:

            .. code-block:: python

                constraints = {"params": 5.0e6, "flops": 4.5e8}

            .. note::

                We recommend to simply provide the most relevant constraint, e.g., flops:

                .. code-block:: python

                    constraints = {"flops": 4.5e8}

            Note that you can also provide a percentage value instead of absolute value, e.g.,

            .. code-block:: python

                # search for a model with <= 60% of the original model flops
                constraints = {"flops": "60%"}

        strict: Raise an error if constraints are not satisfiable.
        verbose: Print detailed profiling results.
        use_centroid: By default, profile reports median of the evaluation results from
            randomly sampled subnets (instead of the evaluation result from deterministic
            centroid subnet). Set use_centroid to True to use the deterministic centroid for
            profiling.

    Returns: A tuple (is_all_sat, stats) where
        is_all_sat is a bool indicating whether all constraints can be satisfied.
        stats is a dictionary containing statistics for the search space if the model is converted,
        e.g., the FLOPs and params for the min, centroid, max subnets and their max/min ratios,
        size of the search space, number of configurable hparams.
    """
    # pre-fetch some low-level properties properties of the search space
    is_configurable_ = is_configurable(model)
    size_of_search_space = search_space_size(model)
    num_configurable = len(get_subnet_config(model, configurable=True))

    # custom print function that tracks print statements
    detailed_msg = ""

    def _print(msg: str, verb: bool = verbose):
        nonlocal detailed_msg
        if detailed_msg:
            detailed_msg += "\n"
        detailed_msg += msg
        if verb:
            print(msg)

    # construct constraints checker function from constraints dict
    if isinstance(constraints, ConstraintsFunc):
        assert_msg = "No deployment or dummy_input allowed for constraints as ConstraintFunc obj!"
        assert deployment is None and dummy_input is None, assert_msg
        assert constraints.model is model, "Model of constraints does not match the given model!"
        constraints_func = constraints
    else:
        assert dummy_input is not None, "Please provide a dummy_input for the constraints."
        constraints = copy.deepcopy(constraints) or {}
        if deployment is not None and "latency" not in constraints:
            constraints["latency"] = None
        constraints_func = get_constraints_func(model, constraints, dummy_input, deployment, False)

    # retrieve sample points and print message
    options: dict[str, SampleFunc]
    if is_configurable_:
        options = constraints_func.get_sample_points()
        msg = f"Profiling the following subnets from the given model: {tuple(options)}."
    else:
        msg = "Profiling full net, since the model is not a converted model."
        options = OrderedDict([("full", max)])
    _print(msg="\n" + msg + "\n" + "-" * 80)

    # extract limits
    limits = constraints_func.limits

    # sample through some distinct networks and collect some stats
    # note that during that process we definitely don't want fast constraint evaluation!
    fast_eval = constraints_func.fast_eval
    constraints_func.fast_eval = False
    info: dict[str, ConstraintsRes] = {}
    for subnet_key, sample_method in options.items():
        # avoid changing the active subnet before and after nas.profile
        with sample_and_reset(model, sample_method):
            info[subnet_key] = constraints_func()[1]

    # Use statistical median values.
    # it does not have a pre-defined sampling method.
    # To further speed up, we can reduce the number of samples.
    if "centroid" in options and use_centroid is False:
        constraints_func.fast_eval = True
        statistics_estimator = StatisticsEstimator(constraints_func, num_of_samples=30)
        with sample_and_reset(model):
            info["centroid"] = statistics_estimator.median(model)

    constraints_func.fast_eval = fast_eval

    # for each sub net and constraint report the constraint evaluation and where it falls within the
    # limits
    is_all_sat = True

    stats = info.copy()
    if is_configurable_:
        stats["max/min ratio"] = {}

    # options that we also print
    options_keys_for_print = [k for k in options if k in {"min", "centroid", "max", "full"}]

    console = Console()
    with console.capture() as capture:
        table_profile = Table(title="\nProfiling Results")
        table_profile.add_column("Constraint", width=12)

        table_constraints = Table(title="\nConstraints Evaluation")
        table_constraints.add_column("Constraint", width=12)
        table_constraints.add_column("Upper Bound", width=12)
        table_constraints.add_column("Satisfiable Upper Bound", width=12)

        for subnet in options_keys_for_print:
            table_profile.add_column(subnet, width=12)

        if is_configurable_:
            table_profile.add_column("max/min ratio")

        for constraint, (lb, ub) in limits.items():
            print_info = [f"{constraint:8}"]
            is_sat_lower = False
            is_sat_upper = False
            # check constraint satisfaction
            for subnet in options_keys_for_print:
                subnet_info = info[subnet]
                val = subnet_info[constraint]
                is_sat_lower |= val >= lb
                is_sat_upper |= val <= ub
                print_info.append(f"{num2hrb(val)}")

            if is_configurable_:
                # add max/min ratio
                ratio_val = info["max"][constraint] / info["min"][constraint]
                print_info.append(f"{ratio_val:4.2f}")
                stats["max/min ratio"].update({constraint: ratio_val})
            # table_profile.add_row()
            table_profile.add_row(*print_info)

            is_all_sat &= is_sat_lower and is_sat_upper
            # print update
            if constraint in constraints_func.effective_limits:
                print_info = [f"{constraint:8}"]
                print_info.extend([f"{num2hrb(ub)}", f"{str(is_sat_upper):5}"])
                table_constraints.add_row(*print_info)

        console.print(table_profile)
        if len(constraints_func.effective_limits) > 0:
            console.print(table_constraints)

    print_info = str(capture.get())
    _print(print_info)  # type: ignore[arg-type]

    if is_configurable_:
        if verbose:
            print_search_space_summary(model)
        # add seach space info
        _print(f"Number of configurable hparams: {num_configurable}")
        stats["number of configurable hparams"] = num_configurable
        _print(f"Total size of the search space: {size_of_search_space:8.2e}")
        stats["search space size"] = size_of_search_space

    # check if all constraints are satisfiable and raise error if not and strict
    if is_all_sat:
        if constraints_func.effective_limits:
            _print("Note: all constraints can be satisfied within the search space!\n")
    elif is_configurable_:
        warn_msg = "" if verbose else f"\n{detailed_msg}\n"
        warn_msg += "NOT all constraints can be satisfied within the search space, see above!"
        if strict:
            raise ValueError(warn_msg)
        else:
            _print(f"Warning: {warn_msg}\n", verb=True)

    return is_all_sat, stats
