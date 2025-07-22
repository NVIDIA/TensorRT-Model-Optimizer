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

"""User-facing quantization API."""

import fnmatch
import inspect
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.core.torch.quantization.algorithms import AutoQuantizeSearcher
from modelopt.torch.opt import apply_mode
from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.opt.utils import forward_with_reshard
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg

from . import config
from .config import QuantizeAlgoCfgType
from .conversion import set_quantizer_attribute
from .mode import QuantizeModeRegistry, get_modelike_from_algo_cfg
from .nn import QuantModule, TensorQuantizer

__all__ = [
    "auto_quantize",
    "calibrate",
    "disable_quantizer",
    "enable_quantizer",
    "fold_weight",
    "postprocess_amax",
    "print_quant_summary",
    "quantize",
]


# TODO: Descriptors for the supported algorithms
def calibrate(
    model: nn.Module,
    algorithm: QuantizeAlgoCfgType = "max",
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Adjusts weights and scaling factors based on selected algorithms.

    In order to calibrate using custom user defined calibration algorithm, refer to
    :ref:`custom calibration algorithm <custom_calibration_algorithm>`

    Args:
        model: A pytorch model with quantizer modules.
        algorithm: A string or dictionary specifying the calibration algorithm to use. Supported
            algorithms are ``"max"``, ``"smoothquant"``, ``"awq_lite"``, ``"awq_full"``, and
            ``"awq_clip"``. If a dictionary is passed, the key ``"method"`` should specify the
            calibration algorithm to use. Other key-value pairs  in this dictionary will be passed
            as kwargs to the algorithm.

            An example dictionary argument:
            ``{"method": "awq_clip", "max_co_batch_size": 4096}``.

            If ``None``, no calibration is performed.
        forward_loop: A callable which takes the model as argument and forwards calibration data
            through the model. This is not required for weight-only quantization with the ``"max"``
            algorithm.

    Returns: The calibrated pytorch model.
    """
    if forward_loop is not None:
        # get the number of arguments of forward_loop
        num_args = len(inspect.signature(forward_loop).parameters)
        if num_args == 0:
            warnings.warn(
                (
                    "forward_loop should take model as argument, but got forward_loop without any"
                    " arguments. This usage will be deprecated in future versions."
                ),
                DeprecationWarning,
            )
            original_forward_loop = forward_loop

            def forward_loop(model):
                return original_forward_loop()  # type: ignore[call-arg]

    # move the model to eval mode
    is_training = model.training
    model.eval()

    with forward_with_reshard(model):
        apply_mode(
            model,
            mode=get_modelike_from_algo_cfg(algorithm),
            mode_kwargs={"forward_loop": forward_loop},
        )

    # TODO: Re-enable when the CUDA error: unspecified launch failure is fixed.
    # clear_cuda_cache()

    model.train(is_training)

    return model


def postprocess_amax(model: nn.Module, key: str, post_process_fn) -> nn.Module:
    """Experimental API to postprocess the amax values after calibration."""
    assert isinstance(key, str), "key should be a string"
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if not hasattr(module, "_amax"):
            continue
        if not fnmatch.fnmatch(name, key):
            continue
        module.amax = post_process_fn(module.amax)

    return model


def quantize(
    model: nn.Module,
    config: dict[str, Any | QuantizeConfig],
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Quantizes and calibrates the model in-place.

    This method performs replacement of modules with their quantized counterparts and
    performs calibration as specified by ``quant_cfg``.
    ``forward_loop`` is used to forward data through the model and gather statistics for calibration.

    Args:
        model: A pytorch model
        config: A dictionary or an instance of
            :class:`QuantizeConfig <modelopt.torch.quantization.config.QuantizeConfig>` specifying the
            values for keys ``"quant_cfg"`` and ``"algorithm"``.
            It is basically a dictionary specifying the values for keys ``"quant_cfg"`` and ``"algorithm"``.
            The ``"quant_cfg"`` key specifies the quantization configurations.
            The ``"algorithm"`` key specifies the ``algorithm`` argument to
            :meth:`calibrate <modelopt.torch.quantization.model_quant.calibrate>`.

            Quantization configurations is a dictionary mapping wildcards or filter functions
            to its quantizer attributes. The wildcards or filter functions  are matched
            against the quantizer module names. The quantizer modules have names ending with
            ``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
            input quantization (or activation quantization) respectively. The quantizer modules
            are instances of
            :class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
            The quantizer attributes are defined by :class:`QuantizerAttributeConfig`. See
            :class:`QuantizerAttributeConfig` for details on the quantizer attributes and their values.

            An example ``config`` dictionary is given below:

            .. code-block::python

                config = {

                    "quant_cfg": {
                        # "num_bits" specifies the number of bits for quantization
                        # "axis" specifies the axis for quantization
                        "*weight_quantizer": {"num_bits": 8, "axis": 0},
                        "*input_quantizer": {"num_bits": 8, "axis": -1},

                        # Default quantization settings
                        "default": {"num_bits": 8, "axis": None},
                    }
                    "algorithm": "max"
                }

            See :ref:`Quantization Formats <quantization-formats>` to learn more about the supported
            quantization formats. See :ref:`Quantization Configs <quantization-configs>` for more details on
            ``config`` dictionary.

        forward_loop: A callable that forwards all calibration data through the model. This is used
            to gather statistics for calibration. It should take model as the argument. It does not need
            to return anything.

            This argument is not required for weight-only quantization with the ``"max"``
            algorithm.

            Here are a few examples for correct ``forward_loop`` definitions:
            Example 1:

            .. code-block::

                    def forward_loop(model) -> None:
                        # iterate over the data loader and forward data through the model
                        for batch in data_loader:
                            model(batch)

            Example 2:

            .. code-block::

                    def forward_loop(model) -> float:
                        # evaluate the model on the task
                        return evaluate(model, task, ....)

            Example 3:

            .. code-block::

                    def forward_loop(model) -> None:
                        # run evaluation pipeline
                        evaluator.model = model
                        evaluator.evaluate()

            .. note::

                Calibration does not require forwarding the entire dataset through the model.
                Please subsample the dataset or reduce the number of batches if needed.

    Returns: A pytorch model which has been quantized and calibrated.
    """
    model = apply_mode(model, mode=[("quantize", config)], registry=QuantizeModeRegistry)
    return calibrate(model, config["algorithm"], forward_loop=forward_loop)


def auto_quantize(
    model: nn.Module,
    constraints: dict[str, float | str] = {"effective_bits": 4.8},
    quantization_formats: list[dict[str, Any] | str] = [
        mtq.NVFP4_AWQ_LITE_CFG,
        mtq.FP8_DEFAULT_CFG,
    ],
    data_loader: Iterable | None = None,
    forward_step: Callable[[nn.Module, Any], Any | torch.Tensor] | None = None,
    loss_func: Callable[[Any, Any], torch.Tensor] | None = None,
    forward_backward_step: Callable[[nn.Module, Any], Any] | None = None,
    disabled_layers: list[str] | str | None = None,
    num_calib_steps: int = 512,
    num_score_steps: int = 128,
    verbose: bool = False,
):
    r"""Perform optimal per-layer quantization by searching for the best quantization formats per-layer.

    ``auto_quantize`` uses a gradient based sensitivity score to rank the per-layer quantization formats and search
    for the best quantization formats per-layer.

    Args:
        model: A pytorch model with quantizer modules.
        constraints: Constraints for the search. Currently we support only ``effective_bits``.
            ``effective_bits`` specifies the effective number of bits for the quantized model.

            Here is an example for valid ``effective_bits`` argument:

            .. code-block:: python

                # For an effective quantization bits of 4.8
                constraints = {"effective_bits": 4.8}

        quantization_formats: A list of quantization format config dictionaries or string names to search for.
            Each config dictionary should be valid as a ``config`` argument in
            :meth:`quantize <modelopt.torch.quantization.model_quant.quantize>`.
            The supported quantization format names are as listed by :attr:`modelopt.torch.quantization.config.choices`.

            Internally we always add "do not quantize" as a choice. Therefore, it is possible that a layer is
            not quantized by any of the quantization formats.

            Custom quantization formats can also be defined and used as a quantization format. This is a  experimental
            feature and the results may not be optimal. Here is an example:

            .. code-block:: python

                INT8_CUSTOM_QUANT_CFG = {
                    "quant_cfg": {
                        "*weight_quantizer": {"num_bits": 8, "axis": 0},
                        "*input_quantizer": {"num_bits": 8, "axis": None},
                    },
                    "algorithm": "smoothquant",
                }

                mtq.auto_quantize(
                    model,
                    constraints,
                    quantization_formats=["INT4_AWQ_CFG", INT8_CUSTOM_QUANT_CFG],
                )

            Internally we always add "do not quantize" as a choice. Therefore, it is possible that a layer is
            not quantized by any of the quantization formats.

            .. note::

                The quantization formats will be applied on a per-layer match basis. The global model level name
                based quantizer attribute setting will be ignored. For example, in ``FP8_DEFAULT_CFG`` quantizer
                configuration the key ``"*lm_head*": {"enable": False}`` disables quantization for the ``lm_head``
                layer. However in ``auto_quantize``, the quantization format for the ``lm_head`` layer will be searched.
                This is because the key ``"*lm_head*"`` sets the quantizer attributes based on the global model level
                name, not per-layer basis. The keys ``"*input_quantizer"``, ``"*weight_quantizer"`` etc. in
                ``FP8_DEFAULT_CFG`` match on a per-layer basis  - hence the corresponding quantizers
                will be set as specified.

            Here is an example `quantization_formats` argument:

             .. code-block:: python

                # A valid `quantization_formats` argument
                # This will search for the best per-layer quantization from FP8, W4A8_AWQ_BETA_CFG or No quantization
                quantization_formats = [mtq.FP8_DEFAULT_CFG, mtq.W4A8_AWQ_BETA_CFG]

        data_loader: An iterator that yields data that is to be used for calibrating quantized layers and estimating
            ``auto_quantize`` scores.
        forward_step: A callable that takes the model and a batch of data from ``data_loader`` as input, forwards
            the data through the model and returns the model output.
            This is a required argument.

            Here is an example for a valid ``forward_step``:

            .. code-block:: python

                # Takes the model and a batch of data as input and returns the model output
                def forward_step(model, batch) -> torch.Tensor:
                    output = model(batch)
                    return output

        loss_func: (Optional) A callable that takes the model output and the batch of data as input and computes the
            loss. The model output is the output given by ``forward_step``. `.backward()` will be called on the loss.

            Here is an example for a valid ``loss_func``:

            .. code-block:: python

                # Takes the model output and a batch of data as input and returns the loss
                def loss_func(output, batch) -> torch.Tensor:
                    ...
                    return loss


                # loss should be a scalar tensor such that loss.backward() can be called
                loss = loss_func(output, batch)
                loss.backward()

            If this argument is not provided, ``forward_backward_step`` should be provided.

        forward_backward_step: (Optional) A callable that takes batch of data from ``data_loader``, forwards it
            through the model, computes the loss and runs backward on the loss.

            Here is an example for a valid ``forward_backward_step`` argument:

            .. code-block:: python

                # Takes the model and a batch of data as input and runs forward and backward pass
                def forward_backward_step(model, batch) -> None:
                    output = model(batch)
                    loss = my_loss_func(output, batch)
                    run_custom_backward(loss)

            If this argument is not provided, ``loss_func`` should be provided.

        disabled_layers: (Optional) One or a list of wildcard strings to disable quantization for the layers. Example:

            .. code-block:: python

                disabled_layers = "*lm_head*"
                disabled_layers = ["*lm_head*", "*mlp*"]

        num_calib_steps: Number of batches to use for calibrating the quantized model. Suggested value is 512.
        num_score_steps: Number of batches to use for estimating ``auto_quantize`` scores. Suggested value is 128.
            A higher value could increase the time taken for performing ``auto_quantize``.
        verbose: If True, prints the search progress/intermediate results.

    Returns: A tuple (model, state_dict) where ``model`` is the searched and quantized model and
        ``state_dict`` contains the history and detailed stats of the search procedure.

    .. note::

        ``auto_quantize`` groups certain layers and restricts the quantization formats for them to be same. For example,
        Q, K, V linear layers belonging to the same transformer layer will have the same quantization format.
        This is to ensure compatibility with TensorRT-LLM which fuses these three linear layers into a single linear
        layer.

        A list of regex pattern rules as defined in :attr:`rules <.algorithms.AutoQuantizeSearcher.rules>`
        are used to specify the group of layers. The first captured group
        in the regex pattern (i.e, ``pattern.match(name).group(1)``) is used to group the layers. All the layers
        that share the same first captured group will have the same quantization format..

        For example, the rule ``r"^(.*?)\.(q_proj|k_proj|v_proj)$"``
        groups the `q_proj`, `k_proj`, `v_proj` linear layers belonging to the same transformer layer.

        You may modify the rules to group the layers as per your requirement.

        .. code-block:: python

            from modelopt.torch.quantization.algorithms import AutoQuantizeSearcher

            # To additionally group the layers belonging to same `mlp` layer,
            # add the following rule
            AutoQuantizeSearcher.rules.append(r"^(.*?)\.mlp")

            # Perform `auto_quantize`
            model, state_dict = auto_quantize(model, ...)

    .. note::

        The ``auto_quantize`` API and algorithm is experimental and subject to change. ``auto_quantize`` searched models
        might not be readily deployable to TensorRT-LLM yet.

    """
    processed_quantization_formats = []
    for i, quant_cfg in enumerate(quantization_formats):
        if quant_cfg is None:
            continue
        if isinstance(quant_cfg, str):
            assert quant_cfg in config.choices, f"Invalid quantization format: {quant_cfg}"
            quant_cfg = getattr(config, quant_cfg)
        elif not any(quant_cfg is getattr(config, choice) for choice in config.choices):
            warnings.warn(
                "Received custom quantization formats for search, auto_quantize results may not be optimal."
            )
        processed_quantization_formats.append(quant_cfg)

    assert len(processed_quantization_formats) > 0, "`quantization_formats` should not be empty"
    model = apply_mode(
        model,
        mode="auto_quantize",
        registry=QuantizeModeRegistry,
    )
    searcher = AutoQuantizeSearcher()
    search_config = {
        "quantization_formats": processed_quantization_formats,
        "data_loader": data_loader,
        "forward_step": forward_step,
        "loss_func": loss_func,
        "forward_backward_step": forward_backward_step,
        "num_calib_steps": num_calib_steps,
        "num_score_steps": num_score_steps,
        "verbose": verbose,
    }
    # Disable all quantizers; AutoQuantize will enable the needed ones
    set_quantizer_by_cfg(model, {"*": {"enable": False}})
    searcher.search(model, constraints, config=search_config)  # type: ignore[arg-type]

    if disabled_layers:
        if isinstance(disabled_layers, str):
            disabled_layers = [disabled_layers]
        for layer_pattern in disabled_layers:
            disable_quantizer(model, layer_pattern)

    return model, searcher.state_dict()


def disable_quantizer(model: nn.Module, wildcard_or_filter_func: str | Callable):
    """Disable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": False})


def enable_quantizer(model: nn.Module, wildcard_or_filter_func: str | Callable):
    """Enable quantizer by wildcard or filter function."""
    set_quantizer_attribute(model, wildcard_or_filter_func, {"enable": True})


def print_quant_summary(model: nn.Module):
    """Print summary of all quantizer modules in the model."""
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, TensorQuantizer):
            print(f"{name:80} {mod}")
            count += 1
    print(f"{count} TensorQuantizers found in model")


def fold_weight(model: nn.Module):
    """Fold weight quantizer for fast evaluation."""
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            module.fold_weight()
