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

"""Module for advanced quantization algorithms."""

import fnmatch
import gc
import types
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import regex as re
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.hparam import CustomHPType, Hparam, HPType
from modelopt.torch.opt.searcher import LPS, BaseSearcher, SearchConfig, SearchStateDict
from modelopt.torch.opt.utils import get_hparam, named_hparams
from modelopt.torch.utils import create_param_grad_clear_hook, print_rank_0, report_memory
from modelopt.torch.utils.distributed import DistributedProcessGroup, is_master

from . import config as mtq_config
from . import model_calib
from .config import QuantizeConfig, QuantizerAttributeConfig
from .conversion import set_quantizer_by_cfg
from .nn import QuantLinearConvBase, QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import is_quantized_linear, multi_context


def estimate_quant_compression(quant_cfg: QuantizeConfig) -> float:
    """Estimate the compression ratio of a quantization configuration.

    Right now, we find the minimum compression ratio across all quantizer attribute configs.
    This is not perfect but is a good proxy for the overall compression ratio. We will improve
    this in future releases.

    Args:
        quant_cfg: The quantization configuration to estimate compression for.

    Returns:
        float: The estimated compression ratio (0.0 to 1.0).
    """

    def estimate_quant_compression_for_quantizer(quantizer_attr_cfg):
        if isinstance(quantizer_attr_cfg, list):
            return min(estimate_quant_compression_for_quantizer(q) for q in quantizer_attr_cfg)
        if isinstance(quantizer_attr_cfg, dict):
            return estimate_quant_compression_for_quantizer(list(quantizer_attr_cfg.values()))

        if isinstance(quantizer_attr_cfg, QuantizerAttributeConfig):
            if not quantizer_attr_cfg.enable:
                return 1.0
            if not hasattr(quantizer_attr_cfg, "num_bits"):
                return 1.0
            if isinstance(quantizer_attr_cfg.num_bits, tuple):
                return (sum(quantizer_attr_cfg.num_bits) + 1) / 16
            elif isinstance(quantizer_attr_cfg.num_bits, int):
                return quantizer_attr_cfg.num_bits / 16
            else:
                raise ValueError(f"Unknown quantization config {quantizer_attr_cfg.num_bits}")

        raise ValueError(f"Unknown type {type(quantizer_attr_cfg)}, {quantizer_attr_cfg}")

    return estimate_quant_compression_for_quantizer(list(quant_cfg.quant_cfg.values()))


class QuantRecipe(CustomHPType):
    """A subclass of QuantizeConfig enabling auto_quantize specific configurations.

    Args:
        quant_cfg: str or dict or None. dict is used for custom quantization formats.
        name: name for custom quantization formats. Only used if quantization format is a custom
            format not available in :mod:`modelopt.torch.quantization.config`.
    """

    def __init__(self, quant_cfg: str | dict[str, Any] | None = None, name: str | None = None):
        """Initialize the QuantRecipe with the quantization configuration."""
        name = self.get_auto_name_for_config(quant_cfg) or name

        if quant_cfg is None:
            quant_cfg = {"quant_cfg": {"*": {"enable": False}}}
        elif isinstance(quant_cfg, str):
            assert hasattr(mtq_config, quant_cfg), f"Unknown quantization format {quant_cfg}"
            quant_cfg = getattr(mtq_config, quant_cfg)
        else:
            assert name is not None, "name must be provided for custom quantization formats"

        self.config = mtq_config.QuantizeConfig(**quant_cfg)  # type: ignore [arg-type]

        # Disable KV Cache quantization
        # Currently KV Cache quantization is enabled for some quantization formats and disabled for others
        # This breaks the monotonicity of the quantization formats in terms of weight compression Vs accuracy
        self.config.quant_cfg["*output_quantizer"] = mtq_config.QuantizerAttributeConfig(
            enable=False
        )

        self.compression = estimate_quant_compression(self.config)

        self._str_repr: str = f"{name}(effective-bits: {self.compression * 16})"

    @staticmethod
    def get_auto_name_for_config(quant_cfg: str | dict[str, Any] | None) -> str | None:
        """Get a name for the quantization configuration."""
        if quant_cfg is None:
            return "NONE"
        if isinstance(quant_cfg, str):
            return quant_cfg
        for quant_cfg_name in mtq_config.choices:
            if quant_cfg == getattr(mtq_config, quant_cfg_name):
                return quant_cfg_name
        return None

    @property
    def num_bits(self) -> int:
        """Get the number of bits for the quantization format."""
        return int(self.compression * 16)

    def __str__(self) -> str:
        return self._str_repr

    def __repr__(self) -> str:
        return self._str_repr

    def __lt__(self, other: "QuantRecipe"):
        return self.compression < other.compression

    def __eq__(self, other: object):
        assert isinstance(other, QuantRecipe)
        return self._str_repr == other._str_repr

    def __hash__(self) -> int:
        return hash(self._str_repr)

    @staticmethod
    def disable_folding_pqs_to_weights():
        """Disable the folding of pre_quant_scale to weights."""
        model_calib._ENABLE_FOLDING_PQS_TO_WEIGHTS = False

    @staticmethod
    def fold_pqs_to_weights(model):
        """Fold the pre_quant_scale in weight_quantizers to weights."""
        model_calib._ENABLE_FOLDING_PQS_TO_WEIGHTS = True
        for name, module in model.named_modules():
            if is_quantized_linear(module):
                with SequentialQuantizer.convert_to_single_quantizer(model):
                    if module.weight_quantizer.pre_quant_scale is not None:
                        weight_pqs = module.weight_quantizer.pre_quant_scale
                        delattr(module.weight_quantizer, "_pre_quant_scale")
                        model_calib._apply_weight_pre_quant_scale(module, weight_pqs)


class QuantRecipeHparam(Hparam):
    """An Hparam for quantization recipes.

    In addition, this Hparam also:
    1. Keeps a link to its modules and sets the quantizers for the module based on the active recipe.
    2. Keeps track of the importance of each recipe in a dict instead of a tensor
    """

    def __init__(
        self,
        choices: Sequence[QuantRecipe] | None = None,
        nn_modules: list[nn.Module] | None = None,
    ) -> None:
        """Initializes Hparam with original value and choices."""
        choices = sorted({*(choices if choices else []), QuantRecipe(quant_cfg=None)})
        super().__init__(choices, original=choices[0])
        self.nn_modules = nn_modules if nn_modules else []

        # This is a hack; We dont want to make the input_quantizer, weight_quantizer, output_quantizer
        # a dynamic attribute for backward compatibility with the model_calib.py
        # TODO: Make input_quantizer, weight_quantizer, output_quantizer a dynamic attribute and get rid of this hack
        self._all_quantizer_choices = {quant_recipe: {} for quant_recipe in self.choices}

        quant_recipe: QuantRecipe
        for quant_recipe in self.choices:
            for nn_module in self.nn_modules:
                for quantizer_attr_name in [
                    "input_quantizer",
                    "weight_quantizer",
                    "output_quantizer",
                ]:
                    setattr(nn_module, quantizer_attr_name, TensorQuantizer())

                set_quantizer_by_cfg(nn_module, quant_recipe.config.quant_cfg)
                self._all_quantizer_choices[quant_recipe][nn_module] = {
                    quantizer_attr_name: getattr(nn_module, quantizer_attr_name)
                    for quantizer_attr_name in [
                        "input_quantizer",
                        "weight_quantizer",
                        "output_quantizer",
                    ]
                }

        self.active = self.original

        self._importance_dict = {
            quant_recipe: dict.fromkeys(self.nn_modules, 0.0) for quant_recipe in self.choices
        }

    @property
    def active(self) -> HPType:
        """Return the currently active value."""
        return self._active

    @active.setter
    def active(self, val: HPType | None):
        """Set the active value with a sanity check for choices and dynamic hparams."""
        val = self.original if val is None else val
        assert val in self._choices, f"val = {val}, choices = {self.choices}"
        if self.is_configurable:
            self._active = val
        else:
            assert self._active == val

        for nn_module, quantizer_choices in self._all_quantizer_choices[val].items():
            for quantizer_attr_name, quantizer in quantizer_choices.items():
                setattr(nn_module, quantizer_attr_name, quantizer)

    @property
    def importance(self) -> dict:
        """Return the importance dict mapping recipe and importance."""
        return {
            quant_recipe: sum(importance_dict.values())
            for quant_recipe, importance_dict in self._importance_dict.items()
        }


class AutoQuantizeSearcher(BaseSearcher):
    """A searcher for AutoQuantize algorithm.

    In AutoQuantize, we search for the best per-layer quantization configuration that minimizes the sum of per-layer
    scores while meeting the specified constraint. AutoQuantize uses Linear Programming Solver to find the
    optimal quantization configuration.

    The auto_quantize score for a layer quantization configuration is an approximation of model loss change change due
    to quantizing the particular layer with the particular configuration.
    The approximation is based on taylor expansion of the loss function wrt to the quantized output of the layer and
    substitution of Fisher information for Hessian.
    This approximation is mathematically correct for models where the loss
    is a log likelihood loss such as BERT, GPT, etc. However, the auto_quantize score can still be used as a proxy
    for other models such as ResNet.
    """

    candidate_stats: dict[str, dict[str, list[float]]]
    best: dict[str, Any]
    gradient_checkpointing_enable_contexts: list[tuple[Callable, Callable]] = []

    rules = [
        r"^(.*?)\.(q_proj|k_proj|v_proj)$",  # q_proj, k_proj, v_proj for llama like models
        r"^(.*?)\.(gate_proj|up_proj)$",  # gate_proj, up_proj for llama like models
        r"^(.*?)\.(\d+\.(w1|w2|w3))$",  # mixtral experts
        r"^(.*?)\.((w1_linear|w2_linear|w3_linear)\.\d+)$",  # dbrx experts
    ]

    @property
    def default_search_config(self):
        """Get the default config for the searcher."""
        return {
            "quantization_formats": ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
            "data_loader": None,
            "forward_step": None,
            "loss_func": None,
            "forward_backward_step": None,
            "num_calib_steps": 512,
            "num_score_steps": 128,
            "deployment": None,
            "disabled_layers": None,
            "verbose": is_master(),
            "checkpoint": None,
        }

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Get the default state dict for AutoQuantize."""
        return {
            "candidate_stats": defaultdict(dict),
            "best": {"recipe": {}, "constraints": {}, "score": float("inf"), "is_satisfied": False},
            "constraints": {},
        }

    def sanitize_search_config(self, config: SearchConfig | None) -> SearchConfig:
        """Sanitize the search config dict."""
        config = config or {}
        if "score_func" in config:
            warnings.warn("`score_func` is ignored for `auto_quantize`.")
            config.pop("score_func")
        config = super().sanitize_search_config(config)
        assert config["data_loader"] is not None, (
            "`data_loader` must be provided for `auto_quantize`."
        )
        assert config["forward_step"] is not None, (
            "`forward_step` must be provided for `auto_quantize`."
        )

        if config["forward_backward_step"] is None:
            assert config["loss_func"] is not None, (
                "`loss_func` or `forward_backward_step` must be provided for `auto_quantize`."
            )
            config["forward_backward_step"] = self._get_default_forward_backward_step()

        return config

    @staticmethod
    def _is_auto_quantize_module(module):
        return (
            is_quantized_linear(module) or isinstance(module, QuantLinearConvBase)
        ) and isinstance(module, QuantModule)

    @staticmethod
    def _get_search_recipes(quantization_formats):
        return sorted(
            {
                QuantRecipe(quant_cfg=q[0], name=q[1])
                if isinstance(q, tuple)
                else QuantRecipe(quant_cfg=q)
                for q in quantization_formats
            }
        )

    @classmethod
    def register_gradient_checkpointing_enable_context(
        cls, is_supported_checker: Callable, context: Callable
    ):
        """Register a gradient checkpointing enable context for `AutoQuantize` score estimation.

        If the `is_supported_checker(model)` returns True, the `context(model)` will be used to enable gradient
        checkpointing.
        """
        cls.gradient_checkpointing_enable_contexts.append((is_supported_checker, context))

    def _get_default_forward_backward_step(self):
        def forward_backward_step(model, data):
            output = self.config["forward_step"](model, data)
            loss = self.config["loss_func"](output, data)
            try:
                loss.backward()
            except RuntimeError as e:
                raise RuntimeError(
                    "AutoQuantize: Error while calling `backward()` on the loss returned by `loss_func`. "
                    "Please fix this!"
                ) from e

        return forward_backward_step

    @torch.enable_grad()
    def _estimate_auto_quantize_scores(self):
        # TODO: remove the no-quant recipe
        def auto_quantize_score_estimate_forward(module, input, *args, **kwargs):
            module.quant_recipe = QuantRecipe(quant_cfg=None)
            output = module._forward_original(input, *args, **kwargs)

            # If gradient checkpointing is enabled, gradient will not be enabled in the global forward pass.
            # With gradient checkpointing, gradients are computed in the local forward pass during backward pass

            # Lets compute the output_diff and save it in memory only if gradient is enabled to be memory efficient
            if not torch.is_grad_enabled():
                return output

            module.output_diff_dict = {}
            with torch.no_grad():
                for recipe in module.get_hparam("quant_recipe").choices:
                    if recipe.compression >= 1.0:
                        continue
                    module.quant_recipe = recipe
                    output_diff = module._forward_original(input, *args, **kwargs)

                    if isinstance(output_diff, tuple):
                        output_diff = output_diff[0] - output[0]
                    else:
                        output_diff -= output
                    module.output_diff_dict[recipe] = output_diff

            return output

        def backward_hook(module, grad_input, grad_output):
            for recipe, output_diff in module.output_diff_dict.items():
                score = ((grad_output[0].float() ** 2) * (output_diff.float() ** 2)).sum()
                module.get_hparam("quant_recipe")._importance_dict[recipe][module] += score.item()
                module.output_diff_dict[recipe] = None

            del module.output_diff_dict

        def setup_params_for_score_estimation(name, param, params_metadata):
            # Let us delete the gradient as soon as they are computed to save memory
            # In addition, this method enables gradient for all parameters
            # This is needed to make sure the re-entrant activation checkpointing works
            params_metadata[name] = {"requires_grad": param.requires_grad}
            param.requires_grad = True
            accum_grad, handle = create_param_grad_clear_hook(param)
            params_metadata[name]["accum_grad"] = accum_grad  # We need to keep the accum_grad alive
            params_metadata[name]["handle"] = handle

        def setup_module_for_score_estimation(module):
            module._forward_original = module.forward
            module.forward = types.MethodType(auto_quantize_score_estimate_forward, module)
            module._backward_hook_handle = module.register_full_backward_hook(backward_hook)

        def cleanup_module_after_score_estimation(module):
            module.forward = module._forward_original
            del module._forward_original

            module._backward_hook_handle.remove()

        def cleanup_params_after_score_estimation(name, param, params_metadata):
            param.requires_grad = params_metadata[name]["requires_grad"]
            params_metadata[name]["handle"].remove()

        for name, module in self.model.named_modules():
            if (
                self._is_auto_quantize_module(module)
                and module.get_hparam("quant_recipe").is_configurable
            ):
                # Monkey patch the forward methods to cache Y(Q(W), Q(X)) - Y(W,X)
                setup_module_for_score_estimation(module)

        params_metadata = {}
        for name, param in self.model.named_parameters():
            # TODO: Enabling gradient for all parameters is not needed and making backward slow
            # We need to enable gradient only for the the first parameter of the module such as embedding weights
            setup_params_for_score_estimation(name, param, params_metadata)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_memory("AutoQuantize: starting score estimation, ")

        self._run_func(
            self.config["forward_backward_step"],
            num_iters=self.config["num_score_steps"],
            desc="Estimating auto_quantize scores",
        )

        if torch.cuda.is_available():
            report_memory("AutoQuantize: After score estimation")

        for name, module in self.model.named_modules():
            if (
                self._is_auto_quantize_module(module)
                and module.get_hparam("quant_recipe").is_configurable
            ):
                cleanup_module_after_score_estimation(module)

        for name, param in self.model.named_parameters():
            cleanup_params_after_score_estimation(name, param, params_metadata)

        # Delete the params_metadata
        del params_metadata
        gc.collect()

    @classmethod
    def insert_hparams_after_merge_rules(cls, model, quant_recipes, disabled_layers=None):
        """Restrict the search space using the merge rules and insert the hparams for the model."""
        # TRTLLM fuses linear layers such as q_proj, k_proj, v_proj into same layer
        # Hence we need to restrict the search space so that all these layers share the same recipe
        # Lets group the modules based on the rules and insert the same hparam for all the modules in the group

        if disabled_layers is None:
            disabled_layers = []
        elif isinstance(disabled_layers, str):
            disabled_layers = [disabled_layers]

        search_map: dict[str, list[tuple[nn.Module, bool]]] = {}
        for name, module in model.named_modules():
            if not cls._is_auto_quantize_module(module):
                continue

            # Skip layers that match disabled_layers patterns
            disabled = False
            for pattern in disabled_layers:
                if fnmatch.fnmatch(name, pattern):
                    disabled = True
                    break

            prefix = name
            for rule in cls.rules:
                pattern = re.compile(rule)
                match = pattern.match(name)
                if match:
                    prefix = match.group(1)
                    # We support only one rule for matching per module
                    break
            if prefix not in search_map:
                search_map[prefix] = [(module, disabled)]
            else:
                search_map[prefix].append((module, disabled))

        for prefix, module_info_list in search_map.items():
            modules = [module for module, _ in module_info_list]
            disabled = any(disabled for _, disabled in module_info_list)
            hparam = (
                QuantRecipeHparam(None, nn_modules=modules)
                if disabled
                else QuantRecipeHparam(quant_recipes, nn_modules=modules)
            )
            for module in modules:
                module._register_hparam("quant_recipe", hparam)

    def _get_formatted_weight_compression_constraint(self):
        effective_bits = self.constraints["effective_bits"]
        assert effective_bits > 0 and effective_bits <= 16, (
            "effective_bits should be between 0 and 16."
        )
        weight_compression = self.constraints["effective_bits"] / 16.0

        return weight_compression

    def _verify_constraint(self, search_recipes):
        assert self.constraints["effective_bits"] >= search_recipes[0].num_bits, (
            f"The effective_bits {self.constraints['effective_bits']} constraint cannot be lower than the "
            f"num_bits of most aggressive quantization format for this search which is "
            f"{search_recipes[0]} whose num_bits = {search_recipes[0].num_bits}."
        )

    def _run_func(self, func, num_iters=1, desc=""):
        for i, data in tqdm(
            zip(range(num_iters), self.config["data_loader"]),
            desc=desc,
            total=num_iters,
        ):
            func(self.model, data)

    def before_search(self):
        """Prepare the model for search by calibrating the quantizers  and collecting ``AutoQuantize`` score."""
        # Import here to avoid circular import
        from modelopt.torch.quantization.model_quant import calibrate

        super().before_search()

        search_recipes = self._get_search_recipes(self.config["quantization_formats"])
        self._verify_constraint(search_recipes)
        self.insert_hparams_after_merge_rules(
            self.model, search_recipes, self.config["disabled_layers"]
        )

        QuantRecipe.disable_folding_pqs_to_weights()

        # Iterate over the search recipes and calibrate the quantizers for each recipe
        for recipe in search_recipes:
            if recipe.compression >= 1.0:
                continue

            # Lets reduce the number of calibration steps for AWQ since it takes longer
            num_calib_steps = (
                self.config["num_calib_steps"]
                if "awq" not in str(recipe.config.algorithm)
                else max(1, self.config["num_calib_steps"] // 4)
            )

            def forward_loop(model):
                self._run_func(
                    self.config["forward_step"],
                    num_iters=num_calib_steps,
                    desc=f"Calibrating for {recipe}",
                )

            for name, hparam in named_hparams(self.model, configurable=True):
                if not isinstance(hparam, QuantRecipeHparam):
                    continue
                hparam.active = recipe

            # Now calibrate the quantizers for the recipe
            calibrate(
                self.model,
                algorithm=recipe.config.algorithm,
                forward_loop=forward_loop,
            )
            # Calibrate adds a new mode to the model. Since auto_quantize mixes the quantization recipes
            # across layers, lets not save this new mode in the modelopt state.
            # TODO: This is a hack. We need to create a mode for auto_quantize to handle this in a clean way.
            ModeloptStateManager(self.model).state_dict().pop()

        self.model.eval()
        with multi_context(
            *(
                context(self.model)
                for is_supported_checker, context in self.gradient_checkpointing_enable_contexts
                if is_supported_checker(self.model)
            )
        ):
            self._estimate_auto_quantize_scores()

    def run_search(self):
        """Search for the best per-layer quantization configuration and return the best model and configuration.

        AutoQuantize uses Linear Programming Solver to find the optimal quantization configuration which
        minimizes the sum of per-layer auto_quantize scores while meeting the specified constraint.
        """

        def get_total_weight_size(modules):
            return sum(
                (module.weight.numel() if self._is_auto_quantize_module(module) else 0)
                for module in modules
            )

        def _get_constraints_for_search(max_weight_size, lower_bound=None):
            constraints = {
                "weight_size_after_compression": (
                    lower_bound * max_weight_size if lower_bound else lower_bound,
                    max_weight_size,
                )
            }
            return constraints, "weight_size_after_compression"

        verbose = self.config["verbose"]
        assert len(self.constraints) == 1 and "effective_bits" in self.constraints, (
            f"`constraints` must contain only 'effective_bits' constraint. "
            f"Got {self.constraints.keys()}"
        )

        compression = self._get_formatted_weight_compression_constraint()
        total_weight_size = get_total_weight_size(self.model.modules())
        weight_size_after_compression = total_weight_size * compression

        for name, hparam in named_hparams(self.model, unique=True):
            if not isinstance(hparam, QuantRecipeHparam):
                continue

            formats, scores, costs = [], [], []
            prev_score = float("inf")
            for recipe in hparam.choices:
                formats.append(recipe)
                score = hparam.importance[recipe]
                cost = get_total_weight_size(hparam.nn_modules) * recipe.compression  # type: ignore [union-attr]

                # Lets get the score across Data Parallel (DP) and Tensor Parallel (TP) groups
                # This way we constraint the same quantization format for the same layer across the DP/TP groups
                # The cost we use here is weight size. They are the same across DP/TP groups.
                _ps = self.model.get_submodule(name.split(".quant_recipe")[0]).parallel_state
                # The score is the sum of the scores across DP and TP groups
                score = DistributedProcessGroup.get_dist_syncd_obj(
                    score, [_ps.data_parallel_group, _ps.tensor_parallel_group], sum
                )

                scores.append(min(score, prev_score))
                costs.append(cost)
                prev_score = score

            self.candidate_stats[name]["formats"] = formats
            self.candidate_stats[name]["scores"] = scores
            self.candidate_stats[name]["costs"] = costs

        for lower_bound in [None, 0.99, 0.90]:
            # The LP solver for auto_quantize sometimes fails to find a solution if a lower bound is not
            # specified. I dont know why this happens.
            # As a workaround, lets specify a lower bound for the weight compression if previous
            # search without lower bound fails.
            constraints, constraint_name = _get_constraints_for_search(
                weight_size_after_compression, lower_bound
            )

            lps = LPS(
                name="AutoQuantize",
                constraints=constraints,
                constraints_to_candidate_costs={
                    constraint_name: [
                        candidate_stat["costs"] for candidate_stat in self.candidate_stats.values()
                    ]
                },
                candidate_scores=[
                    candidate_stat["scores"] for candidate_stat in self.candidate_stats.values()
                ],
                objective_type="minimize",
                verbose=verbose,
            )
            selections, self.status = lps()
            if self.status == "Optimal":
                break

        self.best = {}

        if self.status != "Optimal":
            warnings.warn(
                "AutoQuantize FAILED to find a solution! The searched model might not meet all constraints. "
            )
            self.best["is_satisfied"] = False
        else:
            self.best["is_satisfied"] = True

        best_recipe = {}
        best_constraints, best_scores = 0, 0
        for name, selected_idx in zip(self.candidate_stats.keys(), selections):
            best_recipe_for_name = self.candidate_stats[name]["formats"][selected_idx]

            # LP solver could give different solutions for the same layer across DP/TP groups even though
            # the scores and costs are the same. Lets make sure the same quantization format is selected across DP/TP
            _ps = self.model.get_submodule(name.split(".quant_recipe")[0]).parallel_state
            best_recipe_for_name = DistributedProcessGroup.get_dist_syncd_obj(
                best_recipe_for_name,
                [_ps.data_parallel_group, _ps.tensor_parallel_group],
                lambda a: a[0],
            )

            best_recipe[name] = best_recipe_for_name
            get_hparam(self.model, name).active = best_recipe_for_name
            best_constraints += self.candidate_stats[name]["costs"][selected_idx]
            best_scores += self.candidate_stats[name]["scores"][selected_idx]
            if verbose:
                print_rank_0(
                    f"AutoQuantize best recipe for {name.replace('.quant_recipe', '')}: {best_recipe[name]}"
                )

        effective_bits_from_search = (best_constraints / total_weight_size) * 16
        if verbose:
            print_rank_0(
                f"AutoQuantize effective bits from search: {effective_bits_from_search: .2f}"
            )

        self.best["recipe"] = best_recipe
        self.best["constraints"] = {"effective_bits": effective_bits_from_search}
        self.best["score"] = best_scores

        QuantRecipe.fold_pqs_to_weights(self.model)
