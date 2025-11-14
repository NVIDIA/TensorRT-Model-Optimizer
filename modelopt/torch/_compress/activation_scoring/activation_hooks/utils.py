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
import re

from modelopt.torch._compress.activation_scoring.activation_hooks import hooks
from modelopt.torch._compress.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM


def register_activation_hooks(
    model: DeciLMForCausalLM, activation_hooks_kwargs: dict
) -> tuple[dict[str, hooks.ActivationsHook], hooks.ActivationsHook]:
    hook_class_map = {
        "mlp.down_proj": {
            "independent": hooks.IndependentChannelContributionHook,
            "iterative": hooks.IterativeChannelContributionHook,
        },
        "mlp": {"contraction_metrics": hooks.MlpHook},
        "block": {"contraction_metrics": hooks.BlockHook},
        "actual_block": {
            "contraction_metrics": hooks.BlockHook,
            "io_correlation_metrics": hooks.IOCorrelationBlockHook,
        },
        "self_attn.o_proj": {
            "independent_kv_head_contribution": hooks.IndependentKvHeadContributionHook,
        },
        "router": {
            "stats": hooks.RouterStatsHook,
            "num_active_experts": hooks.RouterNumActiveExpertsStatsHook,
            "num_active_experts_unshuffled": hooks.RouterNumActiveExpertsStatsHookUnshuffled,
            "entropy": hooks.RouterEntropyHook,
            "ranked_choice_voting": hooks.RankedChoiceVotingHook,
        },
        r"regex:experts\.\d+\.down_proj$": {  # For MoE
            "independent": hooks.IndependentChannelContributionHook,
        },
        # TODO: maybe this is too generic, and we should have it specifically for
        # input_layernorm and post_attention_layernorm; now it might select qk_norms
        "layernorm": {
            "layer_norm_contribution": hooks.LayerNormlContributionHook,
        },
    }

    activation_hooks = {}
    target_layer = activation_hooks_kwargs.get("target_layer", "mlp.c_proj")

    if target_layer.startswith("regex:"):
        target_layer_regex = target_layer[len("regex:") :]
        pattern = re.compile(target_layer_regex)

        def match_predicate(module_name, module):
            return pattern.search(module_name)
    else:

        def match_predicate(module_name, module):
            return module_name.endswith(target_layer)

    target_layer_hooks_map = hook_class_map.get(target_layer)
    if target_layer_hooks_map is None:
        raise ValueError(f"no hook classes found for: {target_layer}")

    hook_class = target_layer_hooks_map.get(activation_hooks_kwargs["method"])
    if hook_class is None:
        raise ValueError(f"Unknown hook class: {hook_class}")

    if target_layer == "block":
        pattern = re.compile(r"^transformer\.h\.\d+$")

        def match_predicate(module_name, module):
            return pattern.match(module_name)

    activation_hooks_kwargs["model"] = model
    for module_name, module in model.named_modules():
        if match_predicate(module_name, module):
            block_config = None
            if block_idx_match := re.search(r"\.(\d+)\.", module_name):
                block_idx = int(block_idx_match.group(1))
                block_config = model.config.block_configs[block_idx]
            curr_activation_hooks_kwargs = {
                **activation_hooks_kwargs,
                "block_config": block_config,
            }

            hook = hook_class(module, curr_activation_hooks_kwargs)
            module.register_forward_hook(hook)
            activation_hooks[module_name] = hook

            # TODO: CHECK IF WE NEED THIS FOR OTHER CASES THEN SCOUT MOE
            #
            # if ".experts" in module_name:
            #     moe_module_name = module_name[:module_name.index(".experts")]
            #     moe_module = model.get_submodule(moe_module_name)
            #     if hasattr(moe_module, "router"):
            #         router_module_name = moe_module_name + ".router"
            #         if router_module_name not in activation_hooks:
            #             router = moe_module.router
            #             router_hook = hooks.RouterStatsHook(router, {})
            #             router.register_forward_hook(router_hook)
            #             activation_hooks[moe_module_name + ".router"] = router_hook

    # Hook state loading is now handled by the checkpoint manager
    # if len(activation_hooks) == 0:
    #     raise ValueError(f"couldn't find any hooks for {target_layer} ")
    return activation_hooks, hook_class
