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

"""Conversion and restoration utilities for sparse attention."""

import fnmatch
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict
from modelopt.torch.utils import get_unwrapped_name

from .config import SparseAttentionConfig
from .nn.sparse_attention import SparseAttentionModule, SparseAttentionRegistry
from .plugins.huggingface import register_sparse_attention_on_the_fly


def is_attn_sparsified(model: nn.Module) -> bool:
    """Check if a model has sparse attention applied.

    Similar to quantization's is_quantized for API consistency.

    Args:
        model: Model to check

    Returns:
        True if model contains any SparseAttentionModule instances
    """
    return any(isinstance(module, SparseAttentionModule) for module in model.modules())


def convert_to_sparse_attention_model(
    model: ModelLikeModule, config: SparseAttentionConfig
) -> ConvertReturnType:
    """Convert model to use sparse attention.

    Args:
        model: Model to convert
        config: Sparse attention configuration

    Returns:
        Tuple of (converted_model, metadata)
    """
    # Initialize the true module if necessary
    model = model.init_modellike() if isinstance(model, ModelLikeModule) else model

    # Register sparse attention modules dynamically
    register_sparse_attention_on_the_fly(model)

    # Replace attention modules with sparse versions
    replace_sparse_attention_modules(model, version=ModeloptStateManager(model).state_version)

    # Apply configuration to sparse attention modules
    sparse_cfg = config.sparse_cfg if hasattr(config, "sparse_cfg") else {}
    set_sparse_attention_by_cfg(model, sparse_cfg, config)

    # Create metadata
    metadata = {}
    update_sparse_attention_metadata(model, config, metadata)

    return model, metadata


def replace_sparse_attention_modules(model: nn.Module, version=None):
    """Replace regular attention modules with sparse attention modules.

    Recursively replace all attention modules in the model with their sparse attention counterparts.

    Args:
        model: Model to process
        version: State version for tracking (optional)
    """
    # Recursively replace modules
    _replace_sparse_attention_modules(model, version=version)

    # Count and report replaced modules
    replaced_count = sum(isinstance(m, SparseAttentionModule) for _, m in model.named_modules())
    if replaced_count > 0:
        print(f"Inserted {replaced_count} sparse attention modules")


def _replace_sparse_attention_modules(model: nn.Module, version=None):
    """Helper function for replace_sparse_attention_modules."""
    for name, child in model.named_children():
        if type(child) in SparseAttentionRegistry:
            # REPLACE on the parent (model), not on child
            sparse_module = SparseAttentionRegistry.convert(child)
            setattr(model, name, sparse_module)

        # Now recurse into whichever module is now at `model.name`
        _replace_sparse_attention_modules(getattr(model, name), version=version)


def set_sparse_attention_by_cfg(model: nn.Module, sparse_cfg: dict, config: SparseAttentionConfig):
    """Apply sparse attention configuration to model.

    Similar to quantization's set_quantizer_by_cfg.

    Args:
        model: Model with sparse attention modules
        sparse_cfg: Sparse configuration dictionary
        config: Global sparse attention configuration
    """
    sparse_cfg = sparse_cfg.copy()

    # Apply default first if exists
    if "default" in sparse_cfg:
        set_sparse_attention_attribute(model, "*", sparse_cfg["default"], config)
        sparse_cfg.pop("default")

    # Apply pattern-specific configs
    for pattern, cfg in sparse_cfg.items():
        set_sparse_attention_attribute(model, pattern, cfg, config)


def set_sparse_attention_attribute(
    model: nn.Module,
    wildcard_or_filter: str | Callable,
    attribute_cfg: dict[str, Any],
    global_config: SparseAttentionConfig,
):
    """Set sparse attention attributes for modules matching pattern.

    Similar to quantization's set_quantizer_attribute.

    Args:
        model: Model to configure
        wildcard_or_filter: Pattern to match module names
        attribute_cfg: Attributes to apply
        global_config: Global sparse attention configuration
    """
    # Merge global config fields with pattern config
    # Filter out model-level configs that shouldn't be passed to modules
    module_cfg = {k: v for k, v in attribute_cfg.items() if k != "calibration"}

    full_cfg = {
        "method": global_config.method,
        "collect_stats": global_config.collect_stats,
        **module_cfg,
    }

    for name, module in model.named_modules():
        if not isinstance(module, SparseAttentionModule):
            continue

        # Check pattern match
        matched = False
        if isinstance(wildcard_or_filter, str):
            matched = fnmatch.fnmatch(name, wildcard_or_filter)
        elif callable(wildcard_or_filter):
            matched = wildcard_or_filter(name)
        else:
            continue

        if matched:
            # Apply config using the same method as TensorQuantizer
            module.set_from_attribute_config(full_cfg)


def restore_sparse_attention_model(
    model: ModelLikeModule, config: SparseAttentionConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore sparse attention model from saved state.

    Args:
        model: Model to restore
        config: Sparse attention configuration
        metadata: Saved metadata

    Returns:
        Restored model
    """
    # Convert to sparse attention model
    model, _ = convert_to_sparse_attention_model(model, config)

    # Restore sparse attention state from metadata
    if "sparse_attention_state" in metadata:
        restore_sparse_attention_state(model, metadata["sparse_attention_state"])

    return model


def restore_sparse_attention_state(model: nn.Module, state_dict: dict[str, Any]):
    """Restore sparse attention state from state dict.

    Args:
        model: Model with sparse attention modules
        state_dict: Saved state dictionary
    """
    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule):
            module_name = get_unwrapped_name(name, model)
            if module_name in state_dict:
                module_state = state_dict[module_name]

                # Restore method and config
                if "method" in module_state:
                    module._method = module_state["method"]
                if "method_config" in module_state:
                    # Restore config attributes
                    for key, val in module_state["method_config"].items():
                        setattr(module, f"_{key}", val)

                # Re-setup with restored config
                module._setup()


def update_sparse_attention_metadata(
    model: nn.Module, config: SparseAttentionConfig, metadata: MetadataDict
) -> None:
    """Update metadata with sparse attention state.

    Args:
        model: Model with sparse attention
        config: Configuration used
        metadata: Metadata dict to update
    """
    sparse_state = {}

    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule):
            module_name = get_unwrapped_name(name, model)

            # Collect method config from module attributes
            method_config = {
                k[1:]: v
                for k, v in module.__dict__.items()
                if k.startswith("_") and k not in ("_method", "_enabled", "_sparse_method_instance")
            }

            module_state = {
                "method": module._sparse_method_instance.name,
                "method_config": method_config,
            }

            sparse_state[module_name] = module_state

    metadata["sparse_attention_state"] = sparse_state
    metadata["sparse_attention_config"] = (
        config.model_dump() if hasattr(config, "model_dump") else vars(config)
    )


def disable_sparse_attention(model: nn.Module, wildcard_or_filter_func: str | Callable):
    """Disable sparse attention for matching modules.

    Similar to mtq.disable_quantizer for API consistency.

    Args:
        model: Model with sparse attention applied
        wildcard_or_filter_func: Wildcard string or filter function to match module names.
            For example: "*lm_head*", "*layer_0*", etc.

    Example:
        >>> import modelopt.torch.sparsity.attention_sparsity as sparse_attn
        >>> model = sparse_attn.sparsify(model, config)
        >>> # Disable sparse attention for lm_head
        >>> sparse_attn.disable_sparse_attention(model, "*lm_head*")
    """
    for name, module in model.named_modules():
        if not isinstance(module, SparseAttentionModule):
            continue

        matched = False
        if isinstance(wildcard_or_filter_func, str):
            matched = fnmatch.fnmatch(name, wildcard_or_filter_func)
        elif callable(wildcard_or_filter_func):
            matched = wildcard_or_filter_func(name)

        if matched:
            module.disable()


def enable_sparse_attention(model: nn.Module, wildcard_or_filter_func: str | Callable):
    """Enable sparse attention for matching modules.

    Similar to mtq.enable_quantizer for API consistency.

    Args:
        model: Model with sparse attention applied
        wildcard_or_filter_func: Wildcard string or filter function to match module names.
            For example: "*attention*", "*attn*", etc.

    Example:
        >>> import modelopt.torch.sparsity.attention_sparsity as sparse_attn
        >>> model = sparse_attn.sparsify(model, config)
        >>> # Re-enable sparse attention for all attention modules
        >>> sparse_attn.enable_sparse_attention(model, "*attention*")
    """
    for name, module in model.named_modules():
        if not isinstance(module, SparseAttentionModule):
            continue

        matched = False
        if isinstance(wildcard_or_filter_func, str):
            matched = fnmatch.fnmatch(name, wildcard_or_filter_func)
        elif callable(wildcard_or_filter_func):
            matched = wildcard_or_filter_func(name)

        if matched:
            module.enable()


def print_sparse_attention_summary(model: nn.Module):
    """Print summary of sparse attention modules in the model.

    Similar to mtq.print_quant_summary for API consistency.

    Args:
        model: Model with sparse attention applied

    Prints:
        - Total sparse attention modules
        - Enabled vs disabled count
        - Method distribution
        - Configuration summary by module

    Example:
        >>> import modelopt.torch.sparsity.attention_sparsity as sparse_attn
        >>> model = sparse_attn.sparsify(model, config)
        >>> sparse_attn.print_sparse_attention_summary(model)
    """
    sparse_modules = []
    for name, module in model.named_modules():
        if isinstance(module, SparseAttentionModule):
            sparse_modules.append((name, module))

    if not sparse_modules:
        print("No sparse attention modules found in model")
        return

    enabled_count = sum(1 for _, m in sparse_modules if m.is_enabled)
    disabled_count = len(sparse_modules) - enabled_count

    # Count methods
    method_counts = {}
    for _, module in sparse_modules:
        method = getattr(module, "_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"{'Sparse Attention Summary':^70}")
    print(f"{'=' * 70}")
    print(f"Total sparse attention modules: {len(sparse_modules)}")
    print(f"  Enabled:  {enabled_count}")
    print(f"  Disabled: {disabled_count}")

    if method_counts:
        print("\nMethods:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method}: {count}")

    print(f"\n{'Module Details':^70}")
    print(f"{'-' * 70}")

    for name, module in sparse_modules:
        status = "✓" if module.is_enabled else "✗"
        method = getattr(module, "_method", "unknown")
        threshold = getattr(module, "_threshold", "N/A")

        # Format threshold nicely
        if isinstance(threshold, dict):
            threshold_str = str(threshold)
        elif isinstance(threshold, float):
            threshold_str = f"{threshold:.2e}"
        else:
            threshold_str = str(threshold)

        print(f"{status} {name}")
        print(f"   Method: {method}, Threshold: {threshold_str}")

    print(f"{'=' * 70}\n")
