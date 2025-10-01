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

"""Extensible sparse attention module."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

# Try to import ALL_ATTENTION_FUNCTIONS (requires transformers >= 4.48)
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    HAS_ATTENTION_INTERFACE = True
except (ImportError, AttributeError):
    ALL_ATTENTION_FUNCTIONS = None
    HAS_ATTENTION_INTERFACE = False

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.quantization.utils import replace_function

from ..config import SparseAttentionAttributeConfig
from ..methods import get_sparse_method


class SparseAttentionModule(DynamicModule):
    """Generic sparse attention module wrapper for applying sparsity to attention layers.

    This module wraps existing attention implementations to add sparse attention
    capabilities. It dynamically detects and adapts to different attention
    implementations (eager, sdpa, flash_attention_2) and supports multiple
    sparsity methods with different backends.

    Forward Flow:
    -------------
    1. Check if sparse attention is enabled (pass-through if disabled)
    2. Determine the appropriate context manager based on:
       - Sparse method (e.g., flash_softmax_skip)
       - Backend (pytorch or triton)
    3. Apply sparse attention through the context manager:
       - PyTorch backend: Patches torch.nn.functional.softmax
       - Triton backend: Patches attention implementation with fused kernel
       - Other methods: Use generic attention patching
    4. Forward through the original attention with sparsity applied

    The module supports automatic fallback from Triton to PyTorch backend
    if Triton requirements are not met (e.g., old transformers version).

    Attributes:
    -----------
    _enabled: bool
        Whether sparse attention is enabled
    _method: str
        The sparse attention method to use (e.g., "flash_softmax_skip")
    _method_config: dict
        Configuration dictionary for the sparse method (threshold, br, bc, backend, etc.)
    _sparse_method_instance: SparseAttentionMethod
        Instance of the configured sparse attention method
    """

    def set_from_attribute_config(
        self, attribute_cfg: SparseAttentionAttributeConfig | dict | None = None
    ):
        """Set sparse attention attributes from configuration.

        Similar to TensorQuantizer.set_from_attribute_config.

        Args:
            attribute_cfg: Sparse attention attribute configuration.
                         If None, uses default SparseAttentionAttributeConfig.
        """
        # Use default config if not provided
        attribute_cfg = (
            attribute_cfg if attribute_cfg is not None else SparseAttentionAttributeConfig()
        )

        # Store raw config for method initialization
        self._method_config = {}

        # Define which attributes are method-specific vs module-specific
        # Module-specific attributes control the SparseAttentionModule behavior
        _module_attributes = {"enable", "method"}

        # Ignored attributes - handled at higher level (calibration framework)
        _ignored_attributes = {"calibration"}

        # Custom setters for special module attributes
        _custom_setters = {
            "enable": ("_enabled", lambda val: bool(val)),
            "method": ("_method", lambda val: str(val)),
        }

        # Process each attribute from config
        for attribute, val in attribute_cfg.items():
            # Validate attribute if using config class
            if hasattr(SparseAttentionAttributeConfig, "model_fields"):
                assert attribute in SparseAttentionAttributeConfig.model_fields, (
                    f"{attribute} is not a valid SparseAttentionModule attribute"
                )

            # Skip ignored attributes (handled by calibration framework)
            if attribute in _ignored_attributes:
                continue

            if attribute in _module_attributes:
                # Module-level attribute: store with underscore prefix
                attr_name, setter = _custom_setters.get(attribute, (f"_{attribute}", lambda v: v))
                setattr(self, attr_name, setter(val))
            else:
                # Method-specific attribute: store in config dict
                self._method_config[attribute] = val

        # Initialize sparse method instance
        self._init_sparse_method()

    def _init_sparse_method(self):
        """Initialize the sparse method instance."""
        method_class = get_sparse_method(self._method)

        # Initialize the sparse method instance
        # _method_config is always initialized in set_from_attribute_config
        self._sparse_method_instance = method_class(method_config=self._method_config)  # type: ignore[call-arg]

    def enable(self):
        """Enable sparse attention for this module.

        Similar to TensorQuantizer.enable() for API consistency.
        """
        self._enabled = True

    def disable(self):
        """Disable sparse attention for this module.

        Similar to TensorQuantizer.disable() for API consistency.
        """
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if sparse attention is enabled.

        Returns:
            True if sparse attention is enabled, False otherwise.

        Note: Unlike quantization which uses _disabled (inverted), we use _enabled (direct)
        for better code clarity and consistency with other frameworks.
        """
        return getattr(self, "_enabled", True)

    def get_stats(self) -> dict:
        """Get sparsity statistics from the sparse method.

        Returns:
            Dictionary with sparsity statistics including 'average_sparsity' if available.
            Returns empty dict if no sparse method instance or stats unavailable.
        """
        if not hasattr(self, "_sparse_method_instance") or self._sparse_method_instance is None:
            return {}

        method = self._sparse_method_instance
        stats = {}

        # Get stats from the method if available
        if hasattr(method, "stats") and isinstance(method.stats, dict):
            method_stats = method.stats
            # Calculate average sparsity if we have the data
            if "sparsity" in method_stats:
                stats["average_sparsity"] = method_stats["sparsity"]
            elif "sparse_blocks" in method_stats and "total_blocks" in method_stats:
                total = method_stats.get("total_blocks", 0)
                if total > 0:
                    stats["average_sparsity"] = method_stats["sparse_blocks"] / total

        return stats

    def _setup(self):
        """Setup called by DynamicModule.

        This is called after the module is wrapped by DynamicModule.
        If attributes haven't been set explicitly, apply defaults.
        """
        # If not yet configured, apply default configuration
        # We check for _method as the indicator of configuration
        if not hasattr(self, "_method"):
            self.set_from_attribute_config(None)

        self._detect_attention_type()

    def _detect_attention_type(self):
        """Detect which attention implementation is being used."""
        if not HAS_ATTENTION_INTERFACE:
            raise ImportError(
                "Sparse attention requires transformers >= 4.48.0. "
                "Please upgrade: pip install transformers>=4.48.0"
            )

        # Get what the model is configured to use
        if hasattr(self, "config") and hasattr(self.config, "_attn_implementation"):
            self.attention_impl = self.config._attn_implementation
        else:
            self.attention_impl = "flash_attention_2"

    def forward(self, *args, **kwargs):
        """Forward with selected sparse attention method.

        This method dispatches to the appropriate sparse attention implementation
        based on the configured method and backend.
        """
        # Pass through if sparse attention is disabled
        if not self.is_enabled:
            return super().forward(*args, **kwargs)

        # Get the appropriate context manager for this configuration
        context = self._get_sparse_context()

        # Apply sparse attention through the context
        with context:
            return super().forward(*args, **kwargs)

    def _get_sparse_context(self):
        """Get the appropriate context manager for the current configuration.

        Returns:
            Context manager for applying sparse attention based on method and backend.
        """
        method_name = self._sparse_method_instance.name
        backend = getattr(self._sparse_method_instance, "backend", "pytorch")

        # Dispatch table for different method/backend combinations
        if method_name == "flash_softmax_skip":
            if backend == "triton":
                # Use Triton fused kernel for Flash Attention
                return self._sparse_attention_context_triton()
            else:
                # Use PyTorch backend with softmax patching
                return self._create_softmax_patch_context()
        else:
            # Other methods use generic attention patching
            return self._sparse_attention_context()

    def _create_softmax_patch_context(self):
        """Create context manager for patching softmax function.

        Returns:
            Context manager that patches torch.nn.functional.softmax
        """
        return replace_function(torch.nn.functional, "softmax", self._create_sparse_softmax())

    def _create_sparse_softmax(self):
        """Create sparse softmax function for current method."""
        original_softmax = F.softmax

        def sparse_softmax(input, dim=-1, *args, **kwargs):
            # Let the method handle the sparsification
            _, _, _, sparse_input = self._sparse_method_instance.apply_sparsity(
                None, None, None, input
            )

            # Use sparse input if modified, otherwise use original
            if sparse_input is not None:
                return original_softmax(sparse_input, dim, *args, **kwargs)
            return original_softmax(input, dim, *args, **kwargs)

        return sparse_softmax

    @contextmanager
    def _sparse_attention_context(self):
        """Context manager for non-softmax sparse methods."""
        # Check if attention implementation exists
        if self.attention_impl not in ALL_ATTENTION_FUNCTIONS:
            yield
            return

        # Save original and patch with sparse version
        original_fn = ALL_ATTENTION_FUNCTIONS[self.attention_impl]
        patched_fn = self._patch_attention_interface(original_fn)
        ALL_ATTENTION_FUNCTIONS[self.attention_impl] = patched_fn

        try:
            yield
        finally:
            ALL_ATTENTION_FUNCTIONS[self.attention_impl] = original_fn

    @contextmanager
    def _sparse_attention_context_triton(self):
        """Context manager for Triton backend sparse attention.

        Replaces the model's attention implementation in ALL_ATTENTION_FUNCTIONS
        with our Triton wrapper.
        """
        # Check if the attention implementation exists
        if self.attention_impl not in ALL_ATTENTION_FUNCTIONS:
            import warnings

            available_keys = list(ALL_ATTENTION_FUNCTIONS.keys())
            warnings.warn(
                f"Attention implementation '{self.attention_impl}' not in ALL_ATTENTION_FUNCTIONS. "
                f"Available keys: {available_keys}. "
                "Falling back to PyTorch backend."
            )
            with self._create_softmax_patch_context():
                yield
            return

        # Save original and replace with Triton wrapper
        original_fn = ALL_ATTENTION_FUNCTIONS[self.attention_impl]
        triton_wrapper = self._create_triton_wrapper()
        ALL_ATTENTION_FUNCTIONS[self.attention_impl] = triton_wrapper

        try:
            yield
        finally:
            # Always restore original
            ALL_ATTENTION_FUNCTIONS[self.attention_impl] = original_fn

    def _create_triton_wrapper(self):
        """Create Triton attention wrapper with current configuration.

        Returns:
            Wrapper function for Triton attention forward.
        """
        from ..utils.triton_wrapper import triton_attention_forward

        # Get threshold from method instance
        threshold = getattr(self._sparse_method_instance, "threshold", 1e-4)

        # Create wrapper with threshold bound
        def triton_wrapper_with_config(
            module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs
        ):
            return triton_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout,
                scaling,
                softmax_skip_thresh=threshold,
                **kwargs,
            )

        return triton_wrapper_with_config

    def _patch_attention_interface(self, original_interface):
        """Patch any attention interface with sparse logic."""

        def sparse_attention_wrapper(self_attn, query, key, value, *args, **kwargs):
            # Apply sparsity logic
            query, key, value, _ = self._sparse_method_instance.apply_sparsity(
                query, key, value, None
            )

            return original_interface(self_attn, query, key, value, *args, **kwargs)

        return sparse_attention_wrapper


# Create registry for sparse attention modules
SparseAttentionRegistry = _DMRegistryCls("SparseAttention", SparseAttentionModule)
