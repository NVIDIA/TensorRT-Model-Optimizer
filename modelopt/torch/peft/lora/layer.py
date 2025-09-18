"""LoRA (Low-Rank Adaptation) module implementation."""

import warnings
from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls

__all__ = [
    "LoRAModule",
    "LoRAModuleRegistry",
]


class LoRAModule(DynamicModule):
    """Base class for LoRA (Low-Rank Adaptation) modules.

    This module wraps existing layers and adds trainable low-rank decomposition
    matrices (LoRA adapters) that are added to the original layer's output.

    Attributes:
        _lora_adapters: Dictionary mapping adapter names to their LoRA A and B matrices
        _active_adapters: Set of currently active adapter names
    """

    def _setup(self) -> None:
        """Initialize LoRA-specific attributes."""
        self._lora_adapters: dict[str, dict[str, Any]] = {}
        self._active_adapters: set = set()

    @property
    def adapter_names(self) -> set:
        """Return the set of all registered adapter names."""
        return set(self._lora_adapters.keys())

    @property
    def active_adapters(self) -> set:
        """Return the set of currently active adapter names."""
        return self._active_adapters.copy()

    def activate_adapter(self, adapter_name: str) -> None:
        """Activate a specific adapter.

        Args:
            adapter_name: Name of the adapter to activate

        Raises:
            ValueError: If adapter_name is not registered
        """
        if adapter_name not in self._lora_adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {list(self._lora_adapters.keys())}"
            )
        self._active_adapters.add(adapter_name)

    def deactivate_adapter(self, adapter_name: str) -> None:
        """Deactivate a specific adapter.

        Args:
            adapter_name: Name of the adapter to deactivate
        """
        self._active_adapters.discard(adapter_name)

    def activate_all_adapters(self) -> None:
        """Activate all registered adapters."""
        self._active_adapters = self.adapter_names.copy()

    def deactivate_all_adapters(self) -> None:
        """Deactivate all adapters."""
        self._active_adapters.clear()

    def _register_adapter(
        self, adapter_name: str, lora_a: nn.Module, lora_b: nn.Module, rank: int, scale: float = 1.0
    ) -> None:
        """Register a new LoRA adapter with explicit rank tracking.

        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
            rank: Rank of the LoRA decomposition
            scale: Scale factor for the LoRA output
        """
        # Add as submodules for proper parameter registration
        self.add_module(f"lora_a_{adapter_name}", lora_a)
        self.add_module(f"lora_b_{adapter_name}", lora_b)

        # Store in adapter dictionary with explicit rank
        self._lora_adapters[adapter_name] = {
            "lora_a": lora_a,
            "lora_b": lora_b,
            "rank": rank,  # Store rank explicitly for reliability
            "scale": scale,
        }

        # Automatically activate new adapters
        self.activate_adapter(adapter_name)

    @abstractmethod
    def update_layer_lora(self, adapter_name: str, rank: int = 64, scale: float = 1.0) -> None:
        """Create and register a new LoRA adapter.

        This method must be implemented by subclasses to create the appropriate
        LoRA A and B matrices for the specific layer type.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition (default: 64)
            scale: Scale factor for the LoRA output (default: 1.0)
        """
        raise NotImplementedError("Subclasses must implement update_layer_lora")

    def get_peft_state(self) -> dict[str, Any]:
        """Get PEFT/LoRA state to be saved in checkpoint.

        This method returns the configuration and state of all LoRA adapters
        without including the actual weight tensors.

        Returns:
            Dictionary containing:
            - adapters: Dict mapping adapter names to their configuration
            - active_adapters: List of currently active adapter names
        """
        modelopt_state = {}

        # Store adapter configurations
        adapters_config = {}
        for adapter_name, adapter_modules in self._lora_adapters.items():
            lora_a = adapter_modules["lora_a"]
            lora_b = adapter_modules["lora_b"]

            # Get explicitly stored rank for reliability
            rank = adapter_modules.get("rank", None)

            # If rank is not stored (legacy case), try to infer it
            if rank is None:
                if hasattr(lora_a, "output_size"):
                    rank = lora_a.output_size
                elif hasattr(lora_b, "input_size"):
                    rank = lora_b.input_size
                elif hasattr(lora_a, "out_features"):
                    rank = lora_a.out_features
                elif hasattr(lora_b, "in_features"):
                    rank = lora_b.in_features

            adapters_config[adapter_name] = {
                "rank": rank,
                "is_active": adapter_name in self._active_adapters,
                "lora_a_type": type(lora_a).__name__,
                "lora_b_type": type(lora_b).__name__,
                "scale": adapter_modules.get("scale", 1.0),
            }

        modelopt_state["adapters"] = adapters_config
        modelopt_state["active_adapters"] = list(self._active_adapters)

        return modelopt_state

    def get_extra_state(self) -> dict[str, Any]:
        """Get extra state for distributed checkpointing.

        For distributed/sharded checkpoints (like NeMo-MCore), we store the PEFT state
        as extra_state instead of in metadata. This handles cases where module names
        change with different parallelism settings (TP, PP, EP).

        Returns:
            Dictionary containing the PEFT/LoRA adapter state
        """
        # Only return state if we have adapters
        if not self._lora_adapters:
            return {}

        # Get the current PEFT state
        peft_state = self.get_peft_state()

        return {"modelopt_peft_state": peft_state}

    def set_from_peft_state(self, peft_state: dict[str, Any]) -> None:
        """Restore LoRA adapters from saved PEFT state.

        This method recreates LoRA adapters based on their saved configuration.
        Note: This only restores the adapter structure, not the weights.

        Args:
            peft_state: Dictionary containing adapter configurations
        """
        adapters_config = peft_state.get("adapters", {})

        # Clear existing adapters first
        self._lora_adapters.clear()
        self._active_adapters.clear()

        # Recreate each adapter based on saved configuration
        for adapter_name, config in adapters_config.items():
            rank = config.get("rank")
            scale = config.get("scale", 1.0)

            if rank is not None:
                # Create the adapter with saved configuration
                self.update_layer_lora(adapter_name, rank=rank, scale=scale)

                # Set activation state
                if config.get("is_active", False):
                    self.activate_adapter(adapter_name)
                else:
                    self.deactivate_adapter(adapter_name)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore extra state for distributed checkpointing.

        This method is called during load_state_dict() to restore the PEFT/LoRA state
        from distributed checkpoints. It handles the adapter configuration but not
        the actual weights (which are restored through the normal state_dict mechanism).

        Args:
            state: Dictionary containing the extra state to restore
        """
        if state is None:
            return

        peft_state = state.get("modelopt_peft_state")
        if peft_state is None:
            return

        # Restore the PEFT state
        try:
            self.set_from_peft_state(peft_state)
        except Exception as e:
            warnings.warn(
                f"Failed to restore PEFT state from extra_state: {e}. "
                "This might happen if the model structure has changed."
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor
            *args: Additional positional arguments for the base layer
            **kwargs: Additional keyword arguments for the base layer

        Returns:
            Output from the base layer plus active LoRA adaptations
        """
        # Call the base layer's forward method
        output = super().forward(x, *args, **kwargs)

        # Handle different output types from base layer
        if isinstance(output, tuple):
            # If output is a tuple, assume first element is the main result
            result = output[0]
            other_outputs = output[1:]
        else:
            # If output is a single tensor
            result = output
            other_outputs = ()

        # Apply active LoRA adapters
        if self._active_adapters and self._lora_adapters:
            for adapter_name in self._active_adapters:
                if adapter_name in self._lora_adapters:
                    adapter = self._lora_adapters[adapter_name]
                    # LoRA computation: result = result + B(A(x))
                    lora_a = adapter["lora_a"]
                    lora_b = adapter["lora_b"]

                    # Handle different forward signatures
                    lora_a_output = lora_a(x)
                    if isinstance(lora_a_output, tuple):
                        lora_a_output = lora_a_output[0]

                    lora_b_output = lora_b(lora_a_output)
                    if isinstance(lora_b_output, tuple):
                        lora_b_output = lora_b_output[0]

                    scale = adapter.get("scale", 1.0)
                    result = result + scale * lora_b_output

        # Return output in the same format as the base layer
        if other_outputs:
            return (result,) + other_outputs
        else:
            return result


LoRAModuleRegistry = _DMRegistryCls("LoRA", LoRAModule)
