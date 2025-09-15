"""LoRA (Low-Rank Adaptation) module implementation."""

from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional
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
        self._lora_adapters: Dict[str, Dict[str, nn.Module]] = {}
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
            raise ValueError(f"Adapter '{adapter_name}' not found. Available: {list(self._lora_adapters.keys())}")
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
    
    @abstractmethod
    def update_layer_lora(self, adapter_name: str, rank: int = 64) -> None:
        """Create and register a new LoRA adapter.
        
        This method must be implemented by subclasses to create the appropriate
        LoRA A and B matrices for the specific layer type.
        
        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition (default: 64)
        """
        raise NotImplementedError("Subclasses must implement update_layer_lora")
    
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
                    lora_a = adapter['lora_a']
                    lora_b = adapter['lora_b']
                    
                    # Handle different forward signatures
                    lora_a_output = lora_a(x)
                    if isinstance(lora_a_output, tuple):
                        lora_a_output = lora_a_output[0]
                    
                    lora_b_output = lora_b(lora_a_output)
                    if isinstance(lora_b_output, tuple):
                        lora_b_output = lora_b_output[0]
                    
                    result = result + lora_b_output
        
        # Return output in the same format as the base layer
        if other_outputs:
            return (result,) + other_outputs
        else:
            return result


LoRAModuleRegistry = _DMRegistryCls("LoRA", LoRAModule)