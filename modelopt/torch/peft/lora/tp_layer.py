"""Tensor Parallel LoRA implementations for Megatron layers."""

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.init as init

from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear

from .layer import LoRAModuleRegistry, LoRAModule


# Default rank for LoRA decomposition
DEFAULT_LORA_RANK = 64


class _MegatronParallelLoRABase(LoRAModule):
    """Base class for Megatron tensor parallel LoRA implementations.
    
    This class provides common functionality for both ColumnParallel and RowParallel
    LoRA implementations, reducing code duplication.
    """
    
    def _get_init_methods(self) -> tuple[Callable, Callable]:
        """Get initialization methods for LoRA A and B matrices.
        
        Returns:
            Tuple of (lora_a_init, lora_b_init) initialization functions
        """
        # LoRA A uses Kaiming uniform initialization
        lora_a_init = lambda weight: init.kaiming_uniform_(weight, a=math.sqrt(5))
        # LoRA B is initialized to zero for stable training start
        lora_b_init = lambda weight: init.zeros_(weight)
        return lora_a_init, lora_b_init
    
    def _register_adapter(self, adapter_name: str, lora_a: nn.Module, lora_b: nn.Module) -> None:
        """Register LoRA adapter modules.
        
        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
        """
        # Move LoRA modules to the same device as the parent module
        # Try to get device from parent module's parameters or buffers
        device = None
        for p in self.parameters():
            device = p.device
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                break
        
        # If we found a device, move LoRA modules to it
        if device is not None:
            lora_a = lora_a.to(device)
            lora_b = lora_b.to(device)
        
        # Add as submodules for proper parameter registration
        self.add_module(f'lora_a_{adapter_name}', lora_a)
        self.add_module(f'lora_b_{adapter_name}', lora_b)
        
        # Store in adapter dictionary
        self._lora_adapters[adapter_name] = {
            "lora_a": lora_a,
            "lora_b": lora_b
        }
        
        # Automatically activate new adapters
        self.activate_adapter(adapter_name)


@LoRAModuleRegistry.register({ColumnParallelLinear: "megatron_ColumnParallelLinear"})
class _MegatronColumnParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron ColumnParallelLinear layers.
    
    This implementation creates column-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """
    
    def update_layer_lora(self, adapter_name: str, rank: int = DEFAULT_LORA_RANK) -> None:
        """Create and register a new LoRA adapter for ColumnParallelLinear.
        
        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a_init, lora_b_init = self._get_init_methods()
        
        # Create LoRA A: input_size -> rank (with gather for full reduction)
        lora_a = ColumnParallelLinear(
            self.input_size,
            rank,
            config=self.config,
            bias=False,
            gather_output=True,  # Gather outputs for complete transformation
            init_method=lora_a_init,
            disable_grad_reduce=getattr(self.config, 'sequence_parallel', False),
        )
        
        # Create LoRA B: rank -> output_size (no gather, stays distributed)
        lora_b = ColumnParallelLinear(
            rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=False,  # Keep output distributed like base layer
            init_method=lora_b_init,
        )
        
        self._register_adapter(adapter_name, lora_a, lora_b)


@LoRAModuleRegistry.register({RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron RowParallelLinear layers.
    
    This implementation creates row-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """
    
    def update_layer_lora(self, adapter_name: str, rank: int = DEFAULT_LORA_RANK) -> None:
        """Create and register a new LoRA adapter for RowParallelLinear.
        
        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a_init, lora_b_init = self._get_init_methods()
        
        # Create LoRA A: input_size -> rank (row parallel, input already distributed)
        lora_a = RowParallelLinear(
            self.input_size,
            rank,
            config=self.config,
            input_is_parallel=True,  # Input is already distributed
            skip_bias_add=True,
            bias=False,
            init_method=lora_a_init,
        )
        
        # Create LoRA B: rank -> output_size (column parallel with gather)
        lora_b = ColumnParallelLinear(
            rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=True,  # Gather to match base layer output
            init_method=lora_b_init,
        )
        
        self._register_adapter(adapter_name, lora_a, lora_b)