"""Tensor Parallel LoRA implementations for Megatron layers."""

import math
from collections.abc import Callable

import torch.nn as nn
import torch.nn.init as init
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from ..config import PEFTAttributeConfig
from .layer import LoRAModule, LoRAModuleRegistry

try:
    from modelopt.torch.quantization.plugins.megatron import (
        _MegatronColumnParallelLinear as QuantColumnParallelLinear,
    )
    from modelopt.torch.quantization.plugins.megatron import (
        _MegatronRowParallelLinear as QuantRowParallelLinear,
    )

    QUANT_MODULES_AVAILABLE = True
except ImportError:
    QUANT_MODULES_AVAILABLE = False

DEFAULT_LORA_RANK = 64
DEFAULT_SCALE = 1.0


class _MegatronParallelLoRABase(LoRAModule):
    """Base class for Megatron tensor parallel LoRA implementations.

    This class provides common functionality for both ColumnParallel and RowParallel
    LoRA implementations, reducing code duplication.
    """

    def _get_init_methods(self, lora_a_init, lora_b_init) -> tuple[Callable, Callable]:
        """Get initialization methods for LoRA A and B matrices.

        Returns:
            Tuple of (lora_a_init, lora_b_init) initialization functions
        """
        if lora_a_init is None:
            lora_a_init = lambda weight: init.kaiming_uniform_(weight, a=math.sqrt(5))  # noqa: E731  # LoRA A: Kaiming uniform
        if lora_b_init is None:
            lora_b_init = lambda weight: init.zeros_(weight)  # noqa: E731  # LoRA B: zeros
        return lora_a_init, lora_b_init

    def _register_adapter_with_device(
        self, adapter_name: str, lora_a: nn.Module, lora_b: nn.Module, rank: int, scale: float
    ) -> None:
        """Register LoRA adapter modules and ensure correct device placement.

        Args:
            adapter_name: Name of the adapter
            lora_a: LoRA A module (down-projection)
            lora_b: LoRA B module (up-projection)
            rank: Rank of the LoRA decomposition
        """
        # Move LoRA modules to the same device and dtype as the parent module
        # Try to get device and dtype from parent module's parameters or buffers
        device = None
        dtype = None
        for p in self.parameters():
            device = p.device
            dtype = p.dtype
            break
        if device is None:
            for b in self.buffers():
                device = b.device
                dtype = b.dtype
                break

        # If we found a device and dtype, move LoRA modules to match
        if device is not None:
            lora_a = lora_a.to(device)
            lora_b = lora_b.to(device)
        if dtype is not None:
            lora_a = lora_a.to(dtype)
            lora_b = lora_b.to(dtype)

        super()._register_adapter(adapter_name, lora_a, lora_b, rank, scale)


@LoRAModuleRegistry.register({ColumnParallelLinear: "megatron_ColumnParallelLinear"})
class _LoRAMegatronColumnParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron ColumnParallelLinear layers.

    This implementation creates column-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for ColumnParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = ColumnParallelLinear(
            self.input_size,
            attr_config.rank,
            config=self.config,
            bias=False,
            gather_output=True,
            init_method=attr_config.lora_a_init,
            disable_grad_reduce=getattr(self.config, "sequence_parallel", False),
        )

        lora_b = ColumnParallelLinear(
            attr_config.rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=False,  # Keep output distributed like base layer
            init_method=attr_config.lora_a_init,
        )

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale
        )


@LoRAModuleRegistry.register({RowParallelLinear: "megatron_RowParallelLinear"})
class _LoRAMegatronRowParallelLinear(_MegatronParallelLoRABase):
    """LoRA implementation for Megatron RowParallelLinear layers.

    This implementation creates row-parallel LoRA adapters that match
    the parallelization scheme of the base layer.
    """

    def update_layer_lora(
        self,
        adapter_name: str,
        attr_config: PEFTAttributeConfig,
    ) -> None:
        """Create and register a new LoRA adapter for RowParallelLinear.

        Args:
            adapter_name: Name for the new adapter
            rank: Rank of the LoRA decomposition
        """
        lora_a = RowParallelLinear(
            self.input_size,
            attr_config.rank,
            config=self.config,
            input_is_parallel=True,
            skip_bias_add=True,
            bias=False,
            init_method=attr_config.lora_a_init,
        )

        lora_b = ColumnParallelLinear(
            attr_config.rank,
            self.output_size,
            config=self.config,
            bias=False,
            gather_output=True,
            init_method=attr_config.lora_b_init,
        )

        self._register_adapter_with_device(
            adapter_name, lora_a, lora_b, attr_config.rank, attr_config.scale
        )


# Register quantized versions if available
if QUANT_MODULES_AVAILABLE:
    LoRAModuleRegistry.register({QuantColumnParallelLinear: "quant_megatron_ColumnParallelLinear"})(
        _LoRAMegatronColumnParallelLinear
    )
    LoRAModuleRegistry.register({QuantRowParallelLinear: "quant_megatron_RowParallelLinear"})(
        _LoRAMegatronRowParallelLinear
    )

    from modelopt.torch.quantization.nn import QuantModuleRegistry

    class _QuantLoRAMegatronColumnParallelLinear(
        _LoRAMegatronColumnParallelLinear, QuantColumnParallelLinear
    ):
        """Quantized LoRA ColumnParallelLinear that combines LoRA and quantization.

        This class ensures that the base layer functionality is quantized while
        preserving LoRA adapter functionality.
        """

        def _setup(self):
            QuantColumnParallelLinear._setup(self)

    class _QuantLoRAMegatronRowParallelLinear(
        _LoRAMegatronRowParallelLinear, QuantRowParallelLinear
    ):
        """Quantized LoRA RowParallelLinear that combines LoRA and quantization.

        This class ensures that the base layer functionality is quantized while
        preserving LoRA adapter functionality.
        """

        def _setup(self):
            QuantRowParallelLinear._setup(self)

    QuantModuleRegistry.register(
        {_LoRAMegatronColumnParallelLinear: "lora_megatron_ColumnParallelLinear"}
    )(_QuantLoRAMegatronColumnParallelLinear)
    QuantModuleRegistry.register(
        {_LoRAMegatronRowParallelLinear: "lora_megatron_RowParallelLinear"}
    )(_QuantLoRAMegatronRowParallelLinear)
