"""Megatron Tensor Parallel Model Test Script

This script demonstrates:
1. Creating a Megatron model with tensor parallelism (TP=2)
2. Applying LoRA adapters to tensor parallel layers
3. Testing the model with proper distributed initialization

To run with tensor parallelism:
    torchrun --nproc_per_node=2 test.py
    or
    bash run_tp_test.sh

The model uses ColumnParallelLinear and RowParallelLinear layers which
automatically handle weight sharding across GPUs when TP > 1.
"""

import os

import torch
import torch.nn.init as init
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

import modelopt.torch.peft as mtp


class DummyMegatronModel(MegatronModule):
    """A simple dummy Megatron model with parallel linear layers for testing.
    Uses larger dimensions to better demonstrate tensor parallelism.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        # Larger dimensions for better tensor parallel demonstration
        hidden_size = 1024  # Divisible by 2 for TP=2
        intermediate_size = 4096  # 4x hidden size, typical for transformers

        # Column parallel linear layer (splits output dimension)
        self.linear_0 = tensor_parallel.ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            gather_output=False,
        )
        self.linear_1 = tensor_parallel.RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        # Row parallel linear layer (splits input dimension)
        self.lm_head_0 = tensor_parallel.ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            gather_output=False,
        )
        self.lm_head_1 = tensor_parallel.RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )

    def forward(self, input):
        x = self.linear_0(input)[0]
        x = self.linear_1(x)[0]
        x = self.lm_head_0(x)[0]
        x = self.lm_head_1(x)[0]
        return x


def initialize_distributed(rank=0, world_size=1):
    """Initialize torch distributed for parallel training."""
    if torch.distributed.is_initialized():
        return

    print(f"Initializing torch.distributed with rank: {rank}, world_size: {world_size}")
    torch.cuda.set_device(rank)

    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6001")
    init_method += master_ip + ":" + master_port

    torch.distributed.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, init_method=init_method
    )


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
):
    """Initialize Megatron's model parallel groups."""
    # Destroy existing model parallel if any
    parallel_state.destroy_model_parallel()

    # Initialize distributed if not already done
    if not torch.distributed.is_initialized():
        initialize_distributed()

    # Initialize model parallel groups
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank,
    )


def create_dummy_megatron_model(tensor_model_parallel_size=2):
    """Create and return a dummy Megatron model with tensor parallelism.

    Args:
        tensor_model_parallel_size: Size of tensor model parallelism (default: 2)

    Returns:
        DummyMegatronModel: The initialized model on CUDA
    """
    # Get rank from environment or default to 0
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(tensor_model_parallel_size)))

    # Initialize distributed and model parallel
    initialize_distributed(rank=rank, world_size=world_size)
    initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
    )

    # Set random seed for reproducibility
    model_parallel_cuda_manual_seed(123)

    # Configure the transformer with larger dimensions
    transformer_config = {
        "num_layers": 4,
        "hidden_size": 1024,  # Must match model dimensions
        "num_attention_heads": 16,
        "use_cpu_initialization": True,
        "sequence_parallel": False,  # Set to True for sequence parallelism
    }
    config = TransformerConfig(**transformer_config)

    # Create and return the model
    model = DummyMegatronModel(config=config)

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Model created on rank {rank} with TP size {tensor_model_parallel_size}")

    return model


def cleanup():
    """Clean up distributed and model parallel groups."""
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    """
    To run with tensor parallelism size 2, use:
    torchrun --nproc_per_node=2 test.py
    
    Or manually with:
    RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=6001 python test.py &
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=6001 python test.py
    """
    try:
        # Create the model with TP=2
        tensor_parallel_size = 2
        model = create_dummy_megatron_model(tensor_model_parallel_size=tensor_parallel_size)

        # Get rank for printing
        rank = int(os.environ.get("RANK", "0"))

        if rank == 0:
            print(f"\nCreated dummy Megatron model with TP={tensor_parallel_size}")
            print("Model structure:")
            for name, module in model.named_modules():
                if hasattr(module, "__class__"):
                    print(f"  {name}: {module.__class__.__name__}")

        # Test forward pass
        if torch.cuda.is_available():
            batch_size = 2
            seq_length = 512
            hidden_size = 1024  # Must match model hidden size

            # Create input tensor
            x = torch.randn(batch_size, seq_length, hidden_size).cuda()

            # Synchronize before forward pass
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            output = model(x)

            if rank == 0:
                print("\nForward pass successful!")
                print(f"Input shape: {x.shape}")
                print(f"Output shape: {output.shape}")

        # Test LoRA with tensor parallel model
        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {"*linear*": {"rank": 64}, "*lm_head*": {"rank": 128}},
        }

        if rank == 0:
            print("\nApplying LoRA configuration...")

        model = mtp.update_model(model, lora_config)

        # Test forward pass with LoRA
        if torch.cuda.is_available():
            output_lora = model(x)
            if rank == 0:
                print("LoRA forward pass successful!")
                print(model)
                print(model.linear_0.lora_a_default)
                print(f"Output shape with LoRA: {output_lora.shape}")

        # Optional: Test quantization (commented out)
        # if rank == 0:
        #     print(f"\nApplying quantization...")
        # mtq.quantize(model, mtq.INT8_DEFAULT_CFG)

    except Exception as e:
        print(f"Error on rank {os.environ.get('RANK', '0')}: {e}")
        raise
    finally:
        # Clean up
        cleanup()
        if int(os.environ.get("RANK", "0")) == 0:
            print("\nCleaned up distributed environment")
