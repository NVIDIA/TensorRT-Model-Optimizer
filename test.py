# dummy_megatron_model.py
import os
import torch
import torch.nn.init as init
from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

import modelopt.torch.peft as mtp
import modelopt.torch.quantization as mtq


class DummyMegatronModel(MegatronModule):
    """
    A simple dummy Megatron model with parallel linear layers for testing.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
        # Column parallel linear layer (splits output dimension)
        self.linear_0 = tensor_parallel.ColumnParallelLinear(
            input_size=10,
            output_size=10,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            gather_output=False,
        )
        self.linear_1 = tensor_parallel.RowParallelLinear(
            input_size=10,
            output_size=10,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        # Row parallel linear layer (splits input dimension)
        self.lm_head_0 = tensor_parallel.ColumnParallelLinear(
            input_size=10,
            output_size=10,
            config=config,
            init_method=init.xavier_normal_,
            bias=False,
            gather_output=False,
        )
        self.lm_head_1 = tensor_parallel.RowParallelLinear(
            input_size=10,
            output_size=10,
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
        backend="nccl", 
        world_size=world_size, 
        rank=rank, 
        init_method=init_method
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


def create_dummy_megatron_model():
    """
    Create and return a dummy Megatron model.
    
    Returns:
        DummyMegatronModel: The initialized model on CUDA
    """
    # Initialize model parallel (single GPU by default)
    initialize_model_parallel(
        tensor_model_parallel_size=1, 
        pipeline_model_parallel_size=1
    )
    
    # Set random seed for reproducibility
    model_parallel_cuda_manual_seed(123)
    
    # Configure the transformer
    transformer_config = {
        "num_layers": 2,
        "hidden_size": 12,
        "num_attention_heads": 4,
        "use_cpu_initialization": True,
    }
    config = TransformerConfig(**transformer_config)
    
    # Create and return the model
    model = DummyMegatronModel(config=config)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def cleanup():
    """Clean up distributed and model parallel groups."""
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Example usage
    try:
        # Create the model
        model = create_dummy_megatron_model()
        print(f"Created dummy Megatron model: {model}")
        # Test forward pass
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 10).cuda()
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
        
        # # Print model structure
        # print("\nModel structure:")
        # for name, module in model.named_modules():
        #     print(f"  {name}: {module.__class__.__name__}")
        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*transformer*qkv*": {"rank": 64},
                "*ffn*": {"rank": 128},
                "*linear*": {"rank": 128}
            }
        }
        # model = mtp.update(model, mode=[("peft", lora_config)])
        model = mtp.update_model(model, lora_config)
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 10).cuda()
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
        # mtq.quantize(model, mtq.MXFP4_DEFAULT_CFG)
    finally:
        # Clean up
        cleanup()
        print("\nCleaned up distributed environment")