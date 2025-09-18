# import json
# from copy import deepcopy
# from functools import partial

# import pytest
# import torch
# import transformers
# # from _test_utils.import_helper import skip_if_no_megatron
# # from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
# from _test_utils.torch_dist.plugins.megatron_common import get_mcore_gpt_model
# # from _test_utils.torch_model.transformers_models import create_tiny_llama_dir

# # skip_if_no_megatron(apex_or_te_required=True)

# import modelopt.torch.speculative as mtsp
# from modelopt.torch.export import export_mcore_gpt_to_hf, import_mcore_gpt_from_hf
# from modelopt.torch.speculative.eagle.default_config import default_eagle_config
# from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel
# from modelopt.torch.speculative.plugins.megatron_medusa import _DynamicMedusaGPTModel
import copy
import os
from warnings import warn

import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.parallel_state import (
    initialize_model_parallel,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.utils.plugins import megatron_prefill

try:
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec

    HAS_TE = True
except ImportError as e:
    warn(f"Transformer Engine not installed: {e}")
    HAS_TE = False

try:
    from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
    from megatron.core.ssm.mamba_layer import MambaLayer

    HAS_MAMBA = True
except ImportError as e:
    warn(f"Mamba not installed: {e}")
    HAS_MAMBA = False

try:
    import apex  # noqa: F401

    HAS_APEX = True
except ImportError as e:
    warn(f"Apex not installed: {e}")
    HAS_APEX = False
import modelopt.torch.peft as mtp
import modelopt.torch.quantization as mtq

lora_config = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {"rank": 64},
    },
}


def initialize_for_megatron(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234
):
    """Initialize Megatron model parallelism.

    NOTE: If used in a non-spawned process, make sure to call `megatron.core.parallel_state.destroy_model_parallel()`.
    """
    initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
    model_parallel_cuda_manual_seed(seed)


def get_mcore_gpt_model(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    initialize_megatron: bool = False,
    *,
    num_layers: int = 2,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    hidden_size: int = 64,
    num_attention_heads: int = 8,
    num_query_groups: int | None = None,
    ffn_hidden_size: int | None = 128,
    max_sequence_length: int = 16,
    vocab_size: int = 64,
    activation_func: str = "swiglu",
    normalization: str = "LayerNorm",
    transformer_impl: str = "modelopt" if HAS_TE else "local",
    use_cpu_initialization: bool = False,
    bf16: bool = True,
) -> GPTModel:
    assert activation_func in ["swiglu", "squared_relu"]
    assert normalization in ["LayerNorm", "RMSNorm"]
    assert transformer_impl in ["local", "transformer_engine", "modelopt"]
    print(f"Using `{transformer_impl=}` model spec for building GPT Model.")

    if initialize_megatron:
        initialize_for_megatron(tensor_model_parallel_size, pipeline_model_parallel_size)

    def squared_relu(x):
        return torch.pow(F.relu(x), 2)

    config = TransformerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=False,
        num_layers=num_layers,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        activation_func=squared_relu if activation_func == "squared_relu" else F.silu,
        normalization=normalization,
        gated_linear_unit=(activation_func == "swiglu"),
        add_bias_linear=False,
        use_cpu_initialization=use_cpu_initialization,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
    )

    if transformer_impl == "local":
        assert HAS_APEX, "Apex not installed"
        transformer_layer_spec = get_gpt_layer_local_spec(normalization=normalization)
    else:
        assert HAS_TE, "Transformer Engine not installed"
        transformer_layer_spec = (
            get_gpt_modelopt_spec(config, remap_te_layernorm=True)
            if transformer_impl == "modelopt"
            else get_gpt_layer_with_transformer_engine_spec()
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=is_pipeline_first_stage(),
        post_process=is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
    )
    if bf16:
        model = model.to(torch.bfloat16)

    return model


def _gpt_model_provider(tp_size: int, hidden_size=256, vocab_size=64, meta_device=False):
    """Build the model."""

    if meta_device:
        with torch.device("meta"):
            gpt_model = get_mcore_gpt_model(
                tensor_model_parallel_size=tp_size,
                num_layers=4,
                ffn_hidden_size=None,
                num_attention_heads=4,
                activation_func="squared_relu",
                transformer_impl="local",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                use_cpu_initialization=meta_device,
            )
    else:
        gpt_model = get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            num_layers=4,
            ffn_hidden_size=None,
            num_attention_heads=4,
            activation_func="squared_relu",
            transformer_impl="local",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        ).cuda()
    return gpt_model.eval()


from megatron.core import parallel_state

from tests.gpu.torch.peft.test_forward_megatron import (
    _gpt_model_provider,
    initialize_for_megatron,
    megatron_prefill,
)


def _test_lora_forward():
    """Test LoRA forward pass with Megatron model."""
    # Initialize model parallel groups with proper CUDA RNG seed
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234)

    try:
        # Create model
        model = _gpt_model_provider(tp_size=1)

        # Create input tokens
        prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

        # Run forward pass
        output = megatron_prefill(model, prompt_tokens)
        print(
            f"Forward pass successful! Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}"
        )

        # Now test with LoRA

        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*attention*": {"rank": 32, "scale": 1},
                "*mlp*": {"rank": 64, "scale": 1},
            },
        }

        # # Apply LoRA
        model = mtp.update_model(model, lora_config)
        print("LoRA adapters added successfully!")

        # Test forward pass with LoRA
        output_lora = megatron_prefill(model, prompt_tokens)
        print(
            f"LoRA forward pass successful! Output shape: {output_lora.shape if hasattr(output_lora, 'shape') else 'N/A'}"
        )

        # Check if LoRA modules were added
        lora_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "_lora_adapters"):
                lora_count += 1
                print(f"LoRA module found: {name}")

        print(f"\nTotal LoRA modules: {lora_count}")
        print("Test passed!")

    finally:
        # Clean up model parallel groups
        parallel_state.destroy_model_parallel()


def _test_lora_add_2nd_lora():
    """Test LoRA forward pass with Megatron model."""
    # Initialize model parallel groups with proper CUDA RNG seed
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234)

    try:
        # Create model
        model = _gpt_model_provider(tp_size=1)

        # Create input tokens
        prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

        # Run forward pass
        output = megatron_prefill(model, prompt_tokens)
        print(
            f"Forward pass successful! Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}"
        )

        # Now test with LoRA

        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*attention*": {"rank": 32, "scale": 1},
                "*mlp*": {"rank": 64, "scale": 1},
            },
        }

        lora_2d_config = {
            "adapter_type": "lora",
            "adapter_name": "2nd",
            "adapter_cfg": {
                "*attention*": {"rank": 128, "scale": 1},
                "*mlp*": {"rank": 128, "scale": 1},
            },
        }

        # # Apply LoRA
        model = mtp.update_model(model, lora_config)
        print("LoRA adapters added successfully!")
        model = mtp.update_model(model, lora_2d_config)

        # Test forward pass with LoRA
        output_lora = megatron_prefill(model, prompt_tokens)
        print(
            f"LoRA forward pass successful! Output shape: {output_lora.shape if hasattr(output_lora, 'shape') else 'N/A'}"
        )

        # Check if LoRA modules were added
        lora_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "_lora_adapters"):
                lora_count += 1
                print(f"LoRA module found: {name}")

        print(f"\nTotal LoRA modules: {lora_count}")
        print("Test passed!")

    finally:
        # Clean up model parallel groups
        parallel_state.destroy_model_parallel()


def _test_lora_save_and_restore():
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234)

    try:
        model_ref = _gpt_model_provider(tp_size=1)
        model_test = _gpt_model_provider(tp_size=1)
        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*attention*": {"rank": 32, "scale": 1},
                "*mlp*": {"rank": 64, "scale": 1},
            },
        }
        model_ref = mtp.update_model(model_ref, lora_config)
        state_dict = copy.deepcopy(model_ref.state_dict())
        tmp_path = "./model_ref"
        save_distributed_checkpoint(tmp_path, model_ref)
        save_sharded_modelopt_state([model_ref], tmp_path)
        restore_sharded_modelopt_state([model_test], tmp_path)
        model_test = load_distributed_checkpoint(tmp_path, model_test)

        prompt_tokens = torch.randint(
            0, model_test.vocab_size, (2, model_test.max_sequence_length)
        ).cuda()

        # Run forward pass
        output_test = megatron_prefill(model_test, prompt_tokens)
        output_ref = megatron_prefill(model_ref, prompt_tokens)
        print(
            f"Forward pass successful! Output shape: {output_test.shape if hasattr(output_test, 'shape') else 'N/A'}"
        )
        print(output_test)
        print(output_ref)

    finally:
        # Clean up model parallel groups
        parallel_state.destroy_model_parallel()


def _test_lora_save_and_restore_with2loras():
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234)

    try:
        model_ref = _gpt_model_provider(tp_size=1)
        model_test = _gpt_model_provider(tp_size=1)
        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*attention*": {"rank": 32, "scale": 1},
                "*mlp*": {"rank": 64, "scale": 10},
            },
        }
        lora_2d_config = {
            "adapter_type": "lora",
            "adapter_name": "2nd",
            "adapter_cfg": {
                "*attention*": {"rank": 128, "scale": 1},
                "*mlp*": {"rank": 128, "scale": 1},
            },
        }
        lora_3d_config = {
            "adapter_type": "lora",
            "adapter_name": "3rd",
            "adapter_cfg": {
                "*attention*": {"rank": 128, "scale": 1},
                "*mlp*": {"rank": 128, "scale": 1},
            },
        }
        model_ref = mtp.update_model(model_ref, lora_config)
        model_ref = mtp.update_model(model_ref, lora_2d_config)
        tmp_path = "./model_ref"
        save_distributed_checkpoint(tmp_path, model_ref)
        save_sharded_modelopt_state([model_ref], tmp_path)
        restore_sharded_modelopt_state([model_test], tmp_path)
        model_test = load_distributed_checkpoint(tmp_path, model_test)
        # model_test = mtp.update_model(model_test, lora_3d_config)

        # Debug: Check active adapters
        print("\n=== Active Adapters ===")
        for name, module in model_test.named_modules():
            if hasattr(module, "_lora_adapters") and module._lora_adapters:
                print(
                    f"{name}: adapters={list(module._lora_adapters.keys())}, active={list(module._active_adapters)}"
                )
                break  # Just show one module as example

        prompt_tokens = torch.randint(
            0, model_test.vocab_size, (2, model_test.max_sequence_length)
        ).cuda()

        # Run forward pass
        output_test = megatron_prefill(model_test, prompt_tokens)
        output_ref = megatron_prefill(model_ref, prompt_tokens)
        print(
            f"Forward pass successful! Output shape: {output_test.shape if hasattr(output_test, 'shape') else 'N/A'}"
        )
        print(output_test)
        print(output_ref)

    finally:
        # Clean up model parallel groups
        parallel_state.destroy_model_parallel()


def _test_quantize_then_lora():
    """Test LoRA forward pass with Megatron model."""
    # Initialize model parallel groups with proper CUDA RNG seed
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234)

    try:
        model = _gpt_model_provider(tp_size=1)
        prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
        lora_config = {
            "adapter_type": "lora",
            "adapter_name": "default",
            "adapter_cfg": {
                "*attention*": {"rank": 32, "scale": 1},
                "*mlp*": {"rank": 64, "scale": 1},
            },
        }

        def forward_func(mod):
            output = megatron_prefill(model, prompt_tokens)

        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_func)
        model = mtp.update_model(model, lora_config)
        lora_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "_lora_adapters"):
                lora_count += 1
                print(f"LoRA module found: {name}")
        print(f"\nTotal LoRA modules: {lora_count}")
        output_lora_quant = megatron_prefill(model, prompt_tokens)
        print(
            f"LoRA forward pass successful! Output shape: {output_lora_quant.shape if hasattr(output_lora_quant, 'shape') else 'N/A'}"
        )
        print("Test passed!")
    finally:
        # Clean up model parallel groups
        parallel_state.destroy_model_parallel()


def _test_quantize_then_lora_save_restore():
    pass


def _test_lora_then_quantize():
    pass


def _test_lora_then_quantize_save_restore():
    pass


def _test_disable_lora():
    pass


def _test_disable_lora_restore():
    pass


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    os.makedirs(checkpoint_path, exist_ok=True)
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def main():
    """Main function to setup distributed and run test."""
    # Setup distributed environment
    if not torch.distributed.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    try:
        # _test_lora_forward()
        # _test_lora_save_and_restore()
        # _test_lora_add_2nd_lora()
        # _test_lora_save_and_restore_with2loras()
        _test_quantize_then_lora()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
