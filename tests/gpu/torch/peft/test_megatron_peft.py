from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import (
    get_mcore_gpt_model,
    initialize_for_megatron,
)

skip_if_no_megatron()


import modelopt.torch.peft as mtpf
from modelopt.torch.peft.lora.layer import LoRAModule
from modelopt.torch.utils.plugins import megatron_prefill

DEFAULT_LORA_CFG_TEST = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "lora_a_init": "kaiming_init",
            "lora_b_init": "zero_init",
            "enable": True,
        },
    },
}

DEFAULT_LORA_CFG_RANDOM_INIT_TEST = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "lora_a_init": "kaiming_init",
            "lora_b_init": "kaiming_init",
            "enable": True,
        },
    },
}

DEFAULT_LORA_CFG_RANDOM_INIT_SMALL_RANK_TEST = {
    "adapter_type": "lora",
    "adapter_name": "small",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1,
            "lora_a_init": "kaiming_init",
            "lora_b_init": "kaiming_init",
            "enable": True,
        },
    },
}

LARGE_SCALE_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "large_scale",
    "adapter_cfg": {
        "*": {
            "rank": 16,
            "scale": 10.0,
            "lora_a_init": "kaiming_init",
            "lora_b_init": "zero_init",
            "enable": True,
        },
    },
}

SELECTIVE_LAYER_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "selective",
    "adapter_cfg": {
        "*": {"enable": False},
        "*self_attention*": {
            "rank": 16,
            "scale": 1,
            "lora_a_init": "kaiming_init",
            "lora_b_init": "zero_init",
            "enable": True,
        },
    },
}


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


def _test_forward_with_one_lora(lora_config, rank, size):
    """Test forward pass with a single LoRA adapter with various configurations."""
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    original_output = megatron_prefill(model, prompt_tokens)
    mtpf.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)
    assert lora_output.shape == original_output.shape
    if lora_config == DEFAULT_LORA_CFG_RANDOM_INIT_TEST:
        assert not torch.allclose(lora_output, original_output, rtol=1e-5)
    else:
        assert torch.allclose(lora_output, original_output, rtol=1e-5), (
            f"{lora_output}, {original_output}"
        )
    mtpf.disable_adapters(model)
    lora_disabled_output = megatron_prefill(model, prompt_tokens)
    assert torch.allclose(lora_disabled_output, original_output, rtol=1e-5)
    mtpf.enable_adapters(model)
    lora_reenabled_output = megatron_prefill(model, prompt_tokens)
    assert torch.allclose(lora_reenabled_output, lora_output, rtol=1e-5)
    lora_module_count = 0
    lora_with_adapter_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRAModule):
            lora_module_count += 1

            if lora_config == SELECTIVE_LAYER_LORA_CFG:
                if "self_attention" in name:
                    # Only self_attention modules should have the adapter
                    assert hasattr(module, f"lora_a_{lora_config['adapter_name']}")
                    assert hasattr(module, f"lora_b_{lora_config['adapter_name']}")
                    assert lora_config["adapter_name"] in module._lora_adapters
                    assert module._lora_adapters[lora_config["adapter_name"]]["enable"]
                    lora_with_adapter_count += 1
                else:
                    # Other modules should NOT have the adapter at all
                    assert not hasattr(module, f"lora_a_{lora_config['adapter_name']}")
                    assert not hasattr(module, f"lora_b_{lora_config['adapter_name']}")
                    assert lora_config["adapter_name"] not in module._lora_adapters
            else:
                # For non-selective configs, all LoRA modules should have the adapter
                assert hasattr(module, f"lora_a_{lora_config['adapter_name']}")
                assert hasattr(module, f"lora_b_{lora_config['adapter_name']}")
                lora_with_adapter_count += 1

    assert lora_module_count > 0
    assert lora_with_adapter_count > 0


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_TEST,
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
        SELECTIVE_LAYER_LORA_CFG,
    ],
)
def test_forward_with_one_lora(lora_config):
    spawn_multiprocess_job(
        size=1, job=partial(_test_forward_with_one_lora, lora_config), backend="nccl"
    )


def _test_forward_with_two_loras(lora_config_1, lora_config_2, rank, size):
    """Test forward pass with two LoRA adapters and adapter switching."""
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    original_output = megatron_prefill(model, prompt_tokens)
    mtpf.update_model(model, lora_config_1)
    # output from the first lora only
    lora_1_output = megatron_prefill(model, prompt_tokens)

    mtpf.update_model(model, lora_config_2)

    mtpf.disable_adapters(model, adapters_to_disable=[lora_config_1["adapter_name"]])
    mtpf.enable_adapters(model, adapters_to_enable=[lora_config_2["adapter_name"]])

    # output from the 2nd lora only
    lora_2_output = megatron_prefill(model, prompt_tokens)

    assert lora_1_output.shape == lora_2_output.shape
    # Should not be the same
    assert not torch.allclose(lora_1_output, lora_2_output)

    mtpf.enable_adapters(model, adapters_to_enable=[lora_config_1["adapter_name"]])
    mtpf.enable_adapters(model, adapters_to_enable=[lora_config_2["adapter_name"]])
    lora_all_output = megatron_prefill(model, prompt_tokens)

    assert not torch.allclose(lora_all_output, lora_1_output)
    assert not torch.allclose(lora_all_output, lora_2_output)

    mtpf.disable_adapters(model)
    both_disabled_output = megatron_prefill(model, prompt_tokens)
    assert torch.allclose(both_disabled_output, original_output)

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            assert hasattr(module, f"lora_a_{lora_config_1['adapter_name']}")
            assert hasattr(module, f"lora_b_{lora_config_1['adapter_name']}")
            assert hasattr(module, f"lora_a_{lora_config_2['adapter_name']}")
            assert hasattr(module, f"lora_b_{lora_config_2['adapter_name']}")
            assert len(module._lora_adapters) == 2


@pytest.mark.parametrize(
    ("lora_config_1", "lora_config_2"),
    [
        (DEFAULT_LORA_CFG_RANDOM_INIT_TEST, DEFAULT_LORA_CFG_RANDOM_INIT_SMALL_RANK_TEST),
    ],
)
def test_forward_with_two_loras(lora_config_1, lora_config_2):
    spawn_multiprocess_job(
        size=1,
        job=partial(_test_forward_with_two_loras, lora_config_1, lora_config_2),
        backend="nccl",
    )


# def test_edge_cases_and_error_handling():
#     """Test edge cases and error scenarios."""
#     hidden_size = 320
#     initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
#     model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)

#     # Test 1: Applying same adapter twice should work without issues
#     mtp.update_model(model, DEFAULT_LORA_CFG_TEST)
#     mtp.update_model(model, DEFAULT_LORA_CFG_TEST)  # Should not raise error

#     # Test 2: Disabling non-existent adapter should not raise error
#     mtp.disable_adapters(model, adapters_to_disable=["non_existent"])

#     # Test 3: Empty adapter configuration
#     empty_config = {
#         "adapter_type": "lora",
#         "adapter_name": "empty",
#         "adapter_cfg": {},
#     }
#     # This might not add any adapters but shouldn't crash
#     mtp.update_model(model, empty_config)

#     # Test 4: Very large rank (might be memory intensive, so use small model)
#     large_rank_config = {
#         "adapter_type": "lora",
#         "adapter_name": "large_rank",
#         "adapter_cfg": {
#             "*": {
#                 "rank": 128,  # Large rank relative to hidden size
#                 "scale": 1,
#                 "lora_a_init": kaiming_init,
#                 "lora_b_init": zero_init,
#                 "enable": True,
#             },
#         },
#     }
#     small_model = _gpt_model_provider(tp_size=1, hidden_size=128)
#     mtp.update_model(small_model, large_rank_config)

#     # Verify the model still works
#     prompt_tokens = torch.randint(0, small_model.vocab_size, (1, 16)).cuda()
#     output = megatron_prefill(small_model, prompt_tokens)
#     assert output is not None


# def test_adapter_gradient_flow():
#     """Test that gradients flow correctly through LoRA adapters."""
#     hidden_size = 128
#     initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
#     model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)

#     # Apply LoRA adapter
#     mtp.update_model(model, DEFAULT_LORA_CFG_RANDOM_INIT_TEST)

#     # Set model to training mode
#     model.train()

#     # Forward pass
#     prompt_tokens = torch.randint(0, model.vocab_size, (1, 16)).cuda()
#     output = megatron_prefill(model, prompt_tokens)

#     # Create a dummy loss and backward
#     loss = output.sum()
#     loss.backward()

#     # Check that LoRA parameters have gradients
#     for name, module in model.named_modules():
#         if isinstance(module, LoRAModule):
#             adapter_name = DEFAULT_LORA_CFG_RANDOM_INIT_TEST['adapter_name']
#             lora_a = getattr(module, f"lora_a_{adapter_name}")
#             lora_b = getattr(module, f"lora_b_{adapter_name}")

#             # LoRA parameters should have gradients
#             assert lora_a.grad is not None, f"lora_a in {name} has no gradient"
#             assert lora_b.grad is not None, f"lora_b in {name} has no gradient"

#             # Gradients should be non-zero
#             assert torch.any(lora_a.grad != 0), f"lora_a gradient is all zeros in {name}"
#             assert torch.any(lora_b.grad != 0), f"lora_b gradient is all zeros in {name}"


# def test_adapter_parameter_count():
#     """Test that LoRA reduces trainable parameters significantly."""
#     hidden_size = 256
#     initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
#     model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)

#     # Count original parameters
#     original_params = sum(p.numel() for p in model.parameters())

#     # Apply LoRA with small rank
#     small_rank_config = {
#         "adapter_type": "lora",
#         "adapter_name": "small",
#         "adapter_cfg": {
#             "*": {
#                 "rank": 8,
#                 "scale": 1,
#                 "lora_a_init": kaiming_init,
#                 "lora_b_init": zero_init,
#                 "enable": True,
#             },
#         },
#     }
#     mtp.update_model(model, small_rank_config)

#     # Count LoRA parameters
#     lora_params = 0
#     for module in model.modules():
#         if isinstance(module, LoRAModule):
#             for param_name, param in module.named_parameters():
#                 if "lora_" in param_name:
#                     lora_params += param.numel()

#     # LoRA parameters should be much smaller than original model
#     assert lora_params < original_params * 0.1, (
#         f"LoRA params ({lora_params}) should be < 10% of original params ({original_params})"
#     )

#     # Verify LoRA parameters exist
#     assert lora_params > 0, "No LoRA parameters found"


# def test_multiple_forward_consistency():
#     """Test that multiple forward passes produce consistent results."""
#     hidden_size = 128
#     initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
#     model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)

#     # Apply LoRA adapter
#     mtp.update_model(model, LARGE_SCALE_LORA_CFG)

#     # Set to eval mode for deterministic behavior
#     model.eval()

#     # Run multiple forward passes with same input
#     prompt_tokens = torch.randint(0, model.vocab_size, (2, 32)).cuda()
#     outputs = []
#     for _ in range(3):
#         with torch.no_grad():
#             output = megatron_prefill(model, prompt_tokens)
#             outputs.append(output)

#     # All outputs should be identical
#     for i in range(1, len(outputs)):
#         assert torch.allclose(outputs[0], outputs[i], rtol=1e-6), (
#             f"Output {i} differs from output 0"
#         )


# # Placeholder functions for future implementation
# def test_forward_with_lora_quantize():
#     """Test applying LoRA to an already quantized model."""
#     # TODO: Implement when quantization integration is ready
#     pytest.skip("Quantization integration tests not yet implemented")


# def test_forward_with_quantize_lora():
#     """Test quantizing a model that already has LoRA adapters."""
#     # TODO: Implement when quantization integration is ready
#     pytest.skip("Quantization integration tests not yet implemented")


# def test_one_lora_save_restore():
#     """Test saving and restoring a model with one LoRA adapter."""
#     # TODO: Implement when save/restore functionality is ready
#     pytest.skip("Save/restore tests not yet implemented")


# def test_two_loras_save_restore():
#     """Test saving and restoring a model with multiple LoRA adapters."""
#     # TODO: Implement when save/restore functionality is ready
#     pytest.skip("Save/restore tests not yet implemented")


# def test_one_lora_quantize_save_restore():
#     """Test save/restore of quantized model with one LoRA adapter."""
#     # TODO: Implement when quantization + save/restore is ready
#     pytest.skip("Quantization + save/restore tests not yet implemented")


# def test_two_loras_quantize_save_restore():
#     """Test save/restore of quantized model with multiple LoRA adapters."""
#     # TODO: Implement when quantization + save/restore is ready
#     pytest.skip("Quantization + save/restore tests not yet implemented")
