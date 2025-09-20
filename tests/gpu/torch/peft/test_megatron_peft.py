import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch_dist.plugins.megatron_common import (
    get_mcore_gpt_model,
    initialize_for_megatron,
)

skip_if_no_megatron()


import modelopt.torch.peft as mtp
from modelopt.torch.peft.config import kaiming_init, zero_init
from modelopt.torch.peft.lora.layer import LoRAModule
from modelopt.torch.utils.plugins import megatron_prefill

DEFAULT_LORA_CFG_TEST = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "lora_a_init": kaiming_init,
            "lora_b_init": zero_init,
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
            "lora_a_init": kaiming_init,
            "lora_b_init": kaiming_init,
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


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_TEST,
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_forward_with_one_lora(lora_config):
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
    original_output = megatron_prefill(model, prompt_tokens)
    mtp.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)
    assert lora_output.shape == original_output.shape
    if lora_config == DEFAULT_LORA_CFG_TEST:
        assert torch.allclose(lora_output, original_output)
    else:
        assert not torch.allclose(lora_output, original_output)

    mtp.disable_adapters(model)
    lora_disabled_output = megatron_prefill(model, prompt_tokens)
    assert torch.allclose(lora_disabled_output, original_output)

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            assert hasattr(module, f"lora_a_{lora_config['adapter_name']}")
            assert hasattr(module, f"lora_b_{lora_config['adapter_name']}")


@pytest.mark.parametrize(
    "lora_config_1",
    [
        DEFAULT_LORA_CFG_TEST,
    ],
)
@pytest.mark.parametrize(
    "lora_config_2",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_forward_with_two_loras(lora_config_1, lora_config_2):
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
    mtp.update_model(model, lora_config_1)
    lora_1_output = megatron_prefill(model, prompt_tokens)
    mtp.update_model(model, lora_config_2)
    lora_2_output = megatron_prefill(model, prompt_tokens)

    assert not torch.allclose(lora_1_output, lora_2_output)
    assert lora_1_output.shape == lora_2_output.shape

    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            assert hasattr(module, f"lora_a_{lora_config_1['adapter_name']}")
            assert hasattr(module, f"lora_b_{lora_config_1['adapter_name']}")

            assert hasattr(module, f"lora_a_{lora_config_2['adapter_name']}")
            assert hasattr(module, f"lora_b_{lora_config_2['adapter_name']}")


def test_forward_with_lora_quantize():
    pass


def test_forward_with_quantize_lora():
    pass


def test_one_lora_save_restore():
    pass


def test_two_loras_save_restore():
    pass


def test_one_lora_quantize_save_restore():
    pass


def test_two_loras_quantize_save_restore():
    pass
