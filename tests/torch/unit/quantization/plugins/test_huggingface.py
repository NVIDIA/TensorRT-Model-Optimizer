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

import os
import warnings

import pytest
import torch
import torch.nn as nn
from _test_utils.torch_misc import set_seed

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantLinear, QuantModuleRegistry

transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class HFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # initialization is (out_features, in_features) instead of (in_features, out_features)
            transformers.modeling_utils.Conv1D(5, 3),
            nn.ReLU(),
            transformers.modeling_utils.Conv1D(5, 5),
        )

    def forward(self, x):
        return self.net(x)


class PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            QuantLinear(3, 5),
            nn.ReLU(),
            QuantLinear(5, 5),
        )

    def forward(self, x):
        return self.net(x)


def test_convert_conv1d():
    set_seed()
    assert transformers.modeling_utils.Conv1D in QuantModuleRegistry

    model_ref = HFModel()
    model_test = HFModel()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(module, transformers.modeling_utils.Conv1D):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = torch.randn(2, 3)
    out_1 = model_ref(x)
    out_2 = model_test(x)

    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attribute(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attribute(model_test, "*weight_quantizer", {"enable": True})
    model_ref = PytorchModel()
    model_ref.load_state_dict(model_test.state_dict())

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)


@pytest.mark.skipif(not hasattr(transformers.models, "dbrx"), reason="DBRX is not available")
def test_dbrx():
    from transformers.models.dbrx.configuration_dbrx import DbrxConfig, DbrxFFNConfig
    from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts, DbrxFFN

    assert DbrxExperts in QuantModuleRegistry
    assert DbrxExpertGLU in QuantModuleRegistry

    config = DbrxConfig(
        ffn_config=DbrxFFNConfig(ffn_hidden_size=8, moe_num_experts=2), hidden_size=32
    )

    model_ref = DbrxFFN(config)
    model_test = DbrxFFN(config)
    with torch.no_grad():
        model_ref.experts.mlp.w1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.v1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.w2.copy_(torch.randn(16, 32))

    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)

    expertglu_ref = model_ref.experts.mlp
    expertglu_test = model_test.experts.mlp

    assert hasattr(expertglu_test, "w1_linear") and not hasattr(expertglu_test, "w1")
    assert hasattr(expertglu_test, "v1_linear") and not hasattr(expertglu_test, "v1")
    assert hasattr(expertglu_test, "w2_linear") and not hasattr(expertglu_test, "w2")

    assert torch.allclose(
        torch.concat([weight for weight in expertglu_test.w1_linear.parameters()], dim=0),
        expertglu_ref.w1,
    )

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = torch.randn(1, 4, 32)
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1[0], out_2[0])


def test_autoquantize_huggingface():
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
    )
    input_ids = torch.randint(0, 128, (1, 4))

    warnings.filterwarnings(
        "error", message="AutoQuantize: Error enabling gradient checkpointing for huggingface model"
    )

    with pytest.warns(
        UserWarning,
        match="AutoQuantize: Huggingface model detected - Enabling gradient checkpointing. ",
    ):
        best_model, search_history = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 11.0},
            quantization_formats=["INT8_DEFAULT_CFG", None],
            data_loader=[{"input_ids": input_ids, "labels": input_ids} for _ in range(2)],
            forward_step=lambda model, batch: model(**batch),
            loss_func=lambda output, data: output.loss,
            num_calib_steps=2,
            num_score_steps=2,
            verbose=True,
        )


@pytest.mark.parametrize("model_cls", [LlamaForCausalLM, AutoModelForCausalLM])
@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.NF4_REAL_QUANT_CFG,
        mtq.INT4_AWQ_REAL_QUANT_CFG,
    ],
)
def test_quantized_transformers_save_restore(tmpdir, model_cls, quant_config):
    # update config to fit test cases
    if quant_config == mtq.NF4_REAL_QUANT_CFG:
        # reduce block sizes for simple testing models
        quant_config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {
            -1: 16,
            "scale_bits": 8,
            "scale_block_sizes": {-1: 16},
        }
    if quant_config == mtq.INT4_AWQ_REAL_QUANT_CFG:
        quant_config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 16}

    def create_base_model():
        base_model_path = tmpdir + "/base_model"
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
            )
        )
        model.save_pretrained(base_model_path)
        return base_model_path

    mto.enable_huggingface_checkpointing()

    model_ref = model_cls.from_pretrained(create_base_model())
    mtq.quantize(model_ref, quant_config, lambda model: model(**model.dummy_inputs))
    model_ref.save_pretrained(tmpdir + "/modelopt_model")
    assert os.path.exists(tmpdir + "/modelopt_model/modelopt_state.pth")

    model_test = model_cls.from_pretrained(tmpdir + "/modelopt_model")

    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for module in model_test.modules():
        if hasattr(module, "_is_hf_initialized"):
            delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)

    assert model_ref_state == model_test_state

    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.eval()
    output_ref = model_ref(**inputs).logits
    output_test = model_test(**inputs).logits
    assert torch.allclose(output_ref, output_test)
