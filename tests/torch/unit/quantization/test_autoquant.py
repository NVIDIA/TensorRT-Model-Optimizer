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

import copy
import io

import pytest
import torch
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_quantization.models import SimpleConv, SimpleConvLinear, SimpleLinear

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.core.torch.quantization.algorithms import QuantRecipe, QuantRecipeHparam
from modelopt.torch.utils.distributed import DistributedProcessGroup


class _AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(32, 32)
        self.k_proj = torch.nn.Linear(32, 32)
        self.v_proj = torch.nn.Linear(32, 32)
        self.o_proj = torch.nn.Linear(32, 32)

    def forward(self, x):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            x = layer(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AttentionLayer()
        self.mlp = torch.nn.Linear(32, 32)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x

    def get_input(self):
        return torch.randn(1, 4, 32)


@pytest.mark.parametrize(
    "name, other_name, is_less_than",
    [
        ("FP8_DEFAULT_CFG", None, True),
        ("NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG", True),
        (None, "INT8_DEFAULT_CFG", False),
    ],
)
def test_quant_recipe(name, other_name, is_less_than):
    qr_this = QuantRecipe(name)
    qr_other = QuantRecipe(other_name)
    assert (qr_this < qr_other) == is_less_than

    qr_this_duplicate = QuantRecipe(name)
    assert qr_this_duplicate in set([qr_this])


def test_quant_recipe_hparam():
    model_test = torch.nn.Linear(4, 16)
    model_ref = torch.nn.Linear(4, 16)
    model_ref.load_state_dict(model_test.state_dict())

    model_test = mtq.quantize(model_test, mtq.INT8_DEFAULT_CFG)
    model_ref = mtq.quantize(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)

    search_recipes = [
        QuantRecipe("INT8_DEFAULT_CFG"),
        QuantRecipe("INT4_BLOCKWISE_WEIGHT_ONLY_CFG"),
    ]
    hparam = QuantRecipeHparam(
        search_recipes,
        original=search_recipes[0],
        nn_modules=[model_test],
    )
    model_test._register_hparam("quant_recipe", hparam)
    assert model_test.quant_recipe.name == "INT8_DEFAULT_CFG"
    assert model_test.get_hparam("quant_recipe").choices == sorted(
        search_recipes + [QuantRecipe(None)]
    )

    model_test.quant_recipe = QuantRecipe("INT4_BLOCKWISE_WEIGHT_ONLY_CFG")
    inputs = torch.randn(1, 4, 4)
    output_test = model_test(inputs)
    output_ref = model_ref(inputs)

    assert torch.allclose(output_test, output_ref)


@pytest.mark.parametrize(
    "model_cls",
    [SimpleConv, SimpleConvLinear, SimpleLinear, TransformerBlock],
)
@pytest.mark.parametrize(
    "search_formats",
    [
        ["INT4_BLOCKWISE_WEIGHT_ONLY_CFG", "INT8_DEFAULT_CFG", None],
        ["INT4_AWQ_CFG", "INT8_SMOOTHQUANT_CFG", None],
    ],
)
def test_auto_quantize(model_cls, search_formats):
    model = model_cls()

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 11.0},
        quantization_formats=search_formats,
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )
    assert isinstance(search_history, dict)
    assert search_history["best"]["is_satisfied"]

    if model_cls == TransformerBlock:
        hparam = model.attn.q_proj.get_hparam("quant_recipe")
        for layer in [model.attn.k_proj, model.attn.v_proj]:
            assert layer.get_hparam("quant_recipe") == hparam
        assert ("attn.q_proj.quant_recipe" in search_history["candidate_stats"]) != (
            "attn.k_proj.quant_recipe" in search_history["candidate_stats"]
        )

    # test restore
    buffer = io.BytesIO()
    mto.save(best_model, buffer)
    buffer.seek(0)
    new_model = model_cls()
    new_model = mto.restore(new_model, buffer)

    input = model.get_input()
    output_ref = best_model(input)
    output_test = new_model(input)
    assert torch.allclose(output_ref, output_test)


def test_auto_quantize_disable():
    model = TransformerBlock()
    search_formats = ["INT4_BLOCKWISE_WEIGHT_ONLY_CFG", "INT8_DEFAULT_CFG", None]

    def loss_func(output):
        return output.sum()

    best_model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 5.0},
        quantization_formats=search_formats,
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        disabled_layers=["*mlp*"],
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    assert not best_model.mlp.input_quantizer.is_enabled


def test_auto_quantize_vs_quantize():
    model_ref = SimpleLinear()
    state_dict = copy.deepcopy(model_ref.state_dict())
    dataloader = [model_ref.get_input() for _ in range(2)]

    def calibrate(model):
        for input in dataloader:
            model(input)

    mtq.quantize(model_ref, mtq.INT8_SMOOTHQUANT_CFG, calibrate)

    model_test = SimpleLinear()
    model_test.load_state_dict(state_dict)

    best_model, search_history = mtq.auto_quantize(
        model_test,
        constraints={"effective_bits": 11.0},
        quantization_formats=["INT8_SMOOTHQUANT_CFG"],
        data_loader=dataloader,
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    assert torch.allclose(best_model(dataloader[0]), model_ref(dataloader[0]))


INT4INT8_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": 8, "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}


@pytest.mark.parametrize("config", [mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG, INT4INT8_AWQ_CFG])
def test_pqs_folding(config):
    model_ref = SimpleLinear()
    state_dict_ref = copy.deepcopy(model_ref.state_dict())
    inputs = model_ref.get_input()
    mtq.quantize(model_ref, config, lambda model: model(inputs))

    model_test = SimpleLinear()
    model_test.load_state_dict(state_dict_ref)
    QuantRecipe.disable_folding_pqs_to_weights()
    mtq.quantize(model_test, config, lambda model: model(inputs))

    assert torch.allclose(model_ref(inputs), model_test(inputs))

    QuantRecipe.fold_pqs_to_weights(model_test)
    assert torch.allclose(model_ref(inputs), model_test(inputs))


def _test_data_parallel_auto_quantize(rank, size):
    model = SimpleLinear()

    model, search_history = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 11.0},
        quantization_formats=["INT8_SMOOTHQUANT_CFG", None],
        data_loader=[model.get_input() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )

    search_history_rank0 = DistributedProcessGroup.get_dist_syncd_obj(
        search_history if rank == 0 else None, DistributedProcessGroup(None), lambda a: a[0]
    )

    # Assert that the costs, scores and searched recipes are the same across all ranks
    assert search_history == search_history_rank0

    assert search_history["best"]["is_satisfied"]


def test_data_parallel_auto_quantize():
    spawn_multiprocess_job(4, _test_data_parallel_auto_quantize, backend="gloo")
