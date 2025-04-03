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

import pytest
import torch
import torch.distributed
import torch.distributed as dist

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import is_quantized_linear
from modelopt.torch.utils import torch_to

INT4_AWQ_FULL_CFG = mtq.INT4_AWQ_CFG.copy()

INT4_AWQ_FULL_CFG["algorithm"] = "awq_full"

INT4_AWQ_CLIP_CFG = mtq.INT4_AWQ_CFG.copy()
INT4_AWQ_CLIP_CFG["algorithm"] = "awq_clip"

# SVDQuant test cfg
INT4_SVDQUANT_CFG = mtq.INT4_AWQ_CFG.copy()
INT4_SVDQUANT_CFG["algorithm"] = {"method": "svdquant", "lowrank": 8}

# SVDQuant test cfg
FP4_SVDQUANT_CFG = mtq.NVFP4_AWQ_LITE_CFG.copy()
FP4_SVDQUANT_CFG["algorithm"] = {"method": "svdquant", "lowrank": 8}


def quantize_model_and_forward(model, config, calib_data, compress=False):
    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    model = mtq.quantize(model, config, forward_loop=forward_loop)
    if compress:
        mtq.compress(model)

    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)

    model.train()

    # make sure we can run the forward and backward on the quantized model
    forward_loop(model, run_backward=True)


def save_restore_test(model_cls, device, quant_config, compress=False):
    # test restoring to an unquantized model
    model_quant = model_cls().to(device)
    model_ref = model_cls().to(device)

    calib_data = [model_quant.get_input().to(device) for _ in range(2)]
    quantize_model_and_forward(model_quant.to(device), quant_config, calib_data, compress)

    state_dict = mto.modelopt_state(model_quant)

    mto.restore_from_modelopt_state(model_ref, state_dict)
    model_ref.load_state_dict(model_quant.state_dict())
    assert torch.allclose(model_quant(calib_data[0]), model_ref(calib_data[0]))

    # gpu: test restoring to a model on cpu. If the quantizer states are not initialized correctly,
    # the buffers will be created on cuda and this test will fail
    model_ref = model_cls().to("cpu")
    state_dict = torch_to(state_dict, device="cuda" if torch.cuda.is_available() else "cpu")
    mto.restore_from_modelopt_state(model_ref, state_dict)
    model_ref.load_state_dict(model_quant.state_dict())
    model_ref(calib_data[0].to("cpu"))  # make sure all the buffers are created in the right device
    model_ref.to(device)
    assert torch.allclose(model_quant(calib_data[0]), model_ref(calib_data[0]))

    # Test that smoothquant is restored correctly
    if quant_config == mtq.INT8_SMOOTHQUANT_CFG:
        for name, module in model_ref.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert module.input_quantizer.axis is None

    # test restoring to a quantized model
    model_ref = model_cls().to(device)
    quantize_model_and_forward(model_ref, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, calib_data)
    with pytest.raises(AssertionError, match="Model already has modelopt state!"):
        mto.restore_from_modelopt_state(model_ref, state_dict)


def tensor_parallel_test_helper(model, config, tp_group, dp_group):
    # The input to fist layer, the column parallel should be the same across all tp ranks
    calib_data = model.get_dummy_input().cuda()
    dist.all_reduce(calib_data, op=dist.ReduceOp.AVG, group=tp_group)

    def forward_loop(model):
        model(calib_data)

    model = mtq.quantize(model, config, forward_loop)

    # Sanity check
    forward_loop(model)

    if config in [mtq.INT8_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG, mtq.INT8_SMOOTHQUANT_CFG]:
        # Lets check the amax for row parallel input quantizer; it should be the same across all tp ranks
        activation_amax = model.fc2.input_quantizer.amax.clone()
        dist.all_reduce(activation_amax, op=dist.ReduceOp.MAX, group=tp_group)
        assert torch.allclose(activation_amax, model.fc2.input_quantizer.amax)

        # Lets check the row parallel weight amax; it should be the same across all tp ranks
        weight_amax = model.fc2.weight_quantizer.amax.clone()
        dist.all_reduce(weight_amax, op=dist.ReduceOp.MAX, group=tp_group)
        assert torch.allclose(weight_amax, model.fc2.weight_quantizer.amax)

    if config in [mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG, mtq.W4A8_AWQ_BETA_CFG]:
        # Lets check the column parallel pre_quant_scale; it should be the same across all tp ranks
        input_quantizer = model.fc1.input_quantizer
        pre_quant_scale = input_quantizer.pre_quant_scale.clone()
        dist.all_reduce(pre_quant_scale, op=dist.ReduceOp.MAX, group=tp_group)
        assert torch.allclose(pre_quant_scale, input_quantizer.pre_quant_scale)

    dist.destroy_process_group()


def auto_quantize_helper(model):
    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=["INT4_BLOCKWISE_WEIGHT_ONLY_CFG", "INT8_DEFAULT_CFG", None],
        data_loader=[model.get_dummy_input().cuda() for _ in range(2)],
        forward_step=lambda model, batch: model(batch),
        forward_backward_step=lambda model, batch: model(batch).sum().backward(),
        num_calib_steps=2,
        num_score_steps=2,
        verbose=True,
    )
    search_state_list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(search_state_list, search_state)

    search_state_rank0 = search_state_list[0]
    for search_state in search_state_list[1:]:
        assert search_state == search_state_rank0
