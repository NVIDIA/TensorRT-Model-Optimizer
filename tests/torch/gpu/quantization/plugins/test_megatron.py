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
from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_mcore_dist_ckpt_is_not_supported, skip_if_no_megatron
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_misc import set_seed
from _test_utils.torch_quantization.models import RegularQuantModelForTP
from _test_utils.torch_quantization.quant_utils import get_model_size
from _test_utils.torch_quantization.quantize_common import (
    auto_quantize_helper,
    tensor_parallel_test_helper,
)

skip_if_no_megatron()

import megatron.core.tensor_parallel.layers as megatron_parallel
from _test_utils.torch_dist.plugins.megatron_common import (
    MegatronModel,
    get_mcore_gpt_model,
    initialize_for_megatron,
    run_mcore_gpt_inference,
    sharded_state_dict_test_helper,
)
from megatron.core.parallel_state import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry

SEED = 1234


def test_convert_megatron_parallel_linear(distributed_setup_size_1):
    initialize_for_megatron(seed=SEED)
    set_seed(SEED)

    assert megatron_parallel.ColumnParallelLinear in QuantModuleRegistry

    model_ref = MegatronModel().cuda()
    model_test = MegatronModel().cuda()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(
            module, (megatron_parallel.ColumnParallelLinear, megatron_parallel.RowParallelLinear)
        ):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = model_ref.get_dummy_input().cuda()
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attribute(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attribute(model_test, "*weight_quantizer", {"enable": True})
    model_ref = RegularQuantModelForTP().cuda()
    model_ref.load_state_dict(model_test.state_dict(), strict=False)

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)

    # Clean up since this is not a spawned process
    destroy_model_parallel()


def _test_tensor_parallel_helper(config, rank, size):
    initialize_for_megatron(tensor_model_parallel_size=2, seed=SEED)
    model = MegatronModel(size).cuda()

    tensor_parallel_test_helper(
        model, config, get_tensor_model_parallel_group(), get_data_parallel_group()
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
def test_tensor_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2, job=partial(_test_tensor_parallel_helper, config), backend="nccl"
    )


def _gpt_model_provider(tp_size: int, hidden_size=256):
    """Build the model."""
    gpt_model = get_mcore_gpt_model(
        tensor_model_parallel_size=tp_size,
        num_layers=4,
        ffn_hidden_size=None,
        num_attention_heads=4,
        activation_func="squared_relu",
        transformer_impl="local",
        hidden_size=hidden_size,
    )
    return gpt_model.cuda().eval()


def _test_sharded_state_dict(tmpdir, config, hidden_size, rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size, seed=SEED)

    model_ref = _gpt_model_provider(size, hidden_size)
    model_test = _gpt_model_provider(size, hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return run_mcore_gpt_inference(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, config, forward_fn)

    for module in model_ref.modules():
        if hasattr(module, "_amax_for_smoothing"):
            delattr(module, "_amax_for_smoothing")

    sharded_state_dict_test_helper(tmpdir, model_ref, model_test, forward_fn)


partial_quant_config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
partial_quant_config["quant_cfg"].update(
    {
        "*.0.*": {"enable": False},
        "*.3.*": {"enable": False},
        "*.1.*linear_fc2.weight_quantizer": {"enable": False},
    }
)

mixed_precision_config = copy.deepcopy(mtq.W4A8_AWQ_BETA_CFG)
mixed_precision_config["quant_cfg"].update(
    {
        "*.1.*": {"enable": False},
        "*.2.*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.2.*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.3.*weight_quantizer.0": {"num_bits": 8, "axis": 0},
        "*.3.*weight_quantizer.1": {"enable": False},
        "*.3.*input_quantizer": {"num_bits": 8, "axis": None},
    }
)

mixed_block_size_config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
mixed_block_size_config["quant_cfg"].update(
    {
        "*.1.*": {"enable": False},
        "*.2.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 256}, "enable": True},
        "*.2.*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.3.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*.3.*input_quantizer": {"num_bits": 8, "axis": None},
    }
)


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        partial_quant_config,
        mixed_precision_config,
        mixed_block_size_config,
    ],
)
@pytest.mark.parametrize("hidden_size", [256, 320])
def test_sharded_state_dict(need_2_gpus, tmpdir, config, hidden_size):
    skip_if_mcore_dist_ckpt_is_not_supported()
    spawn_multiprocess_job(
        size=2, job=partial(_test_sharded_state_dict, tmpdir, config, hidden_size), backend="nccl"
    )


@pytest.mark.parametrize(
    "hidden_size",
    [256, 320],
)
def test_regular_state_dict(distributed_setup_size_1, hidden_size):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=SEED)

    model_ref = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return run_mcore_gpt_inference(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, mixed_precision_config, forward_fn)

    mto.restore_from_modelopt_state(model_test, mto.modelopt_state(model_ref))
    model_test.load_state_dict(model_ref.state_dict())

    logits_ref = forward_fn(model_ref)
    logits_test = forward_fn(model_test)
    assert torch.allclose(logits_ref, logits_test)

    # Clean up since this is not a spawned process
    destroy_model_parallel()


def _test_auto_quantize_helper(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size)
    model = MegatronModel().cuda()
    auto_quantize_helper(model)


def test_auto_quantize(need_2_gpus):
    spawn_multiprocess_job(size=2, job=_test_auto_quantize_helper, backend="nccl")


def test_fp8_real_quantize(distributed_setup_size_1):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=SEED)
    hidden_size = 256
    config = mtq.FP8_BLOCKWISE_REAL_QUANT_CFG

    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_fn(model):
        return run_mcore_gpt_inference(model, prompt_tokens)

    # ref ouptut
    logits_ref = forward_fn(model)

    # real quant the model
    cur_mem = get_model_size(model)
    real_quant_model = mtq.quantize(model, config, forward_fn)
    real_quant_mem = get_model_size(model)

    assert real_quant_mem < cur_mem / 2, "Memory after real quantization is not reduced."

    # output with real quant
    logits_real_quant = forward_fn(real_quant_model)
    torch.allclose(logits_ref, logits_real_quant)

    assert real_quant_mem < cur_mem
