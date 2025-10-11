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
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import (
    MegatronModel,
    compare_amax_sync_across_expert_parallel,
    copy_weights_from_grouped_to_non_grouped,
    get_mcore_gpt_model,
    initialize_for_megatron,
    run_mcore_inference,
    sharded_state_dict_test_helper,
)
from _test_utils.torch_misc import set_seed
from _test_utils.torch_quantization.models import RegularQuantModelForTP
from _test_utils.torch_quantization.quant_utils import get_model_size
from _test_utils.torch_quantization.quantize_common import (
    auto_quantize_helper,
    data_tensor_context_parallel_test_helper,
    dp_cp_parallel_test_helper,
    tensor_parallel_test_helper,
)

skip_if_no_megatron()

from megatron.core.parallel_state import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.utils.plugins import megatron_prefill

SEED = 1234


def test_convert_megatron_parallel_linear(distributed_setup_size_1):
    initialize_for_megatron(seed=SEED)
    set_seed(SEED)

    assert ColumnParallelLinear in QuantModuleRegistry
    assert RowParallelLinear in QuantModuleRegistry

    model_ref = MegatronModel().cuda()
    model_test = MegatronModel().cuda()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for module in model_test.modules():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
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


# Unified parallelism test helper
def _test_parallelism_helper(
    config,
    rank,
    size,
    tensor_model_parallel_size=1,
    context_parallel_size=1,
    use_rank_in_seed=False,
    test_pre_quant_scale=True,
):
    """
    Unified helper for testing different parallelism configurations.
    Args:
        config: Quantization config to test
        rank: Current rank in distributed setup
        size: Total number of processes
        tensor_model_parallel_size: Size of tensor model parallel group (default: 1)
        context_parallel_size: Size of context parallel group (default: 1)
        use_rank_in_seed: Whether to add rank to seed for different data across ranks (default: False)
    """
    seed = SEED + rank if use_rank_in_seed else SEED
    initialize_for_megatron(
        tensor_model_parallel_size=tensor_model_parallel_size,
        context_parallel_size=context_parallel_size,
        seed=seed,
    )

    # Determine if we need tp_group and dp_group
    tp_group = get_tensor_model_parallel_group() if tensor_model_parallel_size > 1 else None
    dp_group = get_data_parallel_group(with_context_parallel=True)

    # Create model with appropriate parallelism settings
    model = MegatronModel(
        tp_size=tensor_model_parallel_size,
        cp_size=context_parallel_size,
        tp_group=tp_group,
    ).cuda()

    # Call the test helper with appropriate groups
    data_tensor_context_parallel_test_helper(
        model,
        config,
        dp_group=dp_group,
        tp_group=tp_group,
        test_pre_quant_scale=test_pre_quant_scale,
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
        size=2,
        job=partial(_test_parallelism_helper, config, tensor_model_parallel_size=2),
        backend="nccl",
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
def test_data_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(_test_parallelism_helper, config, use_rank_in_seed=True),
        backend="nccl",
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
def test_context_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(
            _test_parallelism_helper, config, context_parallel_size=2, use_rank_in_seed=True
        ),
        backend="nccl",
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
def test_data_tensor_context_parallel(need_8_gpus, config):
    spawn_multiprocess_job(
        size=8,
        job=partial(
            _test_parallelism_helper,
            config,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_rank_in_seed=True,
            test_pre_quant_scale=False,
        ),
        backend="nccl",
    )


# 2. Data Parallel Test
def _test_data_parallel_helper(config, rank, size):
    initialize_for_megatron(seed=SEED + rank)  # modify seed so data is different across ranks
    model = MegatronModel().cuda()

    dp_cp_parallel_test_helper(model, config, get_data_parallel_group())


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
def test_data_parallel(need_2_gpus, config):
    spawn_multiprocess_job(size=2, job=partial(_test_data_parallel_helper, config), backend="nccl")


# 3. Context Parallel Test
def _test_context_parallel_helper(config, rank, size):
    initialize_for_megatron(
        context_parallel_size=size, seed=SEED + rank
    )  # modify seed so data is different across ranks
    model = MegatronModel(cp_size=size).cuda()

    dp_cp_parallel_test_helper(model, config, get_data_parallel_group(with_context_parallel=True))


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
def test_context_parallel(need_2_gpus, config):
    spawn_multiprocess_job(
        size=2, job=partial(_test_context_parallel_helper, config), backend="nccl"
    )


# 4. DP=2 + TP=2 + CP=2 Test (on 2*2*2=8 GPUs)
def _test_data_tensor_context_parallel_helper(config, rank, size):
    initialize_for_megatron(tensor_model_parallel_size=2, context_parallel_size=2, seed=SEED + rank)
    model = MegatronModel(tp_size=2, cp_size=2).cuda()

    data_tensor_context_parallel_test_helper(
        model,
        config,
        get_data_parallel_group(with_context_parallel=True),
        get_tensor_model_parallel_group(),
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
def test_data_tensor_context_parallel(need_8_gpus, config):
    spawn_multiprocess_job(
        size=8, job=partial(_test_data_tensor_context_parallel_helper, config), backend="nccl"
    )


def _gpt_model_provider(
    tp_size: int,
    hidden_size=256,
    vocab_size=64,
    num_moe_experts=None,
    moe_grouped_gemm=False,
    meta_device=False,
    ep_size=1,
    etp_size=None,
    use_te=False,
):
    """Build the model."""

    if meta_device:
        with torch.device("meta"):
            gpt_model = get_mcore_gpt_model(
                tensor_model_parallel_size=tp_size,
                expert_model_parallel_size=ep_size,
                expert_tensor_parallel_size=etp_size,
                num_layers=4,
                ffn_hidden_size=None,
                num_attention_heads=8,
                activation_func="squared_relu",
                transformer_impl="local",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                use_cpu_initialization=meta_device,
                num_moe_experts=num_moe_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                use_te=use_te,
            )
    else:
        gpt_model = get_mcore_gpt_model(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            num_layers=4,
            ffn_hidden_size=None,
            num_attention_heads=8,
            activation_func="squared_relu",
            transformer_impl="local",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_moe_experts=num_moe_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            use_te=use_te,
        ).cuda()
    return gpt_model.eval()


def _test_sharded_state_dict(
    tmp_path, config, hidden_size, modelopt_version, compress, meta_device, moe_config, rank, size
):
    # Must disable output_layer quantization since output_layer amax cannot be restore via
    # sharded_state_dict. All output_layer quantizers state are removed.
    config["quant_cfg"]["*output_layer*"] = {"enable": False}

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt_version
        mtq.plugins.megatron.__version__ = modelopt_version

    tp_size = moe_config.get("tp_size", size)
    ep_size = moe_config.get("ep_size", 1)
    etp_size = moe_config.get("etp_size", None)
    num_moe_experts = moe_config.get("num_moe_experts", None)
    moe_grouped_gemm = moe_config.get("moe_grouped_gemm", False)
    use_te = moe_config.get("use_te", False)

    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        seed=SEED,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )

    model_ref = _gpt_model_provider(
        tp_size,
        hidden_size,
        vocab_size=256,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te=use_te,
        ep_size=ep_size,
        etp_size=etp_size,
    )
    model_test = _gpt_model_provider(
        tp_size,
        hidden_size,
        vocab_size=256,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te=use_te,
        meta_device=meta_device,
        ep_size=ep_size,
        etp_size=etp_size,
    )

    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, config, forward_fn)
    if compress:
        mtq.compress(model_ref)

    for module in model_ref.modules():
        if hasattr(module, "_amax_for_smoothing"):
            delattr(module, "_amax_for_smoothing")

    sharded_state_dict_test_helper(
        tmp_path,
        model_ref,
        model_test,
        forward_fn,
        meta_device=meta_device,
        version=modelopt_version,
    )

    if modelopt_version is not None:
        mto.conversion.__version__ = modelopt.__version__
        mtq.plugins.megatron.__version__ = modelopt.__version__

    # Make sure all ranks have arrived before destroying NCCL
    torch.distributed.barrier()


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
        "*.2.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 64}, "enable": True},
        "*.2.*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.3.*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128, -2: 64}, "enable": True},
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
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    ],
)
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("meta_device", [False, True])
def test_homogeneous_sharded_state_dict(tmp_path, config, compress, meta_device):
    if compress and config is mtq.W4A8_AWQ_BETA_CFG:
        pytest.skip("W4A8_AWQ_BETA_CFG is not supported for compress")

    size = torch.cuda.device_count()

    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_sharded_state_dict, tmp_path, config, 256, None, compress, meta_device, {}
        ),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "config",
    [
        mixed_precision_config,
        mixed_block_size_config,
    ],
)
def test_heterogenous_sharded_state_dict(need_2_gpus, tmp_path, config):
    spawn_multiprocess_job(
        size=2,
        job=partial(_test_sharded_state_dict, tmp_path, config, 256, None, False, False, {}),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "config",
    [
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mixed_precision_config,
    ],
)
@pytest.mark.parametrize("modelopt_version", ["0.25", "0.27"])
@pytest.mark.skip(
    reason="0.31 has breaking change without backward compatibility. This unittest needs to be refactorized."
)
def test_sharded_state_dict_old_checkpoints(need_2_gpus, tmp_path, config, modelopt_version):
    spawn_multiprocess_job(
        size=2,
        job=partial(
            _test_sharded_state_dict, tmp_path, config, 256, modelopt_version, False, False, {}
        ),
        backend="nccl",
    )


@pytest.mark.parametrize("hidden_size", [256, 320])
def test_regular_state_dict(distributed_setup_size_1, hidden_size):
    initialize_for_megatron(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=SEED)

    model_ref = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    model_test = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(
        0, model_ref.vocab_size, (2, model_ref.max_sequence_length)
    ).cuda()

    def forward_fn(model):
        return run_mcore_inference(model, prompt_tokens)

    model_ref = mtq.quantize(model_ref, mixed_precision_config, forward_fn)

    mto.restore_from_modelopt_state(model_test, mto.modelopt_state(model_ref))
    model_test.load_state_dict(model_ref.state_dict())

    model_test_sd = model_test.state_dict()
    for k, v in model_ref.state_dict().items():
        # The extra_state checkint must be skipped. It can be a byte tensor serialized
        # from a dict where the order can change.
        if "_extra_state" in k:
            continue
        assert not isinstance(v, torch.Tensor) or torch.allclose(v, model_test_sd[k]), k

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


def _test_fp8_real_quantize_helper(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size, pipeline_model_parallel_size=1, seed=SEED
    )
    hidden_size = 256
    config = mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG

    model = _gpt_model_provider(tp_size=1, hidden_size=hidden_size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    forward_fn(model)

    # real quant the model
    cur_mem = get_model_size(model)
    real_quant_model = mtq.quantize(model, config, forward_fn)
    mtq.compress(real_quant_model)
    real_quant_mem = get_model_size(real_quant_model)

    # Since not all parameters are quantized, the size won't be lower than half.
    assert real_quant_mem < (cur_mem / 2) * 1.1, "Memory after real quantization is not reduced."

    # check forward works after real quantization
    forward_fn(real_quant_model)

    assert real_quant_mem < cur_mem


def test_fp8_real_quantize():
    size = torch.cuda.device_count()
    spawn_multiprocess_job(size=size, job=_test_fp8_real_quantize_helper, backend="nccl")


@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
        mtq.NVFP4_DEFAULT_CFG,
    ],
)
@pytest.mark.parametrize("moe_grouped_gemm", [False, True])
def test_moe_sharded_state_dict(tmp_path, config, moe_grouped_gemm):
    size = torch.cuda.device_count()
    # TODO: Add support for compress=True for TEGroupedMLP
    if size < 4:
        pytest.skip("Requires at least 4 GPUs for expert parallel test")
    moe_config = {
        "tp_size": 2,
        "ep_size": 2,
        "etp_size": 2,
        "num_moe_experts": 4,
        "moe_grouped_gemm": moe_grouped_gemm,
        "use_te": moe_grouped_gemm,
    }
    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_sharded_state_dict,
            tmp_path,
            config,
            256,
            None,
            False,
            False,
            moe_config,
        ),
        backend="nccl",
    )


def _test_te_grouped_vs_sequential_quantize_helper(tp_size, ep_size, etp_size, rank, size):
    """Test that TEGrouped and sequential MoE models produce similar amax values."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        seed=SEED,
    )

    # Create input
    prompt_tokens = torch.randint(0, 64, (2, 16)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # Create TEGrouped MoE model
    te_grouped_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        hidden_size=32,
        moe_grouped_gemm=True,
        use_te=True,
        num_moe_experts=4,
    )
    num_te_grouped_mlp = sum(
        isinstance(module, TEGroupedMLP) for module in te_grouped_moe_model.modules()
    )
    assert num_te_grouped_mlp == 4, (
        f"TEGrupedMoEModel has {num_te_grouped_mlp} TEGroupedMLP modules, it should have 4"
    )

    # Create sequential MoE model
    sequential_moe_model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        hidden_size=32,
        moe_grouped_gemm=False,
        num_moe_experts=4,
    )
    num_sequential_mlp = sum(
        isinstance(module, SequentialMLP) for module in sequential_moe_model.modules()
    )
    assert num_sequential_mlp == 4, (
        f"SequentialMoEModel has {num_sequential_mlp} SequentialMLP modules, it should have 4"
    )
    # Copy weights from grouped to non-grouped model
    copy_weights_from_grouped_to_non_grouped(te_grouped_moe_model, sequential_moe_model)

    # Compare model outputs before quantization
    te_grouped_moe_output = forward_fn(te_grouped_moe_model)
    sequential_moe_output = forward_fn(sequential_moe_model)
    assert torch.allclose(te_grouped_moe_output, sequential_moe_output, atol=1e-6, rtol=1e-6)

    # Quantize grouped model
    mtq.quantize(te_grouped_moe_model, mtq.FP8_DEFAULT_CFG, forward_fn)

    # Quantize non-grouped model
    mtq.quantize(sequential_moe_model, mtq.FP8_DEFAULT_CFG, forward_fn)

    # Compare model outputs after quantization
    te_grouped_moe_quant_output = forward_fn(te_grouped_moe_model)
    sequential_moe_quant_output = forward_fn(sequential_moe_model)
    assert torch.allclose(
        te_grouped_moe_quant_output, sequential_moe_quant_output, atol=1e-6, rtol=1e-6
    )


def test_te_grouped_vs_sequential_quantize():
    """Test that TEGrouped and sequential MoE models produce similar quantized models."""

    size = torch.cuda.device_count()
    if size < 4:
        pytest.skip("Requires at least 4 GPUs for expert parallel test")

    spawn_multiprocess_job(
        size=size,
        job=partial(_test_te_grouped_vs_sequential_quantize_helper, 1, 2, 2),
        backend="nccl",
    )


def _test_expert_model_parallel_amax_sync(
    tp_size, ep_size, etp_size, moe_grouped_gemm, config, rank, size
):
    """Test expert parallel synchronization with different configurations."""
    initialize_for_megatron(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        seed=SEED,
    )

    # Create model with expert parallelism
    model = _gpt_model_provider(
        tp_size=tp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        hidden_size=256,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te=moe_grouped_gemm,
        num_moe_experts=4,
    )
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_fn(model):
        return megatron_prefill(model, prompt_tokens)

    # quantize the model
    model = mtq.quantize(model, config, forward_fn)

    # Check initial sync status
    initial_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert initial_sync, (
        f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"
    )
    # Create inconsistent amax values
    cur_rank = torch.distributed.get_rank()
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer):
            # Check if this is an expert quantizer
            is_expert_quantizer = (
                "local_experts" in name  # sequential MoE
                or ("experts" in name and "linear_fc" in name)  # TEGrouped MoE
            )

            if is_expert_quantizer and hasattr(module, "_amax"):
                # Create rank-specific amax values to simulate missing sync
                rank_offset = cur_rank * 0.1
                module.amax = module.amax + rank_offset

    # Test if the amax values are inconsistent
    inconsistent_amax, _, _ = compare_amax_sync_across_expert_parallel(model)
    assert not inconsistent_amax, (
        "Consistent amax across expert parallel ranks, "
        "Amax should not be synchronized across expert parallel ranks since expert parallel is disabled"
    )
    # Re-enable parallel groups and test synchronization
    mtq.model_calib.max_calibrate(model, forward_fn)

    final_sync, quantizer_type, rank_values = compare_amax_sync_across_expert_parallel(model)
    assert final_sync, f"Inconsistent amax for expert {quantizer_type} across ranks: {rank_values}"


@pytest.mark.parametrize(("ep_size", "etp_size"), [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("moe_grouped_gemm", [True, False])
def test_expert_parallel_sync(ep_size, etp_size, moe_grouped_gemm):
    """Test expert model parallel synchronization."""
    size = torch.cuda.device_count()
    if size < ep_size * etp_size:
        pytest.skip(f"Requires at least {ep_size * etp_size} GPUs for expert model parallel test")

    spawn_multiprocess_job(
        size=size,
        job=partial(
            _test_expert_model_parallel_amax_sync,
            2,
            ep_size,
            etp_size,
            moe_grouped_gemm,
            mtq.FP8_DEFAULT_CFG,
        ),
        backend="nccl",
    )
