import torch
import torch.distributed as dist
from _test_utils.torch_dist.dist_utils import spawn_multiprocess_job
from _test_utils.torch_dist.plugins.megatron_common import MegatronModel, initialize_for_megatron
from megatron.core.parallel_state import get_data_parallel_group

from modelopt.torch.quantization.model_calib import awq_lite


def _test_awq_lite_act_scale_sync_helper(rank, size):
    initialize_for_megatron(seed=1234 + rank)
    model = MegatronModel().cuda()

    calib_data = model.get_dummy_input().cuda()

    def forward_loop(model):
        model(calib_data)

    model = awq_lite(model, forward_loop)
    # Sanity check
    forward_loop(model)

    act_scale = model.fc1.weight_quantizer.awq_lite.act_scale.clone()
    dist.all_reduce(act_scale, op=dist.ReduceOp.AVG, group=get_data_parallel_group())
    assert torch.allclose(act_scale, model.fc1.weight_quantizer.awq_lite.act_scale)

    act_scale = model.fc2.weight_quantizer.awq_lite.act_scale.clone()
    dist.all_reduce(act_scale, op=dist.ReduceOp.AVG, group=get_data_parallel_group())
    assert torch.allclose(act_scale, model.fc2.weight_quantizer.awq_lite.act_scale)


def test_awq_lite_act_scale_sync(need_2_gpus):
    spawn_multiprocess_job(size=2, job=_test_awq_lite_act_scale_sync_helper, backend="nccl")
