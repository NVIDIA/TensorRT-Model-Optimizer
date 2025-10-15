import pytest
import torch

import modelopt.torch.quantization as mtq

pytest.importorskip("psx_formats")
from modelopt.torch.quantization.plugins.psx_formats import (
    PSX_WEIGHT_E2M0_ACTIVATION_E4M3_CFG,
    psx_formats_entrypoint,
)

config = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": "E2M0",  # Rubin E2M0
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "E4M3"},
            "enable": True,
            "backend": "psx_formats",
            "backend_extra_args": {"global_scale": "dynamic"},
        },
        "default": {
            "enable": False,
        },
    },
    "algorithm": "max",
}


@pytest.mark.parametrize("config", [PSX_WEIGHT_E2M0_ACTIVATION_E4M3_CFG, config])
def test_psx_formats(config):
    model = torch.nn.Linear(16, 16, bias=False)
    model = model.cuda()

    inputs = torch.randn(1, 16).cuda()

    def forward_loop(m):
        m(inputs)

    mtq.quantize(model, config, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    output_test = model(inputs)

    weights_q = psx_formats_entrypoint(model.weight, model.weight_quantizer)
    # TODO: call psx_formats_entrypoint for input quantization after
    # adding support for point-wise quantization
    inputs_q = model.input_quantizer(inputs)
    # Reference output
    output_ref = inputs_q @ weights_q.T

    assert torch.allclose(output_test, output_ref, atol=1e-5)

    # Test backward
    output_test.sum().backward()

    assert model.weight.grad is not None
