import os
import tempfile
from typing import Optional

import pytest
import torch

import modelopt.torch.quantization as mtq


pytest.importorskip("luts")
import luts
from modelopt.torch.quantization.plugins.luts import luts_entrypoint
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.config import QuantizerAttributeConfig


def test_scalar_lut() -> None:

    scalar_lut_config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "enable": True,
                "backend": "luts",
                "num_bits": "binary",
                "backend_extra_args": {
                    "block_sizes": 16,
                    "lut_type": "scalar_lut",
                    "encode_path": None,
                    "tile_size": 16,
                    "scale_type": "e4m3",
                    "scale_mode": "max",
                    "magnitude": True,
                    "round_mode": "rne",
                },
            },
            "default": {
                "enable": False,
            },
        },
        "algorithm": "max",
    }

    model = torch.nn.Linear(16, 16, bias=False)
    model = model.cuda()

    inputs = torch.randn(1, 16).cuda()

    def forward_loop(m):
        m(inputs)

    mtq.quantize(model, scalar_lut_config, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    output_test = model(inputs)

    weights_q = luts_entrypoint(model.weight, model.weight_quantizer)
    output_ref = inputs @ weights_q.T

    assert torch.allclose(output_test, output_ref, atol=1e-5)

    # Test backward
    output_test.sum().backward()

    assert model.weight.grad is not None


def test_vector_lut() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        encode_format = "my_favourite_vector_lut"

        values = torch.FloatTensor([[0, 1, 2, 3], [0.25, 0.5, 4, 10]])
        dic = {}
        dic["values"] = values
        torch.save(dic, os.path.join(tmp_dir, f"{encode_format}.pt"))
        encode_path = (
            tmp_dir + "/"
        )  # TODO(Frank): Fix luts encode_lut to use os.path.join

        vector_lut_config = {
            "quant_cfg": {
                "*weight_quantizer": {
                    "enable": True,
                    "backend": "luts",
                    "num_bits": encode_format,
                    "backend_extra_args": {
                        "block_sizes": 16,
                        "lut_type": "vector_lut",
                        "encode_path": encode_path,
                        "scale_type": "e4m3",
                        "scale_mode": "max",
                    },
                },
                "default": {
                    "enable": False,
                },
            },
            "algorithm": "max",
        }

        model = torch.nn.Linear(16, 16, bias=False)
        model = model.cuda()

        inputs = torch.randn(1, 16).cuda()

        def forward_loop(m):
            m(inputs)

        mtq.quantize(model, vector_lut_config, forward_loop=forward_loop)
        mtq.print_quant_summary(model)
        output_test = model(inputs)

        weights_q = luts_entrypoint(model.weight, model.weight_quantizer)
        output_ref = inputs @ weights_q.T

        assert torch.allclose(output_test, output_ref, atol=1e-5)

        # Test backward
        output_test.sum().backward()

        assert model.weight.grad is not None


@pytest.mark.parametrize(
    "encode_format",
    [
        "binary",
        "ternary",
        "quinary",
        "e1m0",
        "int2n",
        "int2p",
        "int2n_int2p",
        "e2m0",
        "e0m2",
        "e2m0_e0m2",
        "e3m0",
        "e2m1",
        "e0m3",
        "e3m0_e2m1_e0m3",
    ],
)
@pytest.mark.parametrize("scale_type", ["fp32", "e4m3", "e8m0"])
@pytest.mark.parametrize("scale_mode", ["max", "mean"])
@pytest.mark.parametrize("dmax", [True, False])
def test_luts_scalar_lut_integration(
    encode_format: str,
    scale_type: str,
    scale_mode: str,
    dmax: bool,
):

    input_tensor = torch.randn(1024, 1024).bfloat16().cuda()

    x = input_tensor.clone()
    values, bounds = luts.encode(encode_format, norm=False, cuda=True)
    luts_output = luts.scalar_lut(
        x,
        block_size=16,
        tile_size=16,
        scale_type=scale_type,
        scale_mode=scale_mode,
        global_amax=torch.max(torch.abs(x)).float(),
        values=values,
        bounds=bounds,
        dmax=dmax,
        magnitude=True,
        round_mode="rne",
    )

    modelopt_config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "enable": True,
                "backend": "luts",
                "num_bits": encode_format,
                "backend_extra_args": {
                    "block_sizes": 16,
                    "tile_size": 16,
                    "lut_type": "scalar_lut",
                    "scale_type": scale_type,
                    "scale_mode": scale_mode,
                    "magnitude": True,
                    "round_mode": "rne",
                    "dmax": dmax,
                },
            },
            "default": {
                "enable": False,
            },
        },
        "algorithm": "max",
    }

    # Extract the quantizer config from modelopt_config
    quantizer_config_dict = modelopt_config["quant_cfg"]["*weight_quantizer"]

    # Create a QuantizerAttributeConfig object with defaults filled in
    quantizer_config = QuantizerAttributeConfig(**quantizer_config_dict)

    # Create a TensorQuantizer with the extracted config
    quantizer = TensorQuantizer(quant_attribute_cfg=quantizer_config)

    # Need to set calibration mode and get amax before quantization
    quantizer.enable_calib()
    quantizer(input_tensor)  # Calibrate
    quantizer.load_calib_amax()
    quantizer.disable_calib()
    quantizer.enable()

    modelopt_output = luts_entrypoint(input_tensor, quantizer)

    # Compare the outputs
    torch.testing.assert_close(luts_output, modelopt_output, atol=0, rtol=0)


def _test_luts_vector_lut_integration(
    encode_format: str,
    encode_path: Optional[str],
    scale_type: str,
    scale_mode: str,
):

    input_tensor = torch.randn(1024, 1024).bfloat16().cuda()

    x = input_tensor.clone()
    values, _ = luts.encode(
        dtype=encode_format, path=encode_path, norm=False, cuda=True
    )
    luts_output = luts.vector_lut(
        x,
        block_size=16,
        scale_type=scale_type,
        scale_mode=scale_mode,
        global_amax=torch.max(torch.abs(x)).float(),
        values=values,
    )

    modelopt_config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "enable": True,
                "backend": "luts",
                "num_bits": encode_format,
                "backend_extra_args": {
                    "block_sizes": 16,
                    "lut_type": "vector_lut",
                    "encode_path": encode_path,
                    "scale_type": scale_type,
                    "scale_mode": scale_mode,
                },
            },
            "default": {
                "enable": False,
            },
        },
        "algorithm": "max",
    }

    # Extract the quantizer config from modelopt_config
    quantizer_config_dict = modelopt_config["quant_cfg"]["*weight_quantizer"]

    # Create a QuantizerAttributeConfig object with defaults filled in
    quantizer_config = QuantizerAttributeConfig(**quantizer_config_dict)

    # Create a TensorQuantizer with the extracted config
    quantizer = TensorQuantizer(quant_attribute_cfg=quantizer_config)

    # Need to set calibration mode and get amax before quantization
    quantizer.enable_calib()
    quantizer(input_tensor)  # Calibrate
    quantizer.load_calib_amax()
    quantizer.disable_calib()
    quantizer.enable()

    modelopt_output = luts_entrypoint(input_tensor, quantizer)

    # Compare the outputs
    torch.testing.assert_close(luts_output, modelopt_output, atol=0, rtol=0)


@pytest.mark.parametrize(
    "encode_format",
    [
        "0.5b_vector_lut",
    ],
)
@pytest.mark.parametrize("scale_type", ["fp32", "e4m3", "e8m0"])
@pytest.mark.parametrize("scale_mode", ["max", "mean"])
def test_luts_vector_lut_integration(
    encode_format: str,
    scale_type: str,
    scale_mode: str,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        values = torch.FloatTensor([[0, 1, 2, 3], [0.25, 0.5, 4, 10]])
        dic = {}
        dic["values"] = values
        torch.save(dic, os.path.join(tmp_dir, f"{encode_format}.pt"))
        encode_path = (
            tmp_dir + "/"
        )  # TODO(Frank): Fix luts encode_lut to use os.path.join
        _test_luts_vector_lut_integration(
            encode_format, encode_path, scale_type, scale_mode
        )
