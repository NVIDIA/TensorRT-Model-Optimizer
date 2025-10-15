"""PSX formats quantization backend."""

import torch
from psx_formats import ScaleArguments, convert
from psx_formats.utils import string_to_format

from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
    is_registered_quant_backend,
    register_quant_backend,
)
from modelopt.torch.quantization.tensor_quant import (
    _fake_quant_backward_function,
    _save_for_backward_if_needed,
)

__all__ = []

# An example config for PSX formats quantization with ModelOpt.
PSX_WEIGHT_E2M0_ACTIVATION_E4M3_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": "E2M0",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "E4M3"},
            "enable": True,
            "backend": "psx_formats",
            "pass_through_bwd": True,  # Vanilla STE; no outlier gradient clipping
            "backend_extra_args": {"global_scale": "static"},
        },
        "*input_quantizer": {
            # TODO: Enabled point-wise quantization for PSX formats
            # and use psx-formats backend here
            "num_bits": (4, 3),
            "pass_through_bwd": True,  # Vanilla STE; no outlier gradient clipping
            "enable": True,
        },
        # Disable sensitive layers such as classification head, MoE router etc.
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

PSX_WEIGHT_E2M0_ACTIVATION_NONE_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": "E2M0",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "E4M3"},
            "enable": True,
            "backend": "psx_formats",
            "pass_through_bwd": True,  # Vanilla STE; no outlier gradient clipping
            "backend_extra_args": {"global_scale": "static"},
        },
        "*input_quantizer": {
            "enable": False,  # Disable input quantization
        },
        # Disable sensitive layers such as classification head, MoE router etc.
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

PSX_WEIGHT_E2M1_ACTIVATION_E2M1_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": "E2M1",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "E4M3"},
            "enable": True,
            "backend": "psx_formats",
            "pass_through_bwd": True,  # Vanilla STE; no outlier gradient clipping
            "backend_extra_args": {"global_scale": "static"},
        },
        "*input_quantizer": {
            "num_bits": "E2M1",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "E4M3"},
            "enable": True,
            "backend": "psx_formats",
            "pass_through_bwd": True,  # Vanilla STE; no outlier gradient clipping
            "backend_extra_args": {"global_scale": "static"},
        },
        # Disable sensitive layers such as classification head, MoE router etc.
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

choices: set[str] = {
    "PSX_WEIGHT_E2M0_ACTIVATION_E4M3_CFG",
    "PSX_WEIGHT_E2M0_ACTIVATION_NONE_CFG",
    "PSX_WEIGHT_E2M1_ACTIVATION_E2M1_CFG",
}


class PsxFormatsBlockQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, amax: torch.Tensor | None, tq: TensorQuantizer):
        _save_for_backward_if_needed(ctx, tq._pass_through_bwd, inputs, amax)

        # IMPORTANT: expose amax in the argument to add support for learning the amax
        scale_args = ScaleArguments(
            block_size=tq.block_sizes.get(-1),
            block_scale=tq.block_sizes.get("type"),
            global_scale=tq.backend_extra_args.get("global_scale", "static"),
            static_global_amax=amax,
        )

        # element format and scale format
        elem_config = string_to_format(tq.num_bits)
        scale_config = string_to_format(tq.block_sizes.get("scale_bits", "E4M3"))

        # Fake-quantized output
        y = convert(inputs, scale_args, elem_config, scale_config)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _fake_quant_backward_function(ctx, grad_output, num_args=3)


class PsxFormatsPointWiseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, amax: torch.Tensor | None, tq: TensorQuantizer):
        raise NotImplementedError("TODO: Implement point-wise quantization for PSX formats.")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def psx_formats_entrypoint(inputs, tq: TensorQuantizer):
    # Static global amax for static global amax path
    if tq.backend_extra_args.get("global_scale", "static") == "static":
        amax = tq._get_amax(inputs).to(torch.float32)
    else:
        amax = None

    if tq.block_sizes is not None:
        return PsxFormatsBlockQuant.apply(inputs, amax, tq)
    else:
        return PsxFormatsPointWiseQuant.apply(inputs, amax, tq)


if not is_registered_quant_backend("psx_formats"):
    register_quant_backend("psx_formats", psx_formats_entrypoint)
