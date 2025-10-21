"""PSX formats quantization backend."""

from typing import Set
import os

import torch
import luts

from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
    register_quant_backend,
    is_registered_quant_backend,
)

__all__ = []


LUTS_SCALAR_LUT_E2M0_E4M3_16_16_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": True,
            "backend": "luts",
            "num_bits": "e2m0",
            "backend_extra_args": {
                "block_sizes": 16,
                "lut_type": "scalar_lut",
                "encode_path": None,
                "tile_size": 16,
                "scale_type": "e4m3",
                "scale_mode": "max",
                "dmax": True,
                "magnitude": True,
                "round_mode": "rne",
            },
        },
        "default": {
            "enable": False,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}


choices: Set[str] = {
    "LUTS_SCALAR_LUT_E2M0_E4M3_16_16_CFG",
}

_SCALAR_LUT_ENCODE_FORMATS = [
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
]


class LutsScalarLut(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, amax: torch.Tensor | None, tq: TensorQuantizer
    ):

        if tq.backend_extra_args.get("lut_type", "scalar_lut") != "scalar_lut":
            raise ValueError(
                f"Expecting scalar_lut. Got: {tq.backend_extra_args.get('lut_type', 'scalar_lut')}."
            )

        encode_format = tq.num_bits
        encode_path = tq.backend_extra_args.get("encode_path", None)

        if encode_format not in _SCALAR_LUT_ENCODE_FORMATS:
            if encode_path is None:
                raise ValueError(
                    f"Invalid encode format: {encode_format}. If path is not provided, a fixed encoding must be used."
                )
            else:
                if not os.path.isabs(encode_path):
                    raise ValueError(
                        f"Invalid encode path: {encode_path}. It must be an absolute path."
                    )
                if not os.path.exists(encode_path):
                    raise FileNotFoundError(
                        f"luts encoder path {encode_path} does not exist."
                    )

        # TODO(Frank): Cache the encoding samples since we don't want to load torch
        # tensor from disk every time we are doing a forward.
        if encode_path is not None:
            encode_path += "/"  # TODO(Frank): Fix luts encode_lut to use os.path.join
        values, bounds = luts.encode(
            encode_format, path=encode_path, norm=False, cuda=True
        )

        block_size = tq.backend_extra_args.get("block_sizes", 16)
        if not (block_size >= 4 and block_size <= 128):
            raise ValueError(f"Invalid block size: {block_size}.")

        tile_size = tq.backend_extra_args.get("tile_size", 16)
        if not (tile_size >= 4 and tile_size <= 128):
            raise ValueError(f"Invalid tile size: {tile_size}.")

        scale_type = tq.backend_extra_args.get("scale_type", "e4m3")
        if scale_type not in ["fp32", "e4m3", "e8m0", "none"]:
            raise ValueError(f"Invalid scale type: {scale_type}.")

        scale_mode = tq.backend_extra_args.get("scale_mode", "max")
        if scale_mode not in ["max", "mean"]:
            raise ValueError(f"Invalid scale mode: {scale_mode}.")

        round_mode = tq.backend_extra_args.get("round_mode", "rne")
        if round_mode not in ["rne", "rna"]:
            raise ValueError(f"Invalid round mode: {round_mode}.")

        magnitude = tq.backend_extra_args.get("magnitude", True)

        dmax = tq.backend_extra_args.get("dmax", True)

        return luts.scalar_lut(
            inputs,
            block_size=block_size,
            tile_size=tile_size,
            scale_type=scale_type,
            scale_mode=scale_mode,
            global_amax=amax.float(),
            values=values,
            bounds=bounds,
            dmax=dmax,
            magnitude=magnitude,
            round_mode=round_mode,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


class LutsVectorLut(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, amax: torch.Tensor | None, tq: TensorQuantizer
    ):

        if tq.backend_extra_args.get("lut_type", "vector_lut") != "vector_lut":
            raise ValueError(
                f"Expecting vector_lut. Got: {tq.backend_extra_args.get('lut_type', 'vector_lut')}."
            )

        encode_format = tq.num_bits
        encode_path = tq.backend_extra_args.get("encode_path", None)

        if encode_path is None:
            raise ValueError(
                f"Encode path for vector LUT must be provided."
            )
        else:
            if not os.path.isabs(encode_path):
                raise ValueError(
                    f"Invalid encode path: {encode_path}. It must be an absolute path."
                )
            if not os.path.exists(encode_path):
                raise FileNotFoundError(
                    f"luts encoder path {encode_path} does not exist."
                )

        # TODO(Frank): Cache the encoding samples since we don't want to load torch
        # tensor from disk every time we are doing a forward.
        if encode_path is not None:
            encode_path += "/"  # TODO(Frank): Fix luts encode_lut to use os.path.join
        values, _ = luts.encode(encode_format, path=encode_path, norm=False, cuda=True)

        block_size = tq.backend_extra_args.get("block_sizes", 16)
        if not (block_size >= 4 and block_size <= 128):
            raise ValueError(f"Invalid block size: {block_size}.")

        scale_type = tq.backend_extra_args.get("scale_type", "e4m3")
        if scale_type not in ["fp32", "e4m3", "e8m0", "none"]:
            raise ValueError(f"Invalid scale type: {scale_type}.")

        scale_mode = tq.backend_extra_args.get("scale_mode", "max")
        if scale_mode not in ["max", "mean"]:
            raise ValueError(f"Invalid scale mode: {scale_mode}.")

        return luts.vector_lut(
            inputs,
            block_size=block_size,
            scale_type=scale_type,
            scale_mode=scale_mode,
            values=values,
            global_amax=amax.float(),
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def luts_entrypoint(inputs, tq: TensorQuantizer):
    # Static global amax for static global amax path
    amax = tq._get_amax(inputs)
    lut_type = tq.backend_extra_args.get("lut_type", "scalar_lut")

    if lut_type == "scalar_lut":
        return LutsScalarLut.apply(inputs, amax, tq)
    elif lut_type == "vector_lut":
        return LutsVectorLut.apply(inputs, amax, tq)
    else:
        raise NotImplementedError(
            f"Other lut types other than scalar_lut and vector_lut are not supported yet. Got: {lut_type}."
        )


if not is_registered_quant_backend("luts"):
    register_quant_backend("luts", luts_entrypoint)
