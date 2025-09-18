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

from typing import Literal

from pydantic import ValidationInfo, field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.utils.network import ConstructorLike

BiasType = Literal["static", "dynamic"]
BiasMethod = Literal["mean", "max_min"]


class QuantizerAttributeConfig(ModeloptBaseConfig):
    """Quantizer attribute type."""

    enable: bool = ModeloptField(
        default=True,
        title="Enable quantizer.",
        description="""If True, enables the quantizer. If False, by-pass the quantizer and returns the input tensor.""",
    )

    num_bits: int | tuple[int, int] = ModeloptField(
        default=8,
        title="An integer or a tuple of two integers specifying the number of quantization bits.",
        description="""`num_bits` can be:

        #. A positive integer argument for integer quantization. `num_bits` specify
            the number of bits used for integer quantization.

        #. Constant integer tuple (E,M) for floating point quantization emulating
            Nvidia's FPx quantization. E is the number of exponent bits and M is the number
            of mantissa bits. Supported FPx quantization formats: FP8 (E4M3, E5M2), FP6(E3M2, E2M3), FP4(E2M1).""",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        """Validate quantizer config."""

        def _validate_recursive(value):
            """Recursively validate config structure."""
            if value is None:
                return

            if isinstance(value, list):
                for item in value:
                    _validate_recursive(item)
            elif isinstance(value, dict):
                if len(value) == 1 and "enable" in value and value["enable"] is True:
                    raise ValueError(
                        "Invalid quantizer config: Cannot specify only {'enable': True}. "
                        "Additional parameters are required when enabling quantization."
                    )
                # Recurse into nested dicts
                for v in value.values():
                    _validate_recursive(v)

        _validate_recursive(values)
        return values

    @model_validator(mode="after")
    def validate_num_bits(self):
        """Validate `num_bits`."""
        num_bits = self.num_bits

        if isinstance(num_bits, int) and num_bits < 1:
            raise ValueError("num_bits must be a positive integer or a tuple of positive integers.")

        if not isinstance(num_bits, tuple):
            return self

        if not all(x > 0 for x in num_bits):
            raise ValueError("num_bits must be a positive integer or a tuple of positive integers.")

        block_sizes = self.block_sizes
        if num_bits not in [
            (4, 3),
            (5, 2),
            (2, 1),
            (1, 2),
            (0, 3),
            (3, 0),
            (3, 2),
            (2, 3),
        ]:
            raise ValueError(
                "Supported FPx quantization formats: FP8 (E4M3, E5M2), FP6(E3M2, E2M3), FP4(E2M1)."
            )
        elif num_bits != (4, 3) and (
            block_sizes is None or block_sizes.get("type", None) != "dynamic"
        ):
            raise ValueError(
                "Only blockwise dynamic quantization is supported with quantization "
                "formats E{num_bis[0]}M{num_bits[1]}."
            )
        return self

    axis: int | tuple[int, ...] | None = ModeloptField(
        default=None,
        title="None, integer or a tuple of integers specifying the axis to quantize.",
        description="""This field is for static per-channel quantization. *It cannot coexist with `block_sizes`*.
            You should set axis if you want a fixed shape of scale factor.

            For example, if axis is set to None, the scale factor will be a scalar (per-tensor quantization)
            if the axis is set to 0, the scale factor will be a vector of shape (dim0, ) (per-channel quantization).
            if the axis is set to (-2, -1), the scale factor will be a vector of shape (dim-2, dim-1)

            axis value must be in the range [-rank(input_tensor), rank(input_tensor))
        """,
    )

    fake_quant: bool = ModeloptField(
        default=True,
        title="Enable fake quantization.",
        description="""If True, enable fake quantization.""",
    )

    unsigned: bool = ModeloptField(
        default=False,
        title="Enable unsigned quantization.",
        description="""If True, enable unsigned quantization. Used only for integer quantization.""",
    )

    narrow_range: bool = ModeloptField(
        default=False,
        title="Enable narrow range quantization.",
        description="""If True, enable narrow range quantization. Used only for integer quantization.""",
    )

    learn_amax: bool = ModeloptField(
        default=False,
        title="Enable learning amax.",
        description="""``learn_amax`` is deprecated and reserved for backward compatibility.""",
    )

    @field_validator("learn_amax")
    @classmethod
    def validate_learn_amax(cls, v):
        """Validate learn_amax."""
        assert v is not True, "learn_amax is deprecated and reserved for backward compatibility."
        return v

    type: str = ModeloptField(
        default="static",
        title="""Specify whether the quantization is static or dynamic.""",
        description="""The value is a string from ``["static", "dynamic"]``.
            If ``"dynamic"``, dynamic quantization will be enabled which does not collect any statistics during
            calibration.""",
        pattern=r"^static$|^dynamic$",
    )

    block_sizes: dict[int | str, int | tuple[int, int] | str | dict[int, int] | None] | None = (
        ModeloptField(
            default=None,
            title="Optional dictionary specifying block quantization parameters.",
            description="""This field is for static or dynamic block quantization. *It cannot coexist with ``axis``*.
            You should set block_sizes if you want fixed number of elements to share every scale factor.

            The keys are the axes for block quantization and the
            values are block sizes for quantization along the respective axes. Keys must be in the
            range ``[-tensor.dim(), tensor.dim())``. Values, which are the block sizes for quantization must be
            positive integers or ``None``. A positive block size specifies the block size for quantization along that
            axis. ``None`` means that the block size will be the maximum possible size in that dimension - this is
            useful for specifying certain quantization formats such per-token dynamic quantization which has the `amax`
            shared along the last dimension.

            In addition, there can be special string keys ``"type"``, ``"scale_bits"`` and ``"scale_block_sizes"``.

            Key ``"type"`` should map to ``"dynamic"`` or ``"static"`` where ``"dynamic"``
            indicates dynamic block quantization and "static"
            indicates static calibrated block quantization. By default, the type is ``"static"``.

            Key ``"scale_bits"`` specify the quantization bits for the per-block quantization scale factor
            (i.e a double quantization scheme).

            Key ``"scale_block_sizes"`` specify the block size for double quantization.
            By default per-block quantization scale is not quantized.

            For example, ``block_sizes = {-1: 32}`` will quantize the last axis of the input tensor in
            blocks of size 32 with static calibration, with a total of ``numel(tensor) / 32`` scale factors.
            ``block_sizes = {-1: 32, "type": "dynamic"}`` will perform dynamic block quantization.
            ``block_sizes = {-1: None, "type": "dynamic"}`` can be used to
            specify per-token dynamic quantization.
        """,
        )
    )

    bias: dict[int | str, BiasType | BiasMethod | tuple[int, ...] | bool | int | None] | None = (
        ModeloptField(
            default=None,
            title="Bias configuration.",
            description="""Configuration for bias handling in affine quantization. The keys are:
            - "enable": Boolean to enable/disable bias handling, default is False
            - "type": Specify the type of bias ["static", "dynamic"], default is "static"
            - "method": Specify the method of bias calibration ["mean", "max_min"], default is "mean"
            - "axis": Tuple of integers specifying axes for bias computation, default is None

            Examples:
            bias = {"enable": True}
            bias = {"enable": True, "type": "static", "axis": -1}
            bias = {"enable": True, "type": "dynamic", "axis": (-1, -3)}
        """,
        )
    )

    @staticmethod
    def _get_block_quant_axes_and_sizes(block_sizes):
        if block_sizes is None:
            return None
        return {
            k: v
            for k, v in block_sizes.items()
            if k not in ["type", "scale_bits", "scale_block_sizes"]
        }

    @field_validator("block_sizes")
    @classmethod
    def validate_block_sizes(cls, v, info: ValidationInfo):
        """Validate block sizes."""
        if v is None:
            return v
        assert info.data["axis"] is None, "axis must be None when block_sizes is not None."
        if v.get("type", None) == "dynamic":
            assert len(cls._get_block_quant_axes_and_sizes(v)) == 1, (
                "Dynamic block quantization only supports quantization last axis."
            )
        for _k, _v in v.items():
            if isinstance(_k, str):
                assert _k in ["type", "scale_bits", "scale_block_sizes"]
            else:
                assert isinstance(_k, int) and (_v is None or isinstance(_v, int))
        return v

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v):
        """Validate bias."""
        if v is None:
            return v

        if "type" in v and v["type"] not in ["static", "dynamic"]:
            raise ValueError(f"Invalid bias type: {v['type']}, expected 'static' or 'dynamic'")

        if "method" in v and v["method"] not in ["mean", "max_min"]:
            raise ValueError(f"Invalid bias method: {v['method']}, expected 'mean' or 'max_min'")

        axis = [k for k in v.keys() if k not in ["type", "method"]]  # noqa: SIM118
        assert len(axis) > 0, "The axis for bias computation is not specified."
        for x in axis:
            if not isinstance(x, int):
                raise ValueError(f"Invalid axis type {type(axis)}, expected int")

        return v

    trt_high_precision_dtype: str = ModeloptField(
        default="Float",
        title="TRT StronglyType requires all weights and amax to be in the same dtype.",
        description="""The value is a string from ``["Float", "Half", "BFloat16"]``.
            The QDQs will be assigned the appropriate data type, and this variable will only be
            used when the user is exporting the quantized ONNX model.""",
        pattern=r"^Float$|^Half$|^BFloat16$",
    )

    calibrator: str | ConstructorLike = ModeloptField(
        default="max",
        title="""Specify the calibrator to use.""",
        description="""The calibrator can be a string from ``["max", "histogram"]`` or a constructor
        to create a calibrator which subclasses :class:`_Calibrator <modelopt.torch.quantization.calib._Calibrator>`.
        See :meth:`standardize_constructor_args <modelopt.torch.utils.network.standardize_constructor_args>`
        for more information on how to specify the constructor.""",
    )

    @field_validator("calibrator")
    @classmethod
    def validate_calibrator(cls, v, info: ValidationInfo):
        """Validate calibrator."""
        if isinstance(v, str):
            assert v in ["max", "histogram"]
        return v

    rotate: bool = ModeloptField(
        default=False,
        title="""If rotate the input before quantization.""",
        description=""""If true, the input of the quantizer will be rotated with a hadamard matrix
        given by scipy.linalg.hadamard, i.e.
        ``input = input @ scipy.linalg.hadamard(input.shape[-1]) / sqrt(input.shape[-1])``.

        This can be used for ratation based PTQ methods, e.g. QuaRot or SpinQuant.
        See https://arxiv.org/abs/2404.00456 for example.""",
    )

    pass_through_bwd: bool = ModeloptField(
        default=False,
        title="If set to true, fake quantization will be a pass through for gradient computation.",
        description="""
        Gradient computation where fake quantization is pass through is called
        'Straight-Through Estimator (STE)'. STE does not require saving of the input tensor for
        performing backward pass and hence consumes less memory.

        If set to False, we will use STE with zeroed outlier gradients. This setting could
        yield better QAT accuracy depending on the quantization format. However, this setting
        requires saving of the input tensor for computing gradients which uses more memory.

        For dynamic quantization formats like MXFP4, STE with zeroed outlier gradients
        is not needed since fake quantization with dynamic amax results in minimal/no clipping.
        """,
    )


class QuantizeAlgorithmConfig(ModeloptBaseConfig):
    """Calibration algorithm config base."""

    method: Literal[None] = ModeloptField(
        None,
        title="This field specifies the name of the calibration algorithm. If None, no calibration is performed.",
    )


class SVDQuantConfig(QuantizeAlgorithmConfig):
    """The config for SVDQuant.

    Refer to the `SVDQuant paper <https://arxiv.org/pdf/2411.05007>`_ for more details.
    """

    method: Literal["svdquant"] = ModeloptField("svdquant")

    lowrank: int | None = ModeloptField(
        default=32,
        title="Low-rank dimension for the SVD LoRA",
        description=(
            "Specifies the rank of the LoRA used in the SVDQuant method, "
            "which captures outliers from the original weights."
        ),
    )


# QuantizeQuantCfgType = dict[
#     str | Callable,
#     QuantizerAttributeConfig
#     | list[QuantizerAttributeConfig]
#     | dict[str | Callable, QuantizerAttributeConfig | list[QuantizerAttributeConfig]],
# ]

# _QuantizeAlgoCfgType = str | dict | QuantizeAlgorithmConfig | None

# QuantizeAlgoCfgType = _QuantizeAlgoCfgType | list[_QuantizeAlgoCfgType] | None


# TODO Jingyu Xin
class PEFTConfig(ModeloptBaseConfig):
    """Default configuration for ``peft`` mode."""

    adapter_name: str = ModeloptField(
        default="default",
        title="Placeholder",
        validate_default=True,
    )

    adapter_cfg: dict = ModeloptField(
        default={"default": {"rank": 128}},
        title="Placeholder",
        validate_default=True,
    )

    adapter_type: str = ModeloptField(
        default="lora",
        title="Placeholder",
        validate_default=True,
    )


class ExportPEFTConfig(ModeloptBaseConfig):
    """An empty config."""


class CompressConfig(ModeloptBaseConfig):
    """Default configuration for ``compress`` mode."""

    compress: dict[str, bool] = ModeloptField(
        default={"*": True},
        title="""Enable weight compression for the given pattern. Default is False for all weights.
        Call `compress` function to compress the model weights.""",
    )

    quant_gemm: bool = ModeloptField(
        default=True,
        title="Enable quantized GEMM.",
        description="If True, quantized GEMM compute will be enabled. Otherwise, we only do weight-only quantization.",
    )


CompressCfgType = dict[str, bool] | None | CompressConfig


class _QuantizeExportConfig(ModeloptBaseConfig):
    """An empty config."""


def need_calibration(config):
    """Check if calibration is needed for the given config."""
    if config["algorithm"] is not None and config["algorithm"] != "max":
        return True

    def _not_dynamic(cfg):
        return (
            cfg.get("enable", True)
            and cfg.get("type", "") != "dynamic"
            and cfg.get("*", {}).get("enable", True)
        )

    for name, cfg in config.get("quant_cfg", {}).items():
        if "weight_quantizer" in name:
            # We don't calibrate weight quantizer
            continue
        # quantization like W4A8 has a list of weight quantizers
        if isinstance(cfg, list):
            for _config in cfg:
                if _not_dynamic(_config):
                    print(f"{cfg}: True")
                    return True
        elif _not_dynamic(cfg):
            print(f"{cfg}: True")
            return True

    return False
