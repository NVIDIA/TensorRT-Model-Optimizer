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

"""This document lists the quantization formats supported by Model Optimizer and example quantization configs.

.. _quantization-formats:

Quantization Formats
==========================================

The following table lists the quantization formats supported by Model Optimizer and the corresponding quantization
config. See :ref:`Quantization Configs <example-quantization-configs>` for the
specific quantization config definitions.

Please see :doc:`choosing the right quantization formats <../../guides/_choosing_quant_methods>` to
learn more about the formats and their use-cases.

.. note::

    The recommended configs given below are for LLM models. For CNN models, only INT8 quantization
    is supported. Please use quantization config ``INT8_DEFAULT_CFG`` for CNN models.

=================================   =======================================================
Quantization  Format                Model Optimizer config
=================================   =======================================================
INT8                                ``INT8_SMOOTHQUANT_CFG``

FP8                                 ``FP8_DEFAULT_CFG``

INT4 Weights only AWQ (W4A16)       ``INT4_AWQ_CFG``

INT4-FP8 AWQ (W4A8)                 ``W4A8_AWQ_BETA_CFG``

=================================   =======================================================

.. _quantization-configs:

Quantization Configs
================================

Quantization config is dictionary specifying the values for keys ``"quant_cfg"`` and
``"algorithm"``. The ``"quant_cfg"`` key specifies the quantization configurations. The
``"algorithm"`` key specifies the ``algorithm`` argument to
:meth:`calibrate <modelopt.torch.quantization.model_calib.calibrate>`. Please see :class:`QuantizeConfig`
for the quantization config definition.

'Quantization configurations' is a dictionary mapping wildcards or filter functions
to its 'quantizer attributes'. The wildcards or filter functions  are matched
against the quantizer module names. The quantizer modules have names ending with
``weight_quantizer`` and ``input_quantizer`` and they perform weight quantization and
input quantization (or activation quantization) respectively. The quantizer modules are generally
instances of
:class:`TensorQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.TensorQuantizer>`.
The quantizer attributes are defined by :class:`QuantizerAttributeConfig`. See :class:`QuantizerAttributeConfig`
for details on the quantizer attributes and their values.

The key `"default"` from the quantization configuration dictionary is applied if no other wildcard or filter functions
match the quantizer module name.

The quantizer attributes are applied in the order they are specified. For the missing attributes, the default attributes
as defined by :class:`QuantizerAttributeConfig` are used.

Quantizer attributes can also be a list of dictionaries. In this case, the matched quantizer module
is replaced with a
:class:`SequentialQuantizer <modelopt.torch.quantization.nn.modules.tensor_quantizer.SequentialQuantizer>`
module which is used to quantize a tensor in multiple formats sequentially. Each quantizer attribute
dictionary in the list specifies the quantization formats for each quantization step of the
sequential quantizer. For example, `SequentialQuantizer` is used in 'INT4 Weights, FP8 Activations'
quantization in which the weights are quantized in INT4 followed by FP8.

In addition, the dictionary entries could also be pytorch module class names mapping the class specific
quantization configurations. The pytorch modules should have a quantized equivalent.

To get the string representation of a module class, do:

.. code-block::

    from modelopt.torch.quantization import QuantModuleRegistry

    # Get the class name for nn.Conv2d
    class_name = QuantModuleRegistry.get_key(nn.Conv2d)

Here is an example of a quantization config:

.. code-block::

    MY_QUANT_CFG = {
        "quant_cfg": {
            # Quantizer wildcard strings mapping to quantizer attributes
            "*weight_quantizer": {"num_bits": 8, "axis": 0},
            "*input_quantizer": {"num_bits": 8, "axis": None},

            # Module class names mapping to quantizer configurations
            "nn.LeakyReLU": {"*input_quantizer": {"enable": False}},

        }
    }

.. _example-quantization-configs:

Example Quantization Configurations
==========================================

These example configs can be accessed as attributes of ``modelopt.torch.quantization`` and can be given as
input to :meth:`mtq.quantize() <modelopt.torch.quantization.model_quant.quantize>`. For example:

.. code-block::

    import modelopt.torch.quantization as mtq
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)

You can also create your own config by following these examples.
For instance, if you want to quantize a model with int4 AWQ algorithm, but need to skip quantizing
the layer named ``lm_head``,  you can create a custom config and quantize your model as following:

.. code-block::

    # Create custom config
    CUSTOM_INT4_AWQ_CFG = copy.deepcopy(mtq.INT4_AWQ_CFG)
    CUSTOM_INT4_AWQ_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}

    # quantize model
    model = mtq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)

"""

from collections.abc import Callable
from typing import Literal

from pydantic import ValidationInfo, field_validator, model_validator

# use the form "from x import y as y" syntax to direct mypy that these configs are re-exported
from modelopt.core.torch.quantization.config import NVFP4_AFFINE_KV_CFG as NVFP4_AFFINE_KV_CFG
from modelopt.core.torch.quantization.config import NVFP4_AWQ_CLIP_CFG as NVFP4_AWQ_CLIP_CFG
from modelopt.core.torch.quantization.config import NVFP4_AWQ_FULL_CFG as NVFP4_AWQ_FULL_CFG
from modelopt.core.torch.quantization.config import NVFP4_AWQ_LITE_CFG as NVFP4_AWQ_LITE_CFG
from modelopt.core.torch.quantization.config import NVFP4_DEFAULT_CFG as NVFP4_DEFAULT_CFG
from modelopt.core.torch.quantization.config import NVFP4_KV_CFG as NVFP4_KV_CFG
from modelopt.core.torch.quantization.config import NVFP4_KV_ROTATE_CFG as NVFP4_KV_ROTATE_CFG
from modelopt.core.torch.quantization.config import NVFP4_MXFP8_CFG as NVFP4_MXFP8_CFG
from modelopt.core.torch.quantization.config import (
    NVFP4_SVDQUANT_DEFAULT_CFG as NVFP4_SVDQUANT_DEFAULT_CFG,
)
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.utils.network import ConstructorLike

_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},  # In Whisper model, lm_head has key name proj_out
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "*mlp.gate.*": {"enable": False},  # Skip the MOE router
    "*mlp.shared_expert_gate.*": {"enable": False},  # Skip the MOE router
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
}

INT8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

INT8_SMOOTHQUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "smoothquant",
}

FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

FP8_PER_CHANNEL_PER_TOKEN_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": 0},
        "*input_quantizer": {
            "num_bits": (4, 3),
            "type": "dynamic",
            "block_sizes": {-1: None},
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

# FP8 2D blockwise fake quantization config for deepseek models
FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 128, -2: 128},
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
        "*input_quantizer": {"enable": False},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}

INT4_AWQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 128, "type": "static"},
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

# W4A8 currently uses INT4 blockwise quantization (block size = 128) followed by FP8 quantization
# for weights. This could change in the future
W4A8_AWQ_BETA_CFG = {
    "quant_cfg": {
        "*weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": "awq_lite",
}

MXFP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": None,
}

MXFP6_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (3, 2),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (3, 2),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": None,
}

MXFP4_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": None,
}

W4A8_MXFP4_FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": None,
}

MXINT8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 8,
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": 8,
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "enable": True,
        },
        **_default_disabled_quantizer_cfg,
    },
    "algorithm": None,
}

FP8_KV_CFG = {
    "quant_cfg": {
        "*[kv]_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

FP8_AFFINE_KV_CFG = {
    "quant_cfg": {
        "*[kv]_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "bias": {-2: None, -4: None, "type": "static"},
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

choices: set[str] = {
    "INT8_DEFAULT_CFG",
    "INT8_SMOOTHQUANT_CFG",
    "FP8_DEFAULT_CFG",
    "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",
    "FP8_PER_CHANNEL_PER_TOKEN_CFG",
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "INT4_AWQ_CFG",
    "W4A8_AWQ_BETA_CFG",
    "NVFP4_DEFAULT_CFG",
    "NVFP4_AWQ_LITE_CFG",
    "NVFP4_AWQ_CLIP_CFG",
    "NVFP4_AWQ_FULL_CFG",
    "NVFP4_KV_ROTATE_CFG",
    "FP8_KV_CFG",
    "FP8_AFFINE_KV_CFG",
    "NVFP4_KV_CFG",
    "NVFP4_AFFINE_KV_CFG",
    "MXFP8_DEFAULT_CFG",
    "W4A8_MXFP4_FP8_CFG",
}

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


class QuantizeAlgorithmConfig(ModeloptBaseConfig):
    """Calibration algorithm config base."""

    method: Literal[None] = ModeloptField(
        None,
        title="This field specifies the name of the calibration algorithm. If None, no calibration is performed.",
    )


class MaxCalibConfig(QuantizeAlgorithmConfig):
    """The config for max calibration algorithm.

    Max calibration estimates max values of activations or weights and use this max values
    to set the quantization scaling factor.
    See `Integer Quantization <https://arxiv.org/pdf/2004.09602>`_ for the concepts.
    """

    method: Literal["max"] = ModeloptField("max")

    distributed_sync: bool | None = ModeloptField(
        default=True,
        title="Whether to sync the amax across the distributed processes.",
        description="If True, the amax will be synced across the distributed processes.",
    )


class SmoothQuantCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``smoothquant`` algorithm (SmoothQuant).

    SmoothQuant applies a smoothing factor which balances the scale of outliers in weights and activations.
    See `SmoothQuant paper <https://arxiv.org/pdf/2211.10438>`_ for more details.
    """

    method: Literal["smoothquant"] = ModeloptField("smoothquant")

    alpha: float | None = ModeloptField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="SmoothQuant hyper-parameter alpha.",
        description=(
            "This hyper-parameter controls the migration strength."
            "The migration strength is within [0, 1], "
            "a larger value migrates more quantization difficulty to weights."
        ),
    )


class AWQLiteCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_lite`` (AWQ lite) algorithm.

    AWQ lite applies a channel-wise scaling factor which minimizes the output difference after quantization.
    See `AWQ paper <https://arxiv.org/pdf/2306.00978>`_ for more details.
    """

    method: Literal["awq_lite"] = ModeloptField("awq_lite")

    alpha_step: float | None = ModeloptField(
        default=0.1,
        gt=0.0,
        le=1.0,
        title="Step size for the searching alpha.",
        description="The alpha will be searched from 0 to 1 with the step size specified.",
    )

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named `awq_lite`.",
    )


class AWQClipCalibConfig(QuantizeAlgorithmConfig):
    """The config for ``awq_clip`` (AWQ clip) algorithm.

    AWQ clip searches clipped amax for per-group quantization, This search requires much more compute
    compared to AWQ lite. To avoid any OOM, the linear layer weights are batched along the ``out_features``
    dimension of batch size ``max_co_batch_size``. AWQ clip calibration also takes longer than AWQ lite.
    """

    method: Literal["awq_clip"] = ModeloptField("awq_clip")

    max_co_batch_size: int | None = ModeloptField(
        default=1024,
        title="Maximum output channel batch size while searching clip values.",
        description="Reduce this number if CUDA Out of Memory error occurs.",
    )

    max_tokens_per_batch: int | None = ModeloptField(
        default=64,
        title="Maximum tokens per batch while searching clip values.",
        description="""The total tokens used for clip search would be ``max_tokens_per_batch * number of batches``.
        Original AWQ uses a total of 512 tokens to search for clip values.""",
    )

    min_clip_ratio: float | None = ModeloptField(
        default=0.5,
        gt=0.0,
        lt=1.0,
        title="Minimum clip ratio to search for.",
        description="""It should be in (0, 1.0). Clip will search for the optimal clipping value in the range
        ``[original block amax * min_clip_ratio, original block amax]``.""",
    )

    shrink_step: float | None = ModeloptField(
        default=0.05,
        gt=0.0,
        le=1.0,
        title="Step size to search for clip values.",
        description="""It should be in range (0, 1.0]. The clip ratio will be searched from ``min_clip_ratio`` to 1
        with the step size specified.""",
    )

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description="If True, module's search metadata will be kept as a module attribute named ``awq_clip``.",
    )


class AWQFullCalibConfig(AWQLiteCalibConfig, AWQClipCalibConfig):
    """The config for ``awq`` or ``awq_full`` algorithm (AWQ full).

    AWQ full performs ``awq_lite`` followed by ``awq_clip``.
    """

    method: Literal["awq_full"] = ModeloptField("awq_full")

    debug: bool | None = ModeloptField(
        default=False,
        title="Debug mode.",
        description=(
            "If True, module's search metadata will be kept as "
            "module attributes named ``awq_lite`` and ``awq_clip``."
        ),
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


QuantizeQuantCfgType = dict[
    str | Callable,
    QuantizerAttributeConfig
    | list[QuantizerAttributeConfig]
    | dict[str | Callable, QuantizerAttributeConfig | list[QuantizerAttributeConfig]],
]

_QuantizeAlgoCfgType = str | dict | QuantizeAlgorithmConfig | None

QuantizeAlgoCfgType = _QuantizeAlgoCfgType | list[_QuantizeAlgoCfgType] | None


class QuantizeConfig(ModeloptBaseConfig):
    """Default configuration for ``quantize`` mode."""

    quant_cfg: QuantizeQuantCfgType = ModeloptField(
        default={"default": {"num_bits": 8, "axis": None}},
        title="Quantization configuration",
        validate_default=True,
    )

    algorithm: QuantizeAlgoCfgType = ModeloptField(
        default="max",
        title="Calibration algorithm, see :meth:`calibrate <modelopt.torch.quantization.model_quant.calibrate>` "
        "for more details.",
        validate_default=True,
    )


class CompressConfig(ModeloptBaseConfig):
    """Default configuration for ``compress`` mode."""

    compress: dict[str, bool] = ModeloptField(
        default={"*": True},
        title="""Enable weight compression for the given pattern. Default is False for all weights.
        Call `compress` function to compress the model weights.""",
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
