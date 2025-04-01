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

"""TensorQuantizer Module."""

import contextlib
import math
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.onnx._globals import GLOBALS

from modelopt.torch.utils import standardize_constructor_args
from modelopt.torch.utils.distributed import DistributedProcessGroup

from ... import calib
from ... import utils as quant_utils
from ...calib.bias import add_bias, compute_bias, subtract_bias
from ...config import QuantizerAttributeConfig
from ...qtensor import (
    BaseQuantizedTensor,
    FP8QTensor,
    INT8QTensor,
    INT4QTensor,
    NF4QTensor,
    NVFP4QTensor,
    QTensorWrapper,
)
from ...tensor_quant import dynamic_block_quant, fake_tensor_quant, scaled_e4m3, static_block_quant
from ...utils import is_torch_export_mode
from ..functional import normalized_hadamard_transform

__all__ = ["TensorQuantizer", "SequentialQuantizer"]


class TensorQuantizer(nn.Module):
    """Tensor quantizer module.

    This module manages quantization and calibration of input tensor. It can perform fake (simulated quantization)
    or real quantization for various precisions and formats such as FP8 per-tensor, INT8 per-channel,
    INT4 per-block etc.

    If quantization is enabled, it calls the appropriate quantization functional and
    returns the quantized tensor. The quantized tensor data type will be same as the input tensor data type for
    fake quantization. During calibration mode, the module collects the statistics using its calibrator.

    The quantization parameters are as described in
    :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`. They can be set
    at initialization using ``quant_attribute_cfg`` or later by calling :meth:`set_from_attribute_config`.

    Args:
        quant_attribute_cfg: An instance of
            :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>` or None.
            If None, default values are used.
        if_quant: A boolean. If True, quantization is enabled in the forward path.
        if_calib: A boolean. If True, calibration is enabled in the forward path.
        amax: None or an array like object such as list, tuple, numpy array, scalar
            which can be used to construct amax tensor.
    """

    def __init__(
        self,
        quant_attribute_cfg=None,
        if_quant=True,
        if_calib=False,
        amax=None,
    ):
        """Initialize quantizer and set up required variables."""
        super(TensorQuantizer, self).__init__()
        quant_attribute_cfg = (
            quant_attribute_cfg if quant_attribute_cfg is not None else QuantizerAttributeConfig()
        )
        if amax is not None:
            self.amax = amax

        self.set_from_attribute_config(quant_attribute_cfg)

        self._if_quant = if_quant
        self._if_calib = if_calib
        self._enable_pre_quant_scale = True
        self._dequantize = False
        self._input_dtype = None

        # Lazy initialize the bias calibrator for KV cache quantization
        self._bias_calibrator = None

    def set_from_attribute_config(self, attribute_cfg: Union[QuantizerAttributeConfig, dict]):
        """Set quantizer attributes from attribute_dict.

        The attributes are defined in
        :class:`QuantizerAttributeConfig <modelopt.torch.quantization.config.QuantizerAttributeConfig>`.
        """

        def _calibrator_setter(val):
            if val in ["max", "histogram"]:
                calib_cls = calib.MaxCalibrator if val == "max" else calib.HistogramCalibrator
                args, kwargs = (self._num_bits, self._axis, self._unsigned), {}
            else:
                calib_cls, args, kwargs = standardize_constructor_args(val)
            return calib_cls(*args, **kwargs)

        # Some attributes need custom handling.
        # By default, attributes from config are mapped to a name ``f"_{attribute}"``
        _custom_setters: dict[str, tuple[str, Callable]] = {
            "enable": ("_disabled", lambda val: val is False),
            "type": ("_dynamic", lambda val: val == "dynamic"),
            "calibrator": ("_calibrator", _calibrator_setter),
        }

        for attribute, val in attribute_cfg.items():
            assert attribute in QuantizerAttributeConfig.model_fields, (
                f"{attribute} is not a valid `TensorQuantizer` attribute"
            )
            _tq_attribute_name, _setter = _custom_setters.get(
                attribute, (f"_{attribute}", lambda v: v)
            )
            setattr(self, _tq_attribute_name, _setter(val))

    def dequantize(self, qtensor: BaseQuantizedTensor):
        """De-quantize a real quantized tensor to a given dtype."""
        kwarg = {
            "scale": self._scale,
            "block_sizes": self.block_sizes,
            "double_scale": getattr(self, "_double_scale", None),
            "scale_zeros": getattr(self, "_scale_zeros", None),
        }
        return qtensor.dequantize(**kwarg)

    @property
    def num_bits(self):
        """Return num_bits for quantization."""
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        self._num_bits = value
        self._calibrator._num_bits = value

    @property
    def maxbound(self):
        """Return maxbound for quantization."""
        if self._num_bits == (4, 3):
            return 448.0
        if self._num_bits == (2, 1) and self._block_sizes.get("scale_bits") == (4, 3):
            return 6.0
        return (1 << (self._num_bits - 1 + int(self._unsigned))) - 1

    @property
    def unsigned(self):
        """Return True if unsigned quantization is used."""
        return self._unsigned

    @unsigned.setter
    def unsigned(self, value):
        self._unsigned = value
        self._calibrator._unsigned = value

    @property
    def pre_quant_scale(self):
        """Return pre_quant_scale used for smoothquant."""
        if not hasattr(self, "_pre_quant_scale") or not self._enable_pre_quant_scale:
            return None
        return self._pre_quant_scale

    @pre_quant_scale.setter
    def pre_quant_scale(self, value):
        assert value is not None, "pre_quant_scale cannot be set to None."
        assert self._enable_pre_quant_scale, (
            "pre_quant_scale cannot be set when forward_with_pre_quant_scale is False."
        )
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_pre_quant_scale"):
            self.register_buffer("_pre_quant_scale", value.clone().detach())
        else:
            if self._pre_quant_scale.shape != value.shape:
                raise RuntimeError("Changing shape when setting pre_quant_scale is not allowed.")
            self._pre_quant_scale.data.copy_(
                value.clone().detach().to(self._pre_quant_scale.device)
            )

    @property
    def amax(self):
        """Return amax for quantization."""
        if not hasattr(self, "_amax") or self.is_mx_format:
            return None
        assert not self._dynamic, "Dynamic quantization does not have fixed amax"
        return self._amax

    @amax.setter
    def amax(self, value):
        assert value is not None, "amax cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", value.clone().detach())
        else:
            if self._amax.shape != value.shape:
                raise RuntimeError("Changing shape when setting amax is not allowed.")
            self._amax.data.copy_(value.clone().detach().to(self._amax.device))

    def reset_amax(self):
        """Reset amax to None."""
        if hasattr(self, "_amax"):
            delattr(self, "_amax")
        self._calibrator.reset()

    @property
    def step_size(self):
        """Return step size for integer quantization."""
        if not hasattr(self, "_amax"):
            warnings.warn("step_size is undefined under dynamic amax mode!")
            return None
        assert isinstance(self._num_bits, int), (
            "Step size is not defined for non-integer quantization."
        )
        return self._amax / (2.0 ** (self._num_bits - 1 + int(self._unsigned)) - 1.0)

    @property
    def axis(self):
        """Return axis for quantization."""
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value
        self._calibrator._axis = value

    @property
    def block_sizes(self):
        """Return block_sizes for quantization."""
        return self._block_sizes

    @block_sizes.setter
    def block_sizes(self, value):
        self._axis = None
        self._block_sizes = value

    @property
    def bias(self):
        """Return bias for quantization."""
        if not hasattr(self, "_bias"):
            return None
        return self._bias

    @property
    def bias_axis(self):
        """Return bias_axis for quantization."""
        if not hasattr(self, "_bias_axis"):
            return None
        return self._bias_axis

    @bias_axis.setter
    def bias_axis(self, value):
        assert value is not None, "bias_axis cannot be set to None."
        assert isinstance(value, (tuple, list)), "bias_axis must be a tuple or a list."
        self._bias_axis = value

    @property
    def bias_method(self):
        """Return bias_method for quantization."""
        if self._bias is None:
            return None
        return self._bias.get("method", "mean")

    @property
    def bias_type(self):
        """Return bias_type for quantization."""
        if self._bias is None:
            return None
        return self._bias.get("type", "static")

    @bias_type.setter
    def bias_type(self, value):
        assert value in ["static", "dynamic"], "bias_type must be either 'static' or 'dynamic'."
        self._bias["type"] = value

    @property
    def bias_value(self):
        """Return bias for quantization."""
        if not hasattr(self, "_bias_value"):
            return None
        return self._bias_value

    @bias_value.setter
    def bias_value(self, value):
        assert value is not None, "bias cannot be set to None."

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if not hasattr(self, "_bias_value"):
            self.register_buffer("_bias_value", value.clone().detach())
        else:
            if self._bias_value.shape != value.shape:
                raise RuntimeError("Changing shape when setting bias is not allowed.")
            self._bias_value.data.copy_(value.clone().detach().to(self._bias_value.device))

    @property
    def bias_calibrator(self):
        """Return bias_calibrator for quantization."""
        # Get reduce_axis from bias config
        # Bias calibration supports per-channel and per-token quantization
        if self._bias_calibrator is None and self.bias is not None:
            self.bias_axis = tuple(k for k in self.bias.keys() if isinstance(k, int))
            if self._bias is not None:
                self._bias_calibrator = calib.BiasCalibrator(
                    method=self.bias_method,
                    axis=self.bias_axis,
                )

        return self._bias_calibrator

    @property
    def fake_quant(self):
        """Return True if fake quantization is used."""
        return self._fake_quant

    @fake_quant.setter
    def fake_quant(self, value):
        self._fake_quant = value

    @property
    def narrow_range(self):
        """Return True if symmetric integer range for signed quantization is used."""
        return self._narrow_range

    @narrow_range.setter
    def narrow_range(self, value):
        self._narrow_range = value

    @property
    def is_enabled(self):
        """Return true if the modules is not disabled."""
        return not self._disabled

    def disable(self):
        """Bypass the module.

        Neither of calibration, clipping and quantization will be performed if the module is disabled.
        """
        self._disabled = True

    def enable(self):
        """Enable the module."""
        self._disabled = False

    @property
    def trt_high_precision_dtype(self):
        """Return True if FP16 AMAX is used when exporting the model."""
        return self._trt_high_precision_dtype

    @trt_high_precision_dtype.setter
    def trt_high_precision_dtype(self, value):
        self._trt_high_precision_dtype = value

    @property
    def is_mx_format(self):
        """Check if is MX formats."""
        return (
            self.block_sizes is not None
            and self.block_sizes.get("type", None) == "dynamic"
            and self.block_sizes.get("scale_bits", None) == (8, 0)
        )

    def disable_calib(self):
        """Disable calibration."""
        self._if_calib = False

    def enable_calib(self):
        """Enable calibration."""
        if self._calibrator is None:
            raise ValueError("Calibrator was not created, cannot enable calibration.")

        # Dynamic quantization does not need calibration.
        if self._dynamic:
            return
        self._if_calib = True

    def disable_quant(self):
        """Disable quantization."""
        self._if_quant = False

    def enable_quant(self):
        """Enable quantization."""
        self._if_quant = True

    def load_calib_amax(self, *args, **kwargs):
        """Load amax from calibrator.

        Updates the amax buffer with value computed by the calibrator, creating it if necessary.
        ``*args`` and ``**kwargs`` are directly passed to ``compute_amax``, except ``"strict"`` in
        ``kwargs``. Refer to ``compute_amax`` for more details.
        """
        assert not self._dynamic, "Dynamic quantization does not need calibration."

        strict = kwargs.pop("strict", True)
        if getattr(self, "_calibrator", None) is None:
            raise RuntimeError("Calibrator not created.")
        calib_amax = self._calibrator.compute_amax(*args, **kwargs)
        if calib_amax is None:
            err_msg = (
                "Calibrator returned None. This usually happens when calibrator hasn't seen any"
                " tensor."
            )
            if not strict:
                warnings.warn(err_msg)
                warnings.warn("Set amax to NaN!")
                calib_amax = torch.tensor(math.nan)
            else:
                raise RuntimeError(
                    err_msg
                    + " Passing 'strict=False' to `load_calib_amax()` will ignore the error."
                )
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", calib_amax.clone().detach())
        else:
            self._amax.data.copy_(calib_amax.clone().detach())

    def load_calib_bias(self, *args, **kwargs):
        """Load affine bias for quantization."""
        assert not self._dynamic, "Dynamic quantization does not need calibration."
        calib_bias = self.bias_calibrator.compute_calib_bias(*args, **kwargs)
        if calib_bias is None:
            raise RuntimeError(
                "Calibrator returned None. This usually happens when calibrator hasn't seen any tensor."
            )

        if not hasattr(self, "_bias_value"):
            self.register_buffer("_bias_value", calib_bias.clone().detach())
        else:
            self._bias_value.data.copy_(calib_bias.clone().detach())

    def _get_amax(self, inputs):
        """Get amax from buffer or compute it dynamically."""
        if hasattr(self, "_amax"):
            amax = self._amax
        else:
            if self._axis is None:
                reduce_axis = None
            else:
                reduce_axis = []
                # Swap axis to reduce
                axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
                for i in range(inputs.dim()):
                    if not (i in axis or (i - inputs.dim()) in axis):
                        reduce_axis.append(i)
            amax = quant_utils.reduce_amax(inputs, axis=reduce_axis, keepdims=True).detach()

        amax = amax.detach() if is_torch_export_mode() else amax.data
        return amax

    def _validate_amax(self, amax):
        # Dynamic control flow is not supported by torch dynamo
        if not is_torch_export_mode():
            assert torch.all(amax >= 0) and not torch.any(torch.isinf(amax)), (
                f"Got invalid amax: {amax}"
            )

    def _is_real_quantize_support(self):
        """Check if real quantization is supported for this quant config."""
        if (
            (self._num_bits == 4 and self._block_sizes)  # NF4 and Int4
            or (self._num_bits == (2, 1) and self._block_sizes)  # NVFP4
            or (self._num_bits == (4, 3))  # FP8
        ):
            return True
        return False

    def _real_quantize(self, inputs):
        assert self._is_real_quantize_support(), "Real quantization not supported for this format."

        buffer_to_register = {}
        if self._num_bits == (4, 3):
            # FP8 quantization
            outputs, _scale = FP8QTensor.quantize(
                inputs, axis=self._axis, block_sizes=self._block_sizes
            )
            buffer_to_register["_scale"] = _scale
        elif self._num_bits == 8:
            outputs, _scale = INT8QTensor.quantize(
                inputs, axis=self._axis, block_sizes=self._block_sizes
            )
            buffer_to_register["_scale"] = _scale
        elif self._block_sizes.get("scale_bits", 0) == 8 and self._block_sizes.get(
            "scale_block_sizes", None
        ):
            # NF4 double quantization
            # Return real quantized tensor class and store scales inside the TensorQuantizer
            outputs, scales = NF4QTensor.quantize(inputs, self._block_sizes[-1])
            _scale, _double_scale, _scale_zeros = NF4QTensor.double_quantization(
                scales,
                self._block_sizes["scale_block_sizes"][-1],
                self._block_sizes["scale_bits"],
            )
            buffer_to_register["_scale"] = _scale
            buffer_to_register["_double_scale"] = _double_scale
            buffer_to_register["_scale_zeros"] = _scale_zeros
        elif self._block_sizes.get("scale_bits") == (4, 3):
            # NVFP4 default quantization
            # Return real quantized tensor and store scales inside TensorQuantizer
            outputs, _weights_scaling_factor, _weights_scaling_factor_2 = NVFP4QTensor.quantize(
                inputs,
                self._block_sizes[-1],
            )
            buffer_to_register["_scale"] = _weights_scaling_factor.to(torch.float32)
            buffer_to_register["_double_scale"] = _weights_scaling_factor_2
        else:
            outputs, _scale = INT4QTensor.quantize(inputs, self._block_sizes[-1])
            buffer_to_register["_scale"] = _scale
        for k, v in buffer_to_register.items():
            self.register_buffer(k, v)

        return outputs

    def _compute_dynamic_bias(self, inputs):
        """Compute dynamic bias based on current inputs."""
        if self.bias_method == "mean":
            # mean = (max + min) / 2
            return compute_bias(inputs, self.bias_axis, method="mean")
        elif self.bias_method == "max_min":
            # mean = average(all tokens)
            return compute_bias(inputs, self.bias_axis, method="max_min")
        else:
            raise ValueError(f"Unknown bias method: {self.bias_method}")

    def _validate_static_bias(self):
        """Validate static bias exists."""
        assert self.bias_value is not None, "Bias is not set for static bias quantization."

    def _handle_bias_before_quantization(self, inputs, bias_type):
        """Handle bias subtraction for quantization."""
        # Compute/validate bias
        if bias_type == "dynamic":
            # Compute bias and subtract it from input tensor in dynamic affine quantization
            self.bias_value = self._compute_dynamic_bias(inputs)
        elif bias_type == "static":
            self._validate_static_bias()
        else:
            raise ValueError(f"Unknown bias type: {bias_type}")

        # Subtract bias from input tensor in dynamic/static affine quantization
        inputs = subtract_bias(inputs, self.bias_value)
        return inputs

    def _handle_bias_after_quantization(self, inputs, bias_type):
        """Handle bias addition for quantization."""
        # Add bias back to output tensor in affine quantization
        if bias_type == "dynamic" or bias_type == "static":
            inputs = add_bias(inputs, self.bias_value)
        return inputs

    def _quant_forward(self, inputs):
        """Quantized forward pass."""
        amax = None
        if not self._dequantize and not self.is_mx_format:
            amax = self._get_amax(inputs)
            self._validate_amax(amax)

        if self._fake_quant:
            if self.block_sizes is not None:
                # Block quantization, including dynamic and static block quantization
                block_size = self.block_sizes.get(-1, None) or self.block_sizes.get(
                    inputs.dim() - 1, None
                )
                # Subtract bias from input tensor in affine quantization
                if self.bias_calibrator is not None:
                    inputs = self._handle_bias_before_quantization(inputs, self.bias_type)

                if self.block_sizes.get("type", "static") == "dynamic":
                    # Dynamic block quantization, e.g., NVFP4
                    # Double quantization is supported
                    assert block_size is not None, "block size for dynamic quantization not found."

                    outputs = dynamic_block_quant(
                        inputs,
                        block_size,
                        amax,
                        self._num_bits,
                        self.block_sizes.get("scale_bits", None),
                        getattr(self, "_trt_high_precision_dtype", None),
                        getattr(self, "_onnx_quantizer_type", None),
                    )
                else:
                    # Static block quantization, e.g., INT4_BLOCKWISE
                    # Double quantization is not supported
                    outputs = static_block_quant(
                        inputs,
                        amax,
                        self._num_bits,
                        self._unsigned,
                        self._narrow_range,
                    )

                # Add bias back to output tensor in affine quantization
                if self.bias_calibrator is not None:
                    outputs = self._handle_bias_after_quantization(outputs, self.bias_type)
            elif isinstance(self._num_bits, tuple):
                # Float-point quantization, e.g., FP8
                E, M = self._num_bits  # noqa: N806

                # Subtract bias from input tensor in affine quantization
                if self.bias_calibrator is not None:
                    inputs = self._handle_bias_before_quantization(inputs, self.bias_type)

                outputs = scaled_e4m3(
                    inputs, self._get_amax(inputs), E, M, self._trt_high_precision_dtype
                )

                # Add bias back to output tensor in affine quantization
                if self.bias_calibrator is not None:
                    outputs = self._handle_bias_after_quantization(outputs, self.bias_type)
            else:
                # Integer quantization, e.g., INT8
                outputs = fake_tensor_quant(
                    inputs,
                    amax,
                    self._num_bits,
                    self._unsigned,
                    self._narrow_range,
                    self._trt_high_precision_dtype,
                )
        else:
            # Real quantize
            if not self._dequantize:
                outputs = self._real_quantize(inputs)
                self._dequantize = True
            else:
                # De-quantize
                if isinstance(inputs, QTensorWrapper):
                    inputs = inputs.get_qtensor()
                assert isinstance(inputs, BaseQuantizedTensor), (
                    "Expected input as real quantized tensors."
                )
                return self.dequantize(inputs)

        return outputs

    def _check_onnx_readiness(self, inputs):
        """Check if quantizer is ready for ONNX export."""
        assert hasattr(self, "_amax"), (
            "Quantizer has not been calibrated. ONNX export requires the quantizer to be"
            " calibrated. Calibrate and load amax before exporting to ONNX."
        )

        if self._if_calib:
            warnings.warn(
                "Quantizer is in calibration mode. "
                "Please complete calibration before exporting to ONNX for correct results."
            )

        amax = self._get_amax(inputs)

        # We only support scalar amax for E4M3 ONNX export
        if isinstance(self.num_bits, tuple):
            assert amax.numel() == 1, (
                "E4M3 supports ONNX export only for per-tensor quantization."
                " Per-tensor quantization requires scalar amax. "
                f"Received non-scalar amax of shape: {amax.shape}"
            )

    def _setup_for_blockquant(self, inputs: torch.Tensor):
        # Get reshape sizes and paddings for block-quantization
        def get_axis_quant_params(ax):
            ax = ax if ax in self.block_sizes else ax - inputs.dim()
            bsize = self.block_sizes.get(ax, None)
            padding, ax_slice = None, None
            if bsize is not None and inputs.shape[ax] % bsize != 0:
                padding = (bsize - (inputs.shape[ax] % bsize), 0)
                ax_slice = slice(inputs.shape[ax])
            return bsize, padding, ax_slice

        def set_quant_params(axis, block_reshape_size, padding, slices, amax_shape=None):
            self._axis = tuple(axis)
            if hasattr(self, "_calibrator"):
                self._calibrator._axis = self._axis
            self._original_shape = inputs.shape
            self._block_reshape_size = torch.Size(block_reshape_size)
            if padding is not None:
                self._padding = tuple(padding)
                self._original_shape = F.pad(inputs, self._padding, "constant", 0).shape
            if slices is not None:
                self._slices = slices
            if amax_shape:
                self._amax_shape_for_export = amax_shape

        # Reshape size have already been set
        if hasattr(self, "_block_reshape_size"):
            return

        reshape_size, quantize_axis, paddings, slices = [], [], [], []

        # special handling for block-quantization along the last axis:
        # flatten the input for faster execution
        if (self.block_sizes.get(inputs.dim() - 1, None) or self.block_sizes.get(-1, None)) and len(
            QuantizerAttributeConfig._get_block_quant_axes_and_sizes(self.block_sizes)
        ) == 1:
            bsize, padding, ax_slice = get_axis_quant_params(inputs.dim() - 1)
            slices = None if ax_slice is None else (*(slice(None),) * (inputs.dim() - 1), ax_slice)
            padding = padding if not padding else tuple(reversed(padding))
            amax_shape_for_export = (*(inputs.shape[:-1]), -1)
            set_quant_params((0,), (-1, bsize), padding, slices, amax_shape_for_export)
            return

        for ax in range(inputs.dim()):
            bsize, padding, ax_slice = get_axis_quant_params(ax)
            paddings.append(padding)
            slices.append(ax_slice)
            if bsize is not None:
                reshape_size.extend([math.ceil(inputs.shape[ax] / bsize), bsize])
                quantize_axis.extend([True, False])
            else:
                reshape_size.append(inputs.shape[ax])
                quantize_axis.append(True)

        quant_axis = [i for i in range(len(quantize_axis)) if quantize_axis[i]]

        if all(s is None for s in slices):
            slices = None
        else:
            slices = [s if s else slice(None) for s in slices]

        if all(p is None for p in paddings):
            paddings = None
        else:
            new_paddings = []
            for padding in paddings:
                if not (new_paddings or padding):
                    continue
                new_paddings.extend(padding if padding else (0, 0))
            paddings = tuple(reversed(new_paddings))

        set_quant_params(quant_axis, reshape_size, paddings, slices)

    def _process_for_blockquant(self, inputs: torch.Tensor):
        if hasattr(self, "_padding"):
            inputs = F.pad(inputs, self._padding, "constant", 0)
        assert inputs.shape == self._original_shape, (
            f"Input shape has changed from {self._original_shape} to {inputs.shape}."
            " Block-quantization requires a fixed input shape."
        )
        inputs = inputs.reshape(self._block_reshape_size)
        return inputs

    def _reset_to_original_shape(self, outputs: torch.Tensor):
        outputs = outputs.reshape(self._original_shape)
        if hasattr(self, "_slices"):
            outputs = outputs[self._slices]
        return outputs

    def _block_sizes_to_axis(self, x: torch.Tensor):
        """Convert block_sizes to axis in per-channel/tensor quantization.

        For example, for input tensor with shape (B, T, H),
        {"block_sizes": {-1: None, -3: None}} equals to {axis: (-2)}, amax shape: (1, T, 1),
        {"block_sizes": {-1: None, -2: None, -3: None}} equals to {axis: None}, amax shape: (1, T, 1)
        """
        block_sizes = self._block_sizes
        if block_sizes is None:
            return

        def _check_per_channel_block_sizes(block_sizes):
            # Check per-channel/block quant
            return all(v is None for k, v in block_sizes.items() if isinstance(k, int))

        if _check_per_channel_block_sizes(block_sizes):
            # Convert block_sizes to axis
            assert self.axis is None, "Axis and block_sizes are both set."
            axis = tuple(
                k if k >= 0 else k + x.dim() for k in block_sizes.keys() if isinstance(k, int)
            )
            self.axis = tuple(i for i in range(x.dim()) if i not in axis) or None

            # remove block_sizes
            self._block_sizes = None

    def export_amax(self) -> Optional[torch.Tensor]:
        """Export correctly formatted/shaped amax."""
        if self.block_sizes is not None and self.block_sizes.get("type", None) == "dynamic":
            return self.amax

        if self.amax is None:
            return None

        if not hasattr(self, "_amax_shape_for_export"):
            amax = self.amax
        else:
            amax = self.amax.reshape(self._amax_shape_for_export)
        amax[amax == 0] = self.maxbound
        amax = torch.nan_to_num(amax, nan=self.maxbound)
        clamp_min, clamp_max = torch.finfo(amax.dtype).tiny, torch.finfo(amax.dtype).max
        amax = amax.clamp(min=clamp_min, max=clamp_max)

        self._validate_amax(amax)

        if self.block_sizes is None:
            # tensorrt_llm assumes the scaling_factor dim >= 1 for per-tensor.
            if self.axis is None:
                amax = amax.unsqueeze(0)

            # If single-axis quantization, squeeze amax
            elif isinstance(self.axis, int) or (
                isinstance(self.axis, (list, tuple)) and len(self.axis) == 1
            ):
                amax = amax.squeeze()

        return amax

    def forward(self, inputs):
        """Apply tensor_quant function to inputs.

        Args:
            inputs: A Tensor of type float32/float16/bfloat16.

        Returns:
            outputs: A Tensor of type output_dtype
        """
        if isinstance(inputs, BaseQuantizedTensor):
            assert self._dequantize, "No dequantization stats in the tensor quantizer."
            return self._quant_forward(inputs)

        # Early return if nothing is collected during the forward (e.g. MoE)
        # len(inputs) will break the dynamic shape for torch_export
        if not is_torch_export_mode() and len(inputs) == 0:
            return inputs

        # Activation scaling for smoothquant
        if self.pre_quant_scale is not None:
            inputs = inputs * self.pre_quant_scale

        # Rotating the input
        if self._rotate:
            inputs = normalized_hadamard_transform(inputs)

        if self._disabled:
            # if quantizer is disabled, we still need to track the input dtype for saving the model
            # TODO: This is a temporary solution and needs to be removed once megatron supports
            # non-homogeneous layers
            self._input_dtype = inputs.dtype if hasattr(inputs, "dtype") else None
            return inputs

        # GLOBALS could break TorchDynamo for some Pytorch versions (i.e., 2.3.0)
        if not is_torch_export_mode():
            if GLOBALS.in_onnx_export:
                self._check_onnx_readiness(inputs)

        if self.block_sizes is not None and self._fake_quant:
            # To support the new block_sizes representation for per-channel quantization,
            # convert the dim dict in block_sizes to axis.
            # The axis attribute is still preserved for backward compatibility.
            self._block_sizes_to_axis(inputs)

        if (
            self.block_sizes is not None
            and not self.block_sizes.get("type", None) == "dynamic"
            and self._fake_quant
        ):
            # Tensor reshaping is required for static block quantization
            # Tensor shapes are handled seperately by the quantization kernels for dynamic block quantization
            self._setup_for_blockquant(inputs)
            inputs = self._process_for_blockquant(inputs)

        outputs = inputs

        block_size = None
        if self._if_calib and not self._dynamic:
            if self._calibrator is None:
                raise RuntimeError("Calibrator was not created.")
            # Shape is only known when it sees the first tensor
            if self.block_sizes is not None and self.block_sizes.get("type", None) == "dynamic":
                block_size = self.block_sizes.get(-1, None) or self.block_sizes.get(
                    inputs.dim() - 1, None
                )
                assert block_size is not None, "block size for dynamic quantization not found."

            # Collect calibration data for bias
            if self.bias_calibrator is not None and self.bias_type == "static":
                self.bias_calibrator.collect_calib_bias(inputs)
            self._calibrator.collect(inputs)

        if self._if_quant:
            outputs = self._quant_forward(inputs)

        if (
            self.block_sizes is not None
            and not self.block_sizes.get("type", None) == "dynamic"
            and self._fake_quant
        ):
            outputs = self._reset_to_original_shape(outputs)

        return outputs

    def _short_amax(self, fmt=".4f"):
        """Short description of amax.

        Returns:
            'dynamic': if _amax is not registered
            'amax': if _amax is per-tensor
            '[min, max](size)': if _amax is per-channel
        """
        if self.is_mx_format:
            return "None"
        if not hasattr(self, "_amax"):
            return "dynamic"
        if self._amax is None:
            return "None"
        if self._amax.is_meta:
            return "meta"
        if self._amax.numel() == 1:
            return f"{self._amax.item():{fmt}}"
        return (
            f"[{self._amax.min().item():{fmt}},"
            f" {self._amax.max().item():{fmt}}]({self._amax.numel()})"
        )

    def extra_repr(self):
        """Set the extra information about this module."""
        if self._disabled:
            return "disabled"
        s = f"{'unsigned ' if self._unsigned else ''}{self._num_bits} bit"
        s += " narrow" if (self._narrow_range) else ""
        s += " fake" if (self._fake_quant) else ""
        if self.block_sizes is not None:
            s += f" block_sizes={self._block_sizes},"
        else:
            s += f" axis={self._axis}" if self._axis is not None else " per-tensor"
        s += f" amax={self._short_amax()}"
        s += " pre_quant_scale" if self.pre_quant_scale is not None else ""
        s += " rotated" if self._rotate else ""
        s += (
            f" calibrator={self._calibrator.__class__.__name__}"
            if (self._calibrator is not None)
            else ""
        )
        if self._bias:
            s += f" bias={self._bias}"

        s += " quant" if (self._if_quant) else ""
        s += " calib" if (self._if_calib) else ""
        return s

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Overloaded module function.

        Adds warnings during state_dict loading.
        A workaround is implemented for loading amax from checkpoint and only supports CUDA.

        Args:
            state_dict: A dict containing the state of the top level module
            prefix: A string that prefixes all of this modules state in state_dict, e.g. 'model.conv1.'
        """
        dst_has_amax = "_amax" in self._buffers
        src_has_amax = prefix + "_amax" in state_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not src_has_amax and dst_has_amax:
            warnings.warn(f"{prefix[:-1]}: No amax in state_dict.")
        elif src_has_amax and not dst_has_amax:
            warnings.warn(
                f"{prefix[:-1]}: No '_amax' buffer to load amax into."
                " '_amax` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer("_amax", state_dict[prefix + "_amax"].clone().detach().to(device))

        dst_has_pre_quant_scale = "_pre_quant_scale" in self._buffers
        src_has_pre_quant_scale = prefix + "_pre_quant_scale" in state_dict

        if not src_has_pre_quant_scale and dst_has_pre_quant_scale:
            warnings.warn(f"{prefix[:-1]}: No pre_quant_scale in state_dict.")
        elif src_has_pre_quant_scale and not dst_has_pre_quant_scale:
            warnings.warn(
                f"{prefix[:-1]}: No '_pre_quant_scale' buffer to load pre_quant_scale into."
                " '_pre_quant_scale` will be created as WAR for now. "
                "This behavior will change in future."
            )
            self.register_buffer(
                "_pre_quant_scale",
                state_dict[prefix + "_pre_quant_scale"].clone().detach().to(device),
            )

        super(TensorQuantizer, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_skip_properties_for_modelopt_state(self):
        return {
            "_calibrator",
            "_bias_calibrator",
            "_original_shape",
            "_block_reshape_size",
            "_padding",
        }

    def _get_properties_for_modelopt_state(self):
        return (
            self.__dict__.keys()
            - nn.Module().__dict__.keys()
            - self._get_skip_properties_for_modelopt_state()
        )

    def get_modelopt_state(self, properties_only: bool = False) -> dict[str, Any]:
        """Get meta state to be saved in checkpoint.

        If `properties_only` is True, only the quantizer properties such as `num_bits`, `axis` etc are included.
        For restoring the quantizer fully, use `properties_only=False`.
        """
        modelopt_state = {}
        for k in self._get_properties_for_modelopt_state():
            modelopt_state[k] = getattr(self, k)

        if properties_only:
            return modelopt_state

        if hasattr(self, "_amax"):
            modelopt_state["_has_amax"] = True

        if hasattr(self, "_pre_quant_scale"):
            modelopt_state["_has_pre_quant_scale"] = True

        return modelopt_state

    def set_from_modelopt_state(self, modelopt_state, prefix=""):
        """Set meta state from checkpoint."""
        # Set all properties except the skip properties; this is done for backward compatibility
        for key in modelopt_state.keys() - self._get_skip_properties_for_modelopt_state():
            setattr(self, key, modelopt_state[key])

        # Set the calibrator properties
        # TODO: This might not be sufficient for the custom calibrators - however there is no use-case for it yet
        for key in ["_num_bits", "_axis", "_unsigned"]:
            setattr(self._calibrator, key, getattr(self, key))

        # Create a temporary variable to indicate if the quantizer had amax in the checkpoint
        if "_has_amax" in modelopt_state or "_amax" in modelopt_state:
            self._has_amax = modelopt_state.get("_has_amax", "_amax" in modelopt_state)

        # Create a temporary variable to indicate if the quantizer had pre_quant_scale in the checkpoint
        if "_has_pre_quant_scale" in modelopt_state or "_pre_quant_scale" in modelopt_state:
            self._has_pre_quant_scale = modelopt_state.get(
                "_has_pre_quant_scale", "_pre_quant_scale" in modelopt_state
            )

    def clean_up_after_set_from_modelopt_state(self, prefix=""):
        """Clean up temporary variables created during set_from_modelopt_state."""
        warning_msg = (
            f"Could not initialize the quantizer states for {prefix}. The quantizer"
            " states after `load_state_dict` could be in the wrong device. Please move"
            " the modules to the correct device after loading the state dict."
        )

        if hasattr(self, "_has_amax"):
            if self._has_amax and self.amax is None:
                warnings.warn(warning_msg, UserWarning)
            delattr(self, "_has_amax")

        if hasattr(self, "_has_pre_quant_scale"):
            if self._has_pre_quant_scale and self.pre_quant_scale is None:
                warnings.warn(warning_msg, UserWarning)
            delattr(self, "_has_pre_quant_scale")

    def sync_amax_across_distributed_group(self, parallel_group: DistributedProcessGroup):
        """Synchronize the amax across all ranks in the given group."""
        if parallel_group.is_initialized() and self.amax is not None:
            try:
                dist.all_reduce(self.amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            except RuntimeError as e:
                # This error happens if the distributed backend is using GPU and
                # the tensor is not on GPU (or vice versa).
                warnings.warn(
                    (
                        f"Failed to synchronize amax: {e}, probably because the tensor is on a device which is not"
                        "supported by the current distributed backend. This warning can be ignored"
                        "if happening during modelopt restore."
                    )
                )


class SequentialQuantizer(nn.Sequential):
    """A sequential container for  :class:`TensorQuantizer` modules.

    This modules is used to quantize a tensor in multiple formats sequentially. It takes as input
    :class:`TensorQuantizer` modules and containerize them similar to :class:`torch.nn.Sequential`.

    Args:
        quantizers (TensorQuantizer): :class:`TensorQuantizer` modules to be added to the container.

    """

    def __init__(self, *quantizers: TensorQuantizer):  # noqa: N803
        """Initialize SequentialQuantizer module."""
        assert not any(not isinstance(q, TensorQuantizer) for q in quantizers), (
            "All quantizers must be a TensorQuantizer."
        )
        super().__init__(*quantizers)

    @property
    def fake_quant(self):
        """Return True if only fake quantization is used."""
        return all(q.fake_quant for q in self)

    def get_modelopt_state(self) -> dict[str, Any]:
        """Get meta state to be saved in checkpoint."""
        return {"num_quantizers": len(self), "is_sequential_quantizer": True}

    def disable(self):
        """Disable the quantizer modules."""
        for quantizer in self:
            quantizer.disable()

    def reset_amax(self):
        """Reset amax of the quantizers."""
        for quantizer in self:
            quantizer.reset_amax()

    def set_from_attribute_config(
        self,
        attributes: Union[
            list[Union[dict[str, Any], QuantizerAttributeConfig]],
            Union[dict[str, Any], QuantizerAttributeConfig],
        ],
    ):
        """Set the attributes of contained quantizers from a list of attribute_dicts."""
        if not isinstance(attributes, (list, tuple)):
            assert isinstance(attributes, (dict, QuantizerAttributeConfig)), (
                "attributes must be a list or a dict."
            )
            attributes = [attributes] * len(self)

        for attribute, quantizer in zip(attributes, self):
            quantizer.set_from_attribute_config(attribute)

    @staticmethod
    @contextlib.contextmanager
    def replace_sequential_quantizer_with_single_quantizer(model, indx: int = 0):
        """Replace instances of :class:`SequentialQuantizer` in the model with single quantizers.

        The quantizer indexed by ``indx`` from the sequential quantizer is used to replace it.
        This method is useful for individually calibrating the quantizers in a sequential quantizer.
        """
        original_sequential_quantizers: dict[nn.Module, list] = {}
        for name, module in list(model.named_modules()):
            if isinstance(module, SequentialQuantizer):
                assert len(module) > indx
                parent_module = model.get_submodule(name.rpartition(".")[0])
                if parent_module not in original_sequential_quantizers:
                    original_sequential_quantizers[parent_module] = []
                original_sequential_quantizers[parent_module].append(
                    (name.rpartition(".")[-1], module)
                )
                setattr(parent_module, name.rpartition(".")[-1], module[indx])

        yield

        for parent_module, sequential_quantizers_list in original_sequential_quantizers.items():
            for name, sequential_quantizer in sequential_quantizers_list:
                setattr(parent_module, name, sequential_quantizer)

    @staticmethod
    def tensor_quantizer_iterator(quantizers):
        """Iterator for the quantizers in the container (but yield itself if its a TensorQuantizer)."""
        if quantizers is None:
            return
        if isinstance(quantizers, TensorQuantizer):
            yield quantizers
        elif isinstance(quantizers, SequentialQuantizer):
            for quantizer in quantizers:
                yield quantizer
        else:
            raise ValueError("Invalid quantizer type.")
