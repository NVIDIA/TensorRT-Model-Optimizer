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

"""Basic tensor quantization functions."""

import warnings

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.onnx import symbolic_helper

import modelopt.torch.quantization.triton as triton_kernel

from .config import QuantizerAttributeConfig
from .extensions import get_cuda_ext, get_cuda_ext_fp8, get_cuda_ext_mx

mx_format_map = {
    (4, 3): "E4M3",
    (5, 2): "E5M2",
    (3, 2): "E3M2",
    (2, 3): "E2M3",
    8: "INT8",
    (8, 0): "E8M0",
    (2, 1): "E2M1",
    (1, 2): "E1M2",
    (0, 3): "E0M3",
    (3, 0): "E3M0",
}

DISABLE_TRITON_KERNEL = False


def scaled_e4m3_impl(
    inputs: torch.Tensor,  # TODO: check support for multiple inputs
    amax: torch.Tensor,
    disable_fused_kernel=True,
) -> torch.Tensor:
    """Implementation of fake quantizing input to FP8.

    Args:
        inputs: Torch tensor.
        amax: Absolute max range of the input tensor.

    Returns:
        Input tensors faked quantized to FP8.
    """
    cuda_ext_fp8 = get_cuda_ext_fp8(raise_if_failed=True)

    def is_fusable():
        # ignore no scaling and shape([]) cases
        if amax is None or len(amax.shape) == 0:
            return False
        else:
            # can't have amax.shape = [1, 1, 4, 1] and the like
            amax_last_dim_only = amax.numel() == amax.shape[-1]
            # must be cuda
            all_cuda = inputs.is_cuda and amax.is_cuda

            # also check explicit disable.
            return amax_last_dim_only and all_cuda and (not disable_fused_kernel)

    with torch.cuda.device(
        None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
    ):
        # differentiate between fused & unfused cases
        if is_fusable():
            zero_threshold = 1.0 / (1 << 24)
            outputs = cuda_ext_fp8.fused_fake_e4m3fy(inputs, amax.float(), zero_threshold)
        else:
            zero_mask = inputs.abs() < 1.0 / (1 << 24)

            if amax is None:
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs)
            else:
                scale = 448.0 / amax
                outputs = cuda_ext_fp8.fake_e4m3fy(inputs * scale) / scale

            # Zero out values that are tiny.
            # Tiny values could lead to tiny amax and then large scale which cause overflow/saturation
            # and won't go back to normal value after dividing by scale. The right behavior is to mark them
            # as zero which also get rid of inf/nan
            outputs[zero_mask] = 0.0

        return outputs


def fake_quant_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits=8,
    unsigned=False,
    narrow_range=True,
):
    """Implementation of fake quantizing input according to number of bits."""
    cuda_ext = get_cuda_ext()

    with torch.cuda.device(
        None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
    ):
        if amax.numel() == 1:
            outputs = cuda_ext.fake_tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        else:
            axis = amax.shape.index(amax.numel())
            outputs = cuda_ext.fake_tensor_quant_with_axis(
                inputs, amax.squeeze(), axis, num_bits, unsigned, narrow_range
            )
        return outputs


def _quantize_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
):
    if num_bits == 8 and exponent_bits == 4:
        return scaled_e4m3_impl(inputs=inputs, amax=amax)
    elif isinstance(num_bits, int):
        return fake_quant_impl(
            inputs=inputs,
            amax=amax,
            num_bits=num_bits,
            unsigned=unsigned,
            narrow_range=narrow_range,
        )
    else:
        raise ValueError(
            f"Invalid combination of (num_bits, exponent_bits): ({num_bits}, {exponent_bits})."
        )


def _quantize_impl_abstract(
    input: torch.Tensor,
    amax: torch.Tensor,
    num_bits: int = 8,
    exponent_bits: int = 0,
    unsigned: bool = False,
    narrow_range: bool = True,
) -> torch.Tensor:
    """Register an abstract implementation for quantizing tensor.

    This abstract function returns an empty tensor with the same shape and dtype.
    """
    output = torch.empty_like(input)

    return output


# Argument types: Tensor, int, NoneType, int, int, int, int,
def _dynamic_block_quantize_impl(
    inputs: torch.Tensor,
    block_size: int,
    amax: torch.Tensor,
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
):
    scale_bits = (scale_exponent_bits, scale_num_bits - scale_exponent_bits - 1)
    if exponent_bits != 0:
        num_bits = (exponent_bits, num_bits - exponent_bits - 1)
    if num_bits in mx_format_map:
        assert scale_bits in mx_format_map, f"Scale bits should be in {mx_format_map.keys()}"
        if scale_bits != (8, 0):
            assert amax.is_cuda, "amax must be a CUDA tensor for dynamic block quantization."
            if amax.numel() != 1:
                amax = amax.amax()
        with torch.cuda.device(
            None if inputs.device.index == torch.cuda.current_device() else inputs.device.index
        ):
            if (
                num_bits == (2, 1)  # type: ignore[comparison-overlap]
                and scale_bits == (4, 3)
                and triton_kernel.IS_AVAILABLE
                and not DISABLE_TRITON_KERNEL
                and amax is not None
            ):
                return triton_kernel.fp4_fake_quant_block(inputs, amax.item())
            cuda_ext_mx = get_cuda_ext_mx(raise_if_failed=True)
            return cuda_ext_mx.fused_amax_convert(
                inputs,
                block_size,
                getattr(cuda_ext_mx.Types, mx_format_map[num_bits]),
                getattr(cuda_ext_mx.Types, mx_format_map[scale_bits]),
                amax,
            )
    else:
        raise NotImplementedError(
            f"Unsupported num_bits: {num_bits}, scale_bits: {scale_bits} for dynamic block quantization."
        )


def _dynamic_block_quantize_impl_abstract(
    inputs: torch.Tensor,
    block_size: int,
    amax: torch.Tensor,
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
):
    """Register an abstract implementation for dynamic block quantization.

    This abstract function returns an empty tensor with the same shape and dtype.
    """
    output = torch.empty_like(inputs)

    return output


quantize_op = _quantize_impl
dynamic_block_quantize_op = _dynamic_block_quantize_impl
# Define custom operators via torch.library if supported:
# 1. quantize_op: Applies static quantization to the input tensor using a specified amax.
# 2. dynamic_block_quantize_op: Performs blockwise double quantization with dynamically
#    determined scales, governed by the given quantization format (scale_num_bits, scale_exponent_bits).
try:
    torch.library.define(
        "tensorrt::quantize_op",
        "(Tensor input, Tensor amax, int num_bits, int exponent_bits, "
        "bool unsigned, bool narrow_range) -> Tensor",
    )
    torch.library.define(
        "tensorrt::dynamic_block_quantize_op",
        "(Tensor input, int block_size, Tensor amax, int num_bits, int exponent_bits, "
        "int scale_num_bits, int scale_exponent_bits) -> Tensor",
    )
    torch.library.define(
        "tensorrt::dynamic_block_quantize_op.overload",
        "(Tensor input, int block_size, None amax, int num_bits, int exponent_bits, "
        "int scale_num_bits, int scale_exponent_bits) -> Tensor",
    )

    # Implement the None amax case
    def _dynamic_block_quantize_impl_none_amax(
        inputs: torch.Tensor,
        block_size: int,
        amax: None,
        num_bits: int,
        exponent_bits: int,
        scale_num_bits: int,
        scale_exponent_bits: int,
    ):
        return torch.empty_like(inputs)

    # Register the implementation for both CPU and CUDA
    torch.library.impl("tensorrt::quantize_op", ["cpu", "cuda"])(_quantize_impl)
    torch.library.impl("tensorrt::dynamic_block_quantize_op", ["cpu", "cuda"])(
        _dynamic_block_quantize_impl
    )
    torch.library.impl("tensorrt::dynamic_block_quantize_op.overload", ["cpu", "cuda"])(
        _dynamic_block_quantize_impl_none_amax
    )

    # Register the fake implementation
    torch.library.register_fake("tensorrt::quantize_op")(_quantize_impl_abstract)
    torch.library.register_fake("tensorrt::dynamic_block_quantize_op")(
        _dynamic_block_quantize_impl_abstract
    )
    torch.library.register_fake("tensorrt::dynamic_block_quantize_op.overload")(
        _dynamic_block_quantize_impl_abstract
    )

    quantize_op = torch.ops.tensorrt.quantize_op
    dynamic_block_quantize_op = torch.ops.tensorrt.dynamic_block_quantize_op
except (AttributeError, RuntimeError) as e:
    # torch.library is an experiemental feature, the function signatures may change overtime.
    warnings.warn(
        "Unable to register operators with torch.library. Exporting quantized models with"
        f" torch.export will not be supported.\n{e}"
    )

# Predefined descriptors
QUANT_DESC_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8)
QUANT_DESC_UNSIGNED_8BIT_PER_TENSOR = QuantizerAttributeConfig(num_bits=8, unsigned=True)
QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))
QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL = QuantizerAttributeConfig(num_bits=8, axis=(0))


@torch.jit.script
def _fake_tensor_quant_backward(inputs, amax: torch.Tensor | None, grad_outputs):
    # Skip clip for MX formats
    if amax is None:
        return grad_outputs

    zero = grad_outputs.new_zeros(1)
    grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
    return grad_inputs


class FakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction use CUDA extension."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "t", "i", "b", "b", "s")
    def symbolic(
        g,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_int8

        return export_int8(
            g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
    ):
        """Forward method."""
        if bias is not None:
            inputs = inputs - bias

        ctx.save_for_backward(inputs, amax)

        def legacy_quant_func():
            # The LegacyFakeTensorQuantFunction support cpu and amax with any shape that can be broadcasted to inputs.
            outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
            return outputs / scale.to(inputs.dtype)

        if not inputs.is_cuda:
            outputs = legacy_quant_func()
        else:
            try:
                outputs = quantize_op(
                    inputs,
                    amax,
                    num_bits=num_bits,
                    exponent_bits=0,
                    unsigned=unsigned,
                    narrow_range=narrow_range,
                )
            except (AttributeError, ValueError):
                # AttributeError: cuda_ext is not imported, possibly due to CPU only installation
                # ValueError: cuda_ext is installed, but trying to perform multidimensional quantization (amax dim > 1)
                outputs = legacy_quant_func()

        if bias is not None:
            outputs = outputs + bias

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        inputs, amax = ctx.saved_tensors
        return (
            _fake_tensor_quant_backward(inputs, amax, grad_outputs),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ScaledE4M3Function(Function):
    """E4M3fy input with scale."""

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "t", "i", "i", "s")
    def symbolic(g, inputs, amax=None, bias=None, E=4, M=3, trt_high_precision_dtype=None):  # noqa: N803
        """ONNX symbolic function."""
        from .export_onnx import export_fp8

        return export_fp8(g, inputs, amax, trt_high_precision_dtype)

    @staticmethod
    # Default values could cause errors from TorchDynamo during torch.export
    def forward(ctx, inputs, amax, bias, E, M, trt_high_precision_dtype=None):  # noqa: N803
        """Forward method."""
        if E != 4 or M != 3:
            raise NotImplementedError("Only support E=4 & M=3 for now.")

        if bias is not None:
            inputs = inputs - bias

        ctx.save_for_backward(inputs)
        ctx.amax = amax

        outputs = quantize_op(
            inputs, amax, num_bits=8, exponent_bits=4, unsigned=False, narrow_range=False
        )

        if bias is not None:
            outputs = outputs + bias

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        (inputs,) = ctx.saved_tensors
        amax = torch.tensor(
            ctx.amax if ctx.amax is not None else 448.0, dtype=torch.float32, device=inputs.device
        )
        grad_inputs = _fake_tensor_quant_backward(inputs, amax, grad_outputs)
        return grad_inputs, None, None, None, None, None


def _dynamic_block_quantize_forward(
    ctx,
    inputs,
    block_size,
    amax,
    num_bits,
    scale_bits,
    trt_high_precision_dtype=None,
    onnx_quantizer_type="dynamic",
):
    """Forward method."""
    ctx.save_for_backward(inputs, amax)
    if isinstance(num_bits, int):
        # special case for INT dynamic block quantization, e.g. MXINT8
        exponent_bits = 0
    else:
        assert isinstance(num_bits, tuple) and len(num_bits) == 2
        exponent_bits = num_bits[0]
        num_bits = num_bits[0] + num_bits[1] + 1
    assert isinstance(scale_bits, tuple) and len(scale_bits) == 2
    scale_exponent_bits = scale_bits[0]
    scale_num_bits = scale_bits[0] + scale_bits[1] + 1
    outputs = dynamic_block_quantize_op(
        inputs, block_size, amax, num_bits, exponent_bits, scale_num_bits, scale_exponent_bits
    )
    return outputs


class DynamicBlockQuantizationFunction(Function):
    """Dynamic block quantization functional."""

    @staticmethod
    @symbolic_helper.parse_args("v", "i", "v", "t", "is", "is", "s", "s")
    def symbolic(
        g,
        inputs,
        block_size,
        amax,
        bias,
        num_bits,
        scale_bits,
        trt_high_precision_dtype=None,
        onnx_quantizer_type="dynamic",
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_fp4, export_mxfp8

        if num_bits == (2, 1) and scale_bits == (4, 3):
            return export_fp4(
                g,
                inputs,
                block_size,
                amax,
                num_bits,
                trt_high_precision_dtype,
                onnx_quantizer_type,
            )
        if num_bits == (4, 3) and scale_bits == (8, 0):
            return export_mxfp8(
                g,
                inputs,
                onnx_quantizer_type,
                block_size,
            )
        raise NotImplementedError(
            f"Unsupported num_bits: {num_bits} and scale_bits: {scale_bits} for ONNX export."
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        block_size,
        amax,
        bias,
        num_bits,
        scale_bits,
        trt_high_precision_dtype=None,
        onnx_quantizer_type="dynamic",
    ):
        """Forward method."""
        return _dynamic_block_quantize_forward(
            ctx,
            inputs,
            block_size,
            amax,
            num_bits,
            scale_bits,
            trt_high_precision_dtype,
            onnx_quantizer_type,
        )

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        # Dynamic block quantization does not clip the tensors (i.e in effect) in the forward loop
        # as the block quantization amax is always the block amax. So we dont need to clip the gradients for STE
        inputs, amax = ctx.saved_tensors
        return (
            _fake_tensor_quant_backward(inputs, amax, grad_outputs),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class StaticBlockQuantizationFunction(FakeTensorQuantFunction):
    """Static block quantization functional."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        bias,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
        block_size=None,
    ):
        """Forward method."""

        def legacy_quant_func():
            # The LegacyFakeTensorQuantFunction support cpu and amax with any shape that can be broadcasted to inputs.
            outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
            return outputs / scale.to(inputs.dtype)

        def _pad_to_block_size(x, block_size):
            """Pad the input tensor to a multiple of block_size."""
            pad_size = (block_size - x.shape[-1] % block_size, 0)
            return F.pad(inputs, pad_size, "constant", 0)

        # reshape the input tensor to [..., block_size]
        original_shape = inputs.shape
        if block_size is not None and inputs.shape[-1] != block_size:
            if inputs.shape[-1] % block_size != 0:
                inputs = _pad_to_block_size(inputs, block_size)
            inputs = inputs.reshape(-1, block_size)

        if bias is not None:
            inputs = inputs - bias

        ctx.save_for_backward(inputs, amax)

        if not inputs.is_cuda:
            outputs = legacy_quant_func()
        else:
            try:
                outputs = quantize_op(
                    inputs,
                    amax,
                    num_bits=num_bits,
                    exponent_bits=0,
                    unsigned=unsigned,
                    narrow_range=narrow_range,
                )
            except (AttributeError, ValueError):
                # AttributeError: cuda_ext is not imported, possibly due to CPU only installation
                # ValueError: cuda_ext is installed, but trying to perform multidimensional quantization (amax dim > 1)
                outputs = legacy_quant_func()

        if bias is not None:
            outputs = outputs + bias

        # restore the original shape
        if block_size is not None and original_shape != outputs.shape:
            outputs = outputs.reshape(original_shape)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping."""
        inputs, amax = ctx.saved_tensors
        return (
            _fake_tensor_quant_backward(inputs, amax, grad_outputs),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TensorQuantFunction(Function):
    """A universal tensor quantization function.

    Take an input tensor, output an quantized tensor. The granularity of scale can be interpreted from the
    shape of amax.
    output_dtype indicates whether the quantized value will be stored in integer or float. The reason we want to store
    it in float is the pytorch function takes the quantized value may not accept integer input, e.g. Conv2D.

    It uses 2^num_bits -1 values instead of 2^num_bits. e.g., for num_bits=8, it uses [-127, 127] instead of [-128, 127]
    """

    @staticmethod
    @symbolic_helper.parse_args("v", "t", "t", "i", "b", "b", "s")
    def symbolic(
        g,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
    ):
        """ONNX symbolic function."""
        from .export_onnx import export_int8

        return export_int8(
            g, inputs, amax, num_bits, unsigned, narrow_range, trt_high_precision_dtype
        )

    @staticmethod
    def forward(
        ctx,
        inputs,
        amax,
        bias=None,
        num_bits=8,
        unsigned=False,
        narrow_range=True,
        trt_high_precision_dtype=None,
    ):
        """Forward method.

        Follow tensorflow convention, max value is passed in and used to decide scale, instead of inputing scale
        directly. Though inputing scale directly may be more natural to use.

        Args:
            ctx: A Context object to store tensors for backward.
            inputs: A Tensor of type float32.
            amax: A Tensor of type float32. Inputs will be quantized within range [-amax, amax]
                amax will be broadcasted to inputs tensor.
            num_bits: A integer used to calculate scaling factor, scale = (2^(num_bits-1) - 1) / max
                Effectively, it indicates how many integer bits is used to represent the value. Default 8.
            output_dtype: A type of Tensor. torch.int32 or torch.float32.
            unsigned: A boolean. Use unsigned integer range. E.g. [0, 255] for num_bits=8. Default False.
            narrow_range: A boolean. Use symmetric integer range for signed quantization
                E.g. [-127,127] instead of [-128,127] for num_bits=8. Default True.

        Returns:
            outputs: A Tensor of type output_dtype.
            scale: A Tensor of type float32. outputs / scale will dequantize outputs tensor.

        Raises:
            ValueError:
        """
        if bias is not None:
            inputs = inputs - bias

        ctx.save_for_backward(inputs, amax)

        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)
        # Check if scale overflows FP16
        if outputs.dtype == torch.half and scale.max() > 65504:
            raise ValueError(f"scale is too large for FP16 with amax={amax}")

        if bias is not None:
            outputs = outputs + bias

        return outputs, scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs, grad_scale):
        """Implements straight through estimation with clipping.

        For -amax <= input <= amax the gradient passes straight through, otherwise the gradient is zero.

        Args:
            ctx: A Context object with saved tensors from forward.
            grad_outputs: A tensor of gradient of outputs.
            grad_scale: A tensor of gradient of scale.

        Returns:
            grad_inputs: A tensor of gradient.
        """
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)  # create a zero tensor with the same type and device
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None, None, None


class LegacyFakeTensorQuantFunction(Function):
    """Fake version of TensorQuantFunction.

    See comments of TensorQuantFunction, arguments are the same.
    """

    @staticmethod
    def forward(ctx, inputs, amax, bias, num_bits=8, unsigned=False, narrow_range=True):
        """Forward method."""
        if bias is not None:
            inputs = inputs - bias

        ctx.save_for_backward(inputs, amax)

        outputs, scale = _tensor_quant(inputs, amax, num_bits, unsigned, narrow_range)

        if bias is not None:
            outputs = outputs + bias

        return outputs / scale.to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation."""
        inputs, amax = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(inputs.abs() <= amax, grad_outputs, zero)
        return grad_inputs, None, None, None, None, None


def _tensor_quant(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
    """Shared function body between TensorQuantFunction and FakeTensorQuantFunction."""
    # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
    if unsigned and inputs.min() < 0.0:
        raise TypeError("Negative values encountered in unsigned quantization.")

    # Computation can be done in FP32 to prevent potential over flow.
    input_dtype = inputs.dtype
    if inputs.dtype == torch.half:
        inputs = inputs.float()
    if amax.dtype == torch.half:
        amax = amax.float()

    min_amax = amax.min()
    if min_amax < 0:
        raise ValueError("Negative values in amax")

    max_bound = torch.tensor((2.0 ** (num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    if unsigned:
        min_bound = 0
    elif narrow_range:
        min_bound = -max_bound
    else:
        min_bound = -max_bound - 1
    scale = max_bound / amax

    epsilon = 1.0 / (1 << 24)
    if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
        zero_amax_mask = amax <= epsilon
        scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

    if min_amax <= epsilon:
        scale[zero_amax_mask] = (
            1.0  # Return 1 makes more sense for values quantized to 0 with amax=0
        )

    if input_dtype == torch.half:
        outputs = outputs.half()

    return outputs, scale


class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization.

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, min_range, max_range, num_bits=8):
        """As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        ctx.save_for_backward(inputs, min_range, max_range)

        step_size = (max_range - min_range) / (2.0**num_bits - 1)

        min_bound = -(2.0 ** (num_bits - 1))
        max_bound = 2.0 ** (num_bits - 1) - 1

        quant_zero = torch.round(min_range / step_size) - min_bound
        quantized = torch.round(inputs / step_size) - quant_zero
        quantized = torch.clamp(quantized, min_bound, max_bound)

        outputs = (quantized + quant_zero) * step_size

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        """Implements straight through estimation with clipping.

        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs.

        Returns:
            grad_inputs: A tensor of gradient
        """
        inputs, min_range, max_range = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where((inputs <= max_range) * (inputs >= min_range), grad_outputs, zero)
        return grad_inputs, None, None, None


tensor_quant = TensorQuantFunction.apply
legacy_fake_tensor_quant = LegacyFakeTensorQuantFunction.apply
fake_tensor_quant = FakeTensorQuantFunction.apply
fake_affine_tensor_quant = FakeAffineTensorQuantFunction.apply
scaled_e4m3 = ScaledE4M3Function.apply
dynamic_block_quant = DynamicBlockQuantizationFunction.apply
static_block_quant = StaticBlockQuantizationFunction.apply
