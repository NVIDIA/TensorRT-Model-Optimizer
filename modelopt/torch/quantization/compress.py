# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for compressing the model weights after quantization."""

__all__ = ["compress"]

import fnmatch
import warnings

import torch.nn as nn

from modelopt.torch.opt import apply_mode
from modelopt.torch.opt.conversion import ModelLikeModule, ModeloptStateManager
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.opt.mode import ConvertReturnType, MetadataDict

from .backends.gemm_registry import enable_real_quant_gemm, is_real_quant_gemm_enabled
from .config import CompressCfgType, CompressConfig
from .conversion import _replace_quant_module, set_quantizer_attribute, update_quantize_metadata
from .nn.modules.quant_linear import RealQuantLinear
from .qtensor import QTensorWrapper, pack_real_quantize_weight
from .utils import is_quantized_linear

RealQuantModuleRegistry = _DMRegistryCls("RealQuant")


def compress_convert(
    model, config: CompressConfig, use_real_quant_gemm: bool = True
) -> ConvertReturnType:
    """Compress entry point."""
    for _, module in model.named_modules():
        if is_quantized_linear(module) and type(module) not in RealQuantModuleRegistry:
            RealQuantModuleRegistry.register({type(module): module.__class__.__name__})(
                RealQuantLinear
            )
    # Convert QuantLinear to RealQuantLinear
    _replace_quant_module(
        model, version=ModeloptStateManager(model).state_version, registry=RealQuantModuleRegistry
    )

    compress_cfg = config.compress
    if "default" in compress_cfg and isinstance(compress_cfg["default"], bool):
        set_quantizer_attribute(
            model, "*weight_quantizer*", {"fake_quant": not compress_cfg["default"]}
        )

    for pattern, to_compress in compress_cfg.items():
        if pattern == "default":
            continue
        if isinstance(to_compress, bool):

            def filter_func(name):
                return fnmatch.fnmatch(name, pattern) and "weight_quantizer" in name

            set_quantizer_attribute(model, filter_func, {"fake_quant": not to_compress})
        else:
            raise ValueError(
                f"Invalid compression configuration: {to_compress}, expected a boolean as value."
            )

    # If real quant quantizer is present, real quantize the weights.
    pack_real_quantize_weight(model)

    def _has_qtensorwrapper(module):
        if hasattr(module, "weight") and isinstance(module.weight, QTensorWrapper):
            return True
        return any(_has_qtensorwrapper(submodule) for _, submodule in module.named_children())

    if _has_qtensorwrapper(model):
        warnings.warn(
            "Real quantization has been applied to the model. This feature is still "
            "experimental, and some functionalities may not be supported. For example, "
            "converting the model back to its original state or saving and restoring "
            "the quantized model may not be available."
        )

    # Turn on real quant gemm after compression
    if use_real_quant_gemm:
        enable_real_quant_gemm(model)

    metadata = {}
    update_compress_metadata(model, config, metadata)

    return model, metadata


def compress_restore(
    model: ModelLikeModule, config: CompressConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore the model from the compressed state."""
    # Compress with dummy weights
    model, _ = compress_convert(
        model,
        config,
        use_real_quant_gemm=metadata.get("use_real_quant_gemm", False),
    )
    return model


def update_compress_metadata(model: nn.Module, config: CompressConfig, metadata: MetadataDict):
    update_quantize_metadata(model, config, metadata)
    metadata["use_real_quant_gemm"] = is_real_quant_gemm_enabled(model)


def compress(model, config: CompressCfgType = None):
    """Compress model weights of quantized model.

    This function compresses weights in layers that have an enabled `weight_quantizer` with
    a supported quantization format. The compression is controlled by a pattern-based configuration.

    Args:
        model: The quantized model to compress.
        config: Dictionary mapping layer patterns to boolean compression flags.
            If ``None``, defaults to ``{"default": True}`` which compresses all supported layers.

            Example configuration::

                {
                    "*.mlp.fc1*": False,  # Skip compression for fc1 layers
                    "default": True,  # Compress all other layers
                }

            Note: Each configuration except "default" is applied sequentially; therefore the later
            configurations will override the previous ones if the same layer is matched.


    Note: This function modifies the input model in-place.
    """
    if config is None:
        config = CompressConfig()
    apply_mode(model, [("real_quantize", config)])


def is_real_quantized(model):
    """Check if the model is real quantized."""
    return any(isinstance(_module, RealQuantLinear) for _module in model.modules())
