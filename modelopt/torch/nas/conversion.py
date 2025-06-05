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

"""Main APIs+entrypoints for model pruning."""

from torch import nn

from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.mode import ModeLike, _ModeRegistryCls
from modelopt.torch.utils import ModelLike, unwrap_model

__all__ = ["convert", "export"]

NASModeRegistry = _ModeRegistryCls("nas")


def convert(model: ModelLike, mode: ModeLike) -> nn.Module:
    """Convert a regular PyTorch model into a model that supports design space optimization.

    Args:
        model: A model-like object. Can be an nn.Module, a model class type, or a tuple.
            Tuple must be of the form ``(model_cls,)`` or ``(model_cls, args)`` or ``(model_cls, args, kwargs)``.
            Model will be initialized as ``model_cls(*args, **kwargs)``.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :class:`"autonas"<modelopt.torch.nas.autonas.AutoNASModeDescriptor>`: The ``model`` will
                be converted into a search space and set up to automatically perform operations
                required for AutoNAS-based model training, evaluation, and search. The mode's config
                is described in :class:`AutoNASConfig<modelopt.torch.nas.autonas.AutoNASConfig>`.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration. If not provided, then default
            configuration would be used.

    Returns:
        A converted model with the original weights preserved that can be used for model
        optimization.

    .. note::

        Note that model wrappers (such as DataParallel/DistributedDataParallel) are `not` supported
        during the convert process. Please wrap the model after the convert process.

    .. note::
        ``convert()`` relies on `monkey patching <https://en.wikipedia.org/wiki/Monkey_patch>`_ to
        augment the ``forward()``, ``eval()``, and ``train()`` methods of ``model`` as well as
        augment individual modules to make them dynamic. This renders the conversion incompatible
        with other monkey patches to those methods and modules! Note that ``convert()`` is still
        fully compatible with inheritance.

    .. note::

        #. Configs can be customized for individual layers using
           `glob expressions <https://docs.python.org/3/library/fnmatch.html#fnmatch.fnmatch>`_ on
           qualified submodule names, e.g., as shown for ``nn.Conv2d`` in the above example.

        #. Keys in the config that appear earlier in a dict have lower priority, e.g.,
           ``backbone.stages.1.0.spatial_conv`` will have ``out_channels_ratio``
           ``[0.334, 0.5, 0.667, 1.0]``, not ``[1.0]``.

        #. Config entries without layer qualifiers are also supported, e.g., as shown for
           ``nn.Sequential`` in the above example.

        #. Mixed usage of configurations with and without layer qualifiers is supported for
           *different* layers, e.g., as shown for ``nn.Conv2d`` and ``nn.Sequential`` in the above
           example. For a specific layer type, only configurations with *or* without layer
           qualifiers are supported.

        #. Use ``*`` as a wildcard matching any layer.
    """
    # apply mode and handle model-like object with wrapper
    return apply_mode(model, mode, registry=NASModeRegistry)


def export(model: nn.Module, strict: bool = True, calib: bool = False) -> nn.Module:
    """Export a pruned subnet to a regular model.

    Args:
        model: The pruned subnet to be exported.
        strict: Raise an error when the config does not contain all necessary keys.
        calib: Whether to calibrate the subnet to be exported.

    Returns:
        The current active subnet in regular PyTorch model format.

    .. note::

        if model is a wrapper such as DistributedDataParallel, it will be unwrapped, e.g.,
        ``model.module`` will be returned.
    """
    # unwrap a DP/DDP model
    model = unwrap_model(
        model,
        warn=True,
        msg=(
            f"Unwrapping a {type(model).__name__} model for export! Note that the export is"
            " in-place and the model wrapper should be re-created after export since the wrapper"
            " might not support changing parameters after initialization."
        ),
    )

    # apply export mode and return model
    config = {"strict": strict, "calib": calib}
    return apply_mode(model, [("export", config)], registry=NASModeRegistry)
