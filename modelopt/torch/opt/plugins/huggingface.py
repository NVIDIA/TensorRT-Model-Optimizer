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

"""ModelOpt plugin for enabling automatic save/restore of ModelOpt state for HuggingFace models."""

import functools
import inspect
import os
import threading
import types
from contextlib import contextmanager
from typing import Any

import torch

from modelopt.torch.utils import print_rank_0

from ..conversion import ModeloptStateManager, modelopt_state, restore_from_modelopt_state

__all__ = ["enable_huggingface_checkpointing"]

_PATCHED_LIBRARIES = set()
_MODELOPT_STATE_SAVE_NAMES = {
    "default": "modelopt_state.pth",
}


def _get_modelopt_state_path(obj: Any, model_name_or_path: str) -> str:
    modelopt_state_save_name = _MODELOPT_STATE_SAVE_NAMES["default"]
    for type, value in _MODELOPT_STATE_SAVE_NAMES.items():
        if type == "default":
            continue
        if isinstance(obj, type):  # type: ignore [arg-type]
            modelopt_state_save_name = value
    return os.path.join(model_name_or_path, modelopt_state_save_name)


@contextmanager
def _patch_model_init_for_modelopt(cls, model_path):
    """Patch for `cls.init` method to restore ModelOpt state after `init`."""
    # Note: Keeping original config in local as the package will be shared among threads
    _original__init__ = cls.__init__

    @functools.wraps(_original__init__)
    def new_init_fn(self, *args, **kwargs):
        modelopt_state_path = _get_modelopt_state_path(self, model_path)
        _original__init__(self, *args, **kwargs)
        if os.path.isfile(modelopt_state_path):
            modelopt_state = torch.load(modelopt_state_path, map_location="cpu", weights_only=False)
            if not ModeloptStateManager.has_state_for_mode_type("nas", state=modelopt_state):
                restore_from_modelopt_state(self, modelopt_state)
                print_rank_0(f"Restored ModelOpt state from {modelopt_state_path}")

    cls.__init__ = new_init_fn
    try:
        yield
    finally:
        cls.__init__ = _original__init__


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
    """Patch for `cls.from_pretrained` method to restore ModelOpt state.

    NOTE: This does not work with NAS/Pruned model in Bert example probably due to using accelerate.
    (Error during tracing when restoring since __init__ is called under a context manager that disables weight init).
    Hence, we defer the restoration to the `_load_pretrained_model` method.
    """
    with _patch_model_init_for_modelopt(cls, pretrained_model_name_or_path):
        model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
            pretrained_model_name_or_path, *args, **kwargs
        )
    return model


def _new_from_config(cls, /, config, **kwargs):
    """Patch for `cls.from_config` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(cls, config._name_or_path):
        model = types.MethodType(cls._modelopt_cache["_from_config"].__func__, cls)(
            config, **kwargs
        )
    return model


def _new__load_pretrained_model(cls, *args, **kwargs):
    """Patch for `cls._load_pretrained_model` method to restore ModelOpt state for NAS/Pruned models."""
    # Get the original function signature
    original_func = cls._modelopt_cache["_load_pretrained_model"].__func__
    sig = inspect.signature(original_func)
    param_names = list(sig.parameters.keys())

    # Extract model from args (first positional param)
    model = args[0] if args else kwargs.get("model")

    # Check if pretrained_model_name_or_path was passed as positional or keyword argument
    idx = param_names.index("pretrained_model_name_or_path") - 1  # -1 for cls
    if idx < len(args):
        pretrained_model_name_or_path = args[idx]
    else:
        pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path")

    modelopt_state_path = _get_modelopt_state_path(model, pretrained_model_name_or_path)
    if os.path.isfile(modelopt_state_path) and not ModeloptStateManager.is_converted(model):
        modelopt_state_dict = torch.load(
            modelopt_state_path, map_location="cpu", weights_only=False
        )
        restore_from_modelopt_state(model, modelopt_state_dict)
        print_rank_0(f"Restored ModelOpt state after init from {modelopt_state_path}")

    return types.MethodType(original_func, cls)(*args, **kwargs)


def _new_save_pretrained(self, save_directory, *args, **kwargs):
    self._modelopt_cache["save_pretrained"](self, save_directory, *args, **kwargs)
    if ModeloptStateManager.is_converted(self) and not getattr(
        self, "_disable_modelopt_save", False
    ):
        path = _get_modelopt_state_path(self, save_directory)
        torch.save(modelopt_state(self), path)
        print_rank_0(f"Saved ModelOpt state to {path}")


_DEFAULT_PATCH_METHODS_MAP = {
    "from_pretrained": classmethod(_new_from_pretrained),
    "_load_pretrained_model": classmethod(_new__load_pretrained_model),  # type: ignore [arg-type]
    "save_pretrained": _new_save_pretrained,
    "_from_config": classmethod(_new_from_config),
}


_patch_lock = threading.Lock()


def patch_pretrained_methods(cls, library_name: str, patch_methods_map: dict = None):
    if hasattr(cls, "_modelopt_cache"):
        return

    with _patch_lock:
        # in case multiple threads patch the same library
        if library_name in _PATCHED_LIBRARIES:
            return
        cls._modelopt_cache = {}
        patch_methods_map = patch_methods_map or _DEFAULT_PATCH_METHODS_MAP
        for method_name in patch_methods_map:
            if not hasattr(cls, method_name):
                continue
            cls._modelopt_cache[method_name] = getattr(cls, method_name)
            setattr(cls, method_name, patch_methods_map[method_name])

        _PATCHED_LIBRARIES.add(library_name)


def enable_huggingface_checkpointing():
    """Enables automatic save/restore of ModelOpt state with HuggingFace checkpointing APIs.

    ModelOpt automatically saves `modelopt_state` to `save_directory/modelopt_state.pth` when
    a Huggingface model is saved using
    `model.save_pretrained(save_directory) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained>`_.

    Conversely, ModelOpt restores the saved state from `pretrained_model_name_or_path/modelopt_state.pth` if it exists
    when a Huggingface model is loaded using
    `cls.from_pretrained(pretrained_model_name_or_path) <https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_.


    This function should be called once in the program before loading/saving any HuggingFace models.

    Here is an example usage:

    .. code-block:: python

        from transformers import AutoModelForCausalLM
        import modelopt.torch.opt as mto

        # Enable ModelOpt save/restore for HuggingFace models
        # This only needs to be called once in the program.
        mto.enable_huggingface_checkpointing()

        # Instantiate a HuggingFace model, modelopt_state will be automatically loaded if it exists.
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    """
    # This method simply prints if ModelOpt save/restore is enabled for the HuggingFace libraries.
    for library_name in _PATCHED_LIBRARIES:
        print_rank_0(f"ModelOpt save/restore enabled for `{library_name}` library.")
