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

import os
import types
import warnings
from contextlib import contextmanager

import torch
from transformers import PreTrainedModel
from transformers import modeling_utils as tf_modeling_utils

from .huggingface import _new_save_pretrained, _patch_model_init_for_modelopt, register_for_patching

__all__ = []


@contextmanager
def _undo_torch_init_override_by_transformers():
    if not hasattr(tf_modeling_utils, "TORCH_INIT_FUNCTIONS"):
        yield
        return
    # transformers override weight initialization during model instantiation for faster loading;
    # this leads to a secondary bug causing fx symbolic tracing to fail (torch does not allow
    # overriding torch.nn.init functions - fx tracing asserts that this does not happen and fails)
    # correct fx symbolic tracing is needed for NAS/Pruned model restoration
    # lets restore the original init functions before modelopt restore so that tracing works during nas restore
    # weight initialization is anyways done, so this wont affect performance
    modelopt_reverted_torch_init_funcs = {}
    for name, init_func in tf_modeling_utils.TORCH_INIT_FUNCTIONS.items():
        torch_init_func = getattr(torch.nn.init, name)
        # Check if the init function has been overridden by transformers
        if id(torch_init_func) != id(init_func):
            modelopt_reverted_torch_init_funcs[name] = torch_init_func
            setattr(torch.nn.init, name, init_func)

    yield

    for name, init_func in modelopt_reverted_torch_init_funcs.items():
        setattr(torch.nn.init, name, init_func)


def _new_from_pretrained(cls, /, pretrained_model_name_or_path, *args, **kwargs):
    """Patch for `cls.from_pretrained` method to restore ModelOpt state."""
    if kwargs.get("tp_plan") is not None:
        raise NotImplementedError(
            "ModelOpt does not support tensor parallelism for Huggingface transformers models yet. "
            "Please use multi-GPU non-tensor parallel inference by specifying `device_map` in the `from_pretrained` API"
        )

    original_world_size = None
    if kwargs.get("device_map") == "auto" and os.environ.get("WORLD_SIZE"):
        # Transformers overrides device_map ="auto" when world_size is > 0 to use tensor parallelism
        # We dont support tensor parallelism yet, so lets unset WORLD_SIZE env variable when the original
        # `from_pretrained` is called and restore it after the model is loaded
        # TODO: remove this once we support tensor parallelism
        original_world_size = os.environ["WORLD_SIZE"]
        del os.environ["WORLD_SIZE"]
        warnings.warn(
            f"Distributed setup with world_size={original_world_size} detected with device_map='auto' - Huggingface"
            "transformers now uses tensor parallelism for this case. "
            "ModelOpt does not support tensor parallelism for Huggingface transformers models yet. "
            "Hence, overriding Huggingface transformers behavior to disable tensor parallelism."
        )

    with _patch_model_init_for_modelopt(
        cls, pretrained_model_name_or_path, extra_context=_undo_torch_init_override_by_transformers
    ):
        model = types.MethodType(cls._modelopt_cache["from_pretrained"].__func__, cls)(
            pretrained_model_name_or_path, *args, **kwargs
        )

    if original_world_size is not None:
        os.environ["WORLD_SIZE"] = original_world_size

    return model


def _new_from_config(cls, /, config, **kwargs):
    """Patch for `cls.from_config` method to restore ModelOpt state."""
    with _patch_model_init_for_modelopt(
        cls, config._name_or_path, extra_context=_undo_torch_init_override_by_transformers
    ):
        model = types.MethodType(cls._modelopt_cache["_from_config"].__func__, cls)(
            config, **kwargs
        )
    return model


pretrained_model_patch_methods = [
    ("from_pretrained", classmethod(_new_from_pretrained)),
    # We need to patch _from_config of PreTrainedModel; from_config is a private method in _BaseAutoModelClass and
    # patching it is more complex
    ("_from_config", classmethod(_new_from_config)),
    ("save_pretrained", _new_save_pretrained),
]

register_for_patching("transformers", PreTrainedModel, pretrained_model_patch_methods)
