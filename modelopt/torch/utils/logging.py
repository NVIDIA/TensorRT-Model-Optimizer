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

"""Utility functions for logging."""

import contextlib
import os
import re
import warnings
from contextlib import contextmanager
from inspect import signature

import tqdm

from . import distributed as dist

__all__ = ["DeprecatedError", "no_stdout", "num2hrb", "print_rank_0", "silence_matched_warnings"]


def num2hrb(num: float, suffix="") -> str:
    """Convert big floating number to human readable string."""
    step = 1000  # step between units is 1000
    units = ["", "K", "M", "G", "T", "P", "E"]
    while abs(num) >= step and len(units) > 1:
        num /= step
        units.pop(0)
    return f"{num:3.2f}{units[0]}{suffix}"


@contextmanager
def _monkeypatched(obj, name, patch):
    """Temporarily monkeypatch."""
    old_attr = getattr(obj, name)
    setattr(obj, name, patch(old_attr))
    try:
        yield
    finally:
        setattr(obj, name, old_attr)


@contextmanager
def _disable_tqdm():
    """Context manager to disable tqdm.

    tqdm progress bar outputs to sys.stderr.
    Silencing sys.stderr to silence tqdm will also prevent error messages from streamed out.

    Therefore, monkey patching is used to silence tqdm module without silencing sys.stderr.
    """

    def _patch(old_init):
        def _new_init(self, *args, **kwargs):
            # get the index of disable from function signature
            index_disable = list(signature(old_init).parameters.keys()).index("disable") - 1

            if len(args) >= index_disable:
                # if arg "disable" is passed as a positional arg,
                # overwrite pos args with disable = False
                args = (*args[: index_disable - 1], True, *args[index_disable:])
            else:
                kwargs["disable"] = True

            # initialize tqdm with updated args reflecting "disable" = True
            old_init(self, *args, **kwargs)

        return _new_init

    with _monkeypatched(tqdm.std.tqdm, "__init__", _patch):
        yield


@contextmanager
def no_stdout():
    """Silences stdout within the invoked context."""
    # Special disable for tqdm
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), _disable_tqdm():
        yield


def print_rank_0(*args, **kwargs):
    """Prints only on the master process."""
    if dist.is_master():
        print(*args, **kwargs, flush=True)


@contextlib.contextmanager
def silence_matched_warnings(pattern=None):
    """Silences warnings that match a given pattern.

    Args:
        pattern (str): The pattern to match against warning messages. Defaults to None.

    """
    try:
        compiled_pattern = re.compile(pattern)
    except (re.error, TypeError):
        compiled_pattern = None

    if compiled_pattern is not None:

        def custom_showwarning(message, category, filename, lineno, file=None, line=None):
            if compiled_pattern is None or not compiled_pattern.search(str(message)):
                original_showwarning(message, category, filename, lineno, file, line)

        original_showwarning = warnings.showwarning
        warnings.showwarning = custom_showwarning
    try:
        yield
    finally:
        if compiled_pattern is not None:
            warnings.showwarning = original_showwarning


class DeprecatedError(NotImplementedError):
    """Error for deprecated functions."""
