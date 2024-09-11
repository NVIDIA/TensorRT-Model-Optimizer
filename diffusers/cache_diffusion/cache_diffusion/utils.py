# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re

SDXL_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: "up_blocks.2" not in name,
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]

PIXART_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: not re.search(
            r"transformer_blocks\.(2[1-7])\.", name
        ),
        "select_cache_step_func": lambda step: (step % 3) != 0,
    }
]

SVD_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: "up_blocks.3" not in name,
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]

SD3_DEFAULT_CONFIG = [
    {
        "wildcard_or_filter_func": lambda name: re.search(
            r"^((?!transformer_blocks\.(1[6-9]|2[0-3])).)*$", name
        ),
        "select_cache_step_func": lambda step: (step % 2) != 0,
    }
]


def replace_module(parent, name_path, new_module):
    path_parts = name_path.split(".")
    for part in path_parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path_parts[-1], new_module)
