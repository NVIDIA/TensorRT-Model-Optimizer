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

from torch import nn


class CachedModule(nn.Module):
    def __init__(self, block, select_cache_step_func) -> None:
        super().__init__()
        self.block = block
        self.select_cache_step_func = select_cache_step_func
        self.cur_step = 0
        self.cached_results = None
        self.enabled = True

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def if_cache(self):
        return self.select_cache_step_func(self.cur_step) and self.enabled

    def enable_cache(self):
        self.enabled = True

    def disable_cache(self):
        self.enabled = False
        self.cur_step = 0

    def forward(self, *args, **kwargs):
        if not self.if_cache():
            self.cached_results = self.block(*args, **kwargs)
        if self.enabled:
            self.cur_step += 1
        return self.cached_results
