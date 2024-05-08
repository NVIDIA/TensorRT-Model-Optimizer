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

import torch


class BenchmarkProfiler(object):
    cuda_event_dict: dict
    timer_dict: dict
    aux_info: dict
    started: bool

    def __init__(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}
        self.started = False

    def clean(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def get_cuda_event(self, name: str):
        if name not in self.cuda_event_dict.keys():
            event = torch.cuda.Event(enable_timing=True)
            self.cuda_event_dict[name] = event
        return self.cuda_event_dict[name]

    def record_cuda_event(self, name: str):
        if not self.started:
            return
        event = self.get_cuda_event(name)
        event.record()

    def get_timer_value(self, timer_name: str):
        # timer is in milliseconds
        return self.timer_dict[timer_name]

    def record_elapsed_time(self, start_event_name: str, end_event_name: str, timer_name: str):
        if timer_name not in self.timer_dict.keys():
            self.timer_dict[timer_name] = 0.0
        if not self.started:
            return
        self.get_cuda_event(start_event_name).synchronize()
        self.get_cuda_event(end_event_name).synchronize()
        self.timer_dict[timer_name] += self.get_cuda_event(start_event_name).elapsed_time(
            self.get_cuda_event(end_event_name)
        )

    def get_aux_info(self, aux_name):
        return self.aux_info[aux_name]

    def add_aux_info(self, aux_name: str, add_value):
        if aux_name not in self.aux_info.keys():
            self.aux_info[aux_name] = 0
        if not self.started:
            return
        self.aux_info[aux_name] += add_value
