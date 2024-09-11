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

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.run.run import RuntimeConfig


def _override_get_config(self) -> trtllm.ExecutorConfig:
    return trtllm.ExecutorConfig(
        scheduler_config=self.settings_config.get_scheduler_config(),
        kv_cache_config=self.settings_config.get_kvcache_config(),
        parallel_config=self.world_config.get_parallel_config(),
        batching_type=trtllm.BatchingType.INFLIGHT,
        iter_stats_max_iterations=0,
        request_stats_max_iterations=0,
        max_batch_size=self.settings_config.max_batch_size,
        max_num_tokens=self.settings_config.max_num_tokens,
        # ModelOpt modification
        # TODO: set this according to engine config
        enable_chunked_context=False,
    )


RuntimeConfig.get_config = _override_get_config
