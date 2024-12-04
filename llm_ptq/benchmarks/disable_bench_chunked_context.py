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

from pathlib import Path
from typing import Dict, Tuple, Union

import tensorrt_llm.bench.benchmark.utils as utils
from tensorrt_llm.bench.benchmark.utils import (
    get_settings_from_engine as get_settings_from_engine_original,
)


def get_settings_from_engine_override(
    engine_path: Path,
) -> Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]:
    exec_settings, build_cfg = get_settings_from_engine_original(engine_path)
    # Do not enable chunking if use_paged_context_fmha if not enabled.
    exec_settings["settings_config"]["chunking"] = build_cfg["plugin_config"][
        "use_paged_context_fmha"
    ]
    return exec_settings, build_cfg


utils.get_settings_from_engine = get_settings_from_engine_override
