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

from pathlib import Path
from typing import Union

import tensorrt_llm.bench.benchmark.utils.general as general
from tensorrt_llm.bench.benchmark.utils.general import (
    get_settings_from_engine as get_settings_from_engine_original,
)


def get_settings_from_engine_override(
    engine_path: Path,
) -> tuple[dict[str, Union[str, int]], dict[str, Union[str, int]]]:
    exec_settings, build_cfg = get_settings_from_engine_original(engine_path)
    # Do not enable chunking if use_paged_context_fmha if not enabled.
    exec_settings["settings_config"]["chunking"] = build_cfg["plugin_config"][
        "use_paged_context_fmha"
    ]
    return exec_settings, build_cfg


general.get_settings_from_engine = get_settings_from_engine_override
