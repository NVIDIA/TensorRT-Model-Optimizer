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

"""Handles plugins for third-party modules."""

from modelopt.torch.utils import import_plugin

from .huggingface import *

with import_plugin("megatron core dist checkpointing"):
    from .mcore_dist_checkpointing import *

with import_plugin("transformers"):
    from .transformers import *

with import_plugin("diffusers"):
    from .diffusers import *

with import_plugin("peft"):
    from .peft import *

with import_plugin("megatron"):
    from .megatron import *
