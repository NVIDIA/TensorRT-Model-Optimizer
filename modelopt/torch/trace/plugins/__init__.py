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

"""Handles tracing plugins for third-party modules."""

import warnings as _warnings

try:
    from .megatron import *

except ImportError:
    pass
except Exception as e:
    _warnings.warn(f"Failed to import megatron plugin due to: {e!r}")

try:
    from .transformers import *

except ImportError:
    pass
except Exception as e:
    _warnings.warn(f"Failed to import transformers plugin due to: {e!r}")
