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

"""
Date and time utility functions for the compress module.
"""

import datetime


def get_timestamp() -> str:
    """Get a formatted timestamp string for logging.

    Returns:
        A formatted timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamped(message: str) -> str:
    """Add a timestamp prefix to a message.

    Args:
        message: The message to prefix with a timestamp.

    Returns:
        The message with a timestamp prefix in the format '[YYYY-MM-DD HH:MM:SS] message'.
    """
    return f"[{get_timestamp()}] {message}"
