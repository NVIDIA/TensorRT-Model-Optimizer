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

"""Deprecated. Placeholder module for throwing deprecated error."""

from modelopt.torch.utils import DeprecatedError


def match_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")


def group_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")


def freeze_parameters(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This module is deprecated. ")


def quant_weight_inplace(*args, **kwargs):
    """Deprecated. Placeholder function for throwing deprecated error."""
    raise DeprecatedError("This method is deprecated. ")
