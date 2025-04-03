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


import json
import os
from contextlib import contextmanager
from io import TextIOWrapper
from typing import IO, Any, Iterator, TextIO, Union


@contextmanager
def file_descriptor(f: Union[str, IO], mode: str = "r") -> Iterator[IO]:
    opened = False
    try:
        if isinstance(f, str):
            f = open(f, mode)
            opened = True
        yield f
    finally:
        if opened:
            assert isinstance(f, TextIOWrapper), type(f)
            f.close()


def save_jsonl(f: Union[str, TextIO], obj: Any, **kwargs) -> None:
    assert isinstance(f, str), type(f)
    os.makedirs(os.path.dirname(f), exist_ok=True)

    with file_descriptor(f, mode="w") as fd:
        fd.write("\n".join(json.dumps(datum, **kwargs) for datum in obj))
