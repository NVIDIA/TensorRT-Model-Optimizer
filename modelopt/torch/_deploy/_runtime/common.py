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

import logging
import time
from collections.abc import Callable


def timeit(method: Callable) -> Callable:
    """This function is supposed to use as a decorator to measure the execution time of another function.

    If the decorator is applied and no changes are done at the call site, this will print out the
    timing information on the log console. If the call site wants to get the time info returned, they
    should pass a dictionary named log_time like below-

    (regular_returns, ...), func_exec_time = func(regular_params, ..., log_time={})
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = (te - ts) * 1000
            return result, kw["log_time"][name]
        else:
            logging.info(f"{method.__name__}: {(te - ts) * 1000} ms")
            return result

    return timed


def init_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s P%(process)d T%(thread)d %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        file_bytes = file.read()
        return file_bytes


def read_string(file_path: str) -> str:
    with open(file_path) as file:
        data = file.read()
        return data


def write_bytes(data: bytes, file_path: str) -> None:
    with open(file_path, "wb") as file:
        file.write(data)


def write_string(data: str, file_path: str) -> None:
    with open(file_path, "w") as file:
        file.write(data)
