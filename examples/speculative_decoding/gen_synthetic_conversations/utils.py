# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio


class AsyncPrioritySemaphore:
    """An asyncio Semaphore that respects task priority.

    Waiters acquire the semaphore in priority order. A lower number
    represents a higher priority. Waiters with the same priority are
    served in a first-in, first-out (FIFO) order.

    This implementation is not a subclass of asyncio.Semaphore but provides a
    compatible interface.
    """

    def __init__(self, value: int = 1) -> None:
        """Initialize the semaphore with a given initial value."""
        if value < 0:
            msg = "Semaphore initial value must be >= 0"
            raise ValueError(msg)
        self._count = value
        self._lock = asyncio.Lock()
        self._waiters = asyncio.PriorityQueue()
        self._fifo_counter = 0

    async def acquire(self, priority: int = 0) -> None:
        """Acquire the semaphore, blocking if necessary, respecting priority.

        Args:
            priority (int): The priority of the acquiring task.
                           Lower numbers are higher priority. Defaults to 0.
        """
        async with self._lock:
            if self._count > 0:
                self._count -= 1
                return

            self._fifo_counter += 1
            future = asyncio.get_running_loop().create_future()
            await self._waiters.put((priority, self._fifo_counter, future))

        try:
            await future
        except asyncio.CancelledError:
            # If the waiting task is cancelled, we must try to remove its
            # future from the queue to prevent memory leaks. This is a
            # best-effort attempt. The release() method also handles
            # cancelled futures.
            if not future.done():
                future.cancel()
            raise

    async def release(self) -> None:
        """Release the semaphore, waking up the highest-priority waiter if any."""
        async with self._lock:
            self._count += 1
            while not self._waiters.empty():
                _, _, future = self._waiters.get_nowait()

                if future.cancelled():
                    continue

                future.set_result(True)
                self._count -= 1
                break

    def locked(self) -> bool:
        """Returns True if the semaphore cannot be acquired immediately."""
        return self._count == 0

    @property
    def value(self) -> int:
        """The current semaphore count."""
        return self._count

    async def __aenter__(self) -> "AsyncPrioritySemaphore":
        """Acquire the semaphore when entering the context manager."""
        await self.acquire(priority=0)
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        """Release the semaphore when exiting the context manager."""
        await self.release()
