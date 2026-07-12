from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


class OperationBusyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = True


@dataclass
class OperationLease:
    _limiter: "OperationLimiter"
    _released: bool = False

    async def __aenter__(self) -> "OperationLease":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._limiter._release()


class OperationLimiter:
    """Deterministic immediate-acquisition limiter for one expensive operation class."""

    def __init__(self, limit: int, *, busy_code: str, busy_message: str) -> None:
        if limit <= 0:
            raise ValueError("Operation concurrency limit must be positive")
        self.limit = limit
        self.busy_code = busy_code
        self.busy_message = busy_message
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> OperationLease:
        async with self._lock:
            if self._active >= self.limit:
                raise OperationBusyError(self.busy_code, self.busy_message)
            self._active += 1
        return OperationLease(self)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        lease = await self.acquire()
        try:
            yield
        finally:
            await lease.release()

    async def _release(self) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("Operation limiter released without an active lease")
            self._active -= 1

