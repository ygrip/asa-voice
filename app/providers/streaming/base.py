from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.services.stt_context import SttDecodeContext
from app.services.stt_stream_protocol import (
    SttErrorEvent,
    SttFinalEvent,
    SttPartialEvent,
    SttReadyEvent,
)


@dataclass(frozen=True)
class SttOptions:
    protocol_version: str
    session_id: str
    request_id: str | None
    client_id: str
    mode: str
    provider: str
    sample_rate: int
    channels: int
    sample_format: str
    frame_duration_ms: int
    language: str | None
    prompt: str | None
    hotwords: tuple[str, ...]
    max_duration_seconds: int
    max_frame_bytes: int
    max_session_bytes: int
    max_queue_ms: int

    def decode_context(self) -> SttDecodeContext:
        return SttDecodeContext(
            language=self.language,
            prompt=self.prompt,
            hotwords=list(self.hotwords),
            requestId=self.request_id,
        )


@dataclass(frozen=True)
class SttStreamState:
    configured: bool
    accepting_audio: bool
    closed: bool
    final_emitted: bool
    received_bytes: int
    duration_ms: int
    buffered_bytes: int
    utterance_bytes: int
    capped: bool
    sequence: int
    provider: str | None
    model: str | None
    audio_dropped_ms: int
    finality: str | None
    cleanup_count: int


class SttStreamError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def to_event(self) -> SttErrorEvent:
        return SttErrorEvent(
            type="error",
            code=self.code,
            message=str(self),
            retryable=self.retryable,
        )


@runtime_checkable
class StreamingSttSession(Protocol):
    @property
    def state(self) -> SttStreamState: ...

    async def configure(self, options: SttOptions) -> SttReadyEvent: ...

    async def append_pcm(self, frame: bytes) -> None: ...

    def should_decode(self) -> bool: ...

    def has_buffered_audio(self) -> bool: ...

    def is_silent(self) -> bool: ...

    def last_frame_had_speech(self) -> bool: ...

    async def decode_partial(self) -> SttPartialEvent | None: ...

    async def flush(self) -> SttFinalEvent: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...
