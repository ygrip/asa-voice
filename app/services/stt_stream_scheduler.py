from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.config import settings
from app.providers.streaming.base import SttStreamError, StreamingSttSession
from app.services.operation_limiter import OperationLimiter
from app.services.stt_stream_protocol import SttFinalEvent, SttPartialEvent
from app.services.voice_metrics import LABEL_VALUES, voice_metrics

log = logging.getLogger("asa.stt.scheduler")


PartialSink = Callable[[SttPartialEvent], Awaitable[None]]
DecodeErrorSink = Callable[[Exception], Awaitable[None]]


@dataclass(frozen=True)
class SttStreamSchedulerMetrics:
    decode_count: int
    decode_signals: int
    coalesced_signals: int
    real_time_factor: float
    partial_interval_ms: int
    active_tasks: int


class SttStreamScheduler:
    """One bounded ingestion path and one coalescing decoder task per WebSocket session.

    PCM append never awaits inference. Decode requests collapse into one ``Event`` and the decoder
    owns the provider operation slot, so a slow model cannot create a task per frame or block the reader
    from servicing controls and ASGI keepalive traffic.
    """

    def __init__(
        self,
        session: StreamingSttSession,
        decode_limiter: OperationLimiter | None,
        emit_partial: PartialSink,
        emit_decode_error: DecodeErrorSink,
        *,
        final_limiter: OperationLimiter | None,
        base_interval_ms: int,
        max_interval_ms: int,
        rtf_slow_threshold: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if base_interval_ms <= 0 or max_interval_ms < base_interval_ms:
            raise ValueError("STT partial intervals must be positive and ordered")
        if rtf_slow_threshold <= 0:
            raise ValueError("STT RTF slow threshold must be positive")
        self.session = session
        self._decode_limiter = decode_limiter
        self._final_limiter = final_limiter
        self._emit_partial = emit_partial
        self._emit_decode_error = emit_decode_error
        self._base_interval_ms = base_interval_ms
        self._max_interval_ms = max_interval_ms
        self._rtf_slow_threshold = rtf_slow_threshold
        self._clock = clock
        self._decode_event = asyncio.Event()
        self._terminal_lock = asyncio.Lock()
        self._decoder_task: asyncio.Task[None] | None = None
        self._accepting_audio = True
        self._stopping_decoder = False
        self._closed = False
        self._decode_signals = 0
        self._coalesced_signals = 0
        self._decode_count = 0
        self._real_time_factor = 0.0
        self._partial_interval_ms = base_interval_ms
        self._last_decode_audio_ms = 0
        self._next_decode_at = 0.0
        self._waiting_for_deadline = False

    @property
    def metrics(self) -> SttStreamSchedulerMetrics:
        task = self._decoder_task
        active_tasks = int(task is not None and not task.done())
        return SttStreamSchedulerMetrics(
            decode_count=self._decode_count,
            decode_signals=self._decode_signals,
            coalesced_signals=self._coalesced_signals,
            real_time_factor=self._real_time_factor,
            partial_interval_ms=self._partial_interval_ms,
            active_tasks=active_tasks,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("STT scheduler is closed")
        if self._decoder_task is not None and not self._decoder_task.done():
            return
        self._stopping_decoder = False
        self._decoder_task = asyncio.create_task(
            self._decoder_loop(),
            name="asa-stt-coalescing-decoder",
        )

    async def append_pcm(self, frame: bytes) -> None:
        if self._closed or not self._accepting_audio:
            raise SttStreamError("STT_SESSION_FINALIZED", "STT session is not accepting audio")
        await self.session.append_pcm(frame)
        if self.session.should_decode():
            self.request_decode()

    def request_decode(self) -> None:
        if self._closed or not self._accepting_audio or self._stopping_decoder:
            return
        self._decode_signals += 1
        if self._decode_event.is_set() or self._waiting_for_deadline:
            self._coalesced_signals += 1
        self._decode_event.set()

    async def flush(self) -> SttFinalEvent:
        async with self._terminal_lock:
            self._accepting_audio = False
            settle_started = self._clock()
            await self._settle_decoder()
            settle_elapsed = self._clock() - settle_started
            if settle_elapsed > 0.05:
                # Time spent here is waiting for an in-flight PARTIAL decode to finish before the
                # final decode can start - if this is a large share of the client-observed
                # flush-to-final gap, the bottleneck is a slow/queued partial decode, not the
                # final decode itself.
                log.debug("stt flush: waited %.3fs for in-flight partial decode to settle", settle_elapsed)
            started = self._clock()
            try:
                if self._final_limiter is None:
                    final = await asyncio.wait_for(
                        self.session.flush(),
                        timeout=settings.stt_provider_final_timeout_seconds,
                    )
                else:
                    async with self._final_limiter.slot():
                        final = await asyncio.wait_for(
                            self.session.flush(),
                            timeout=settings.stt_provider_final_timeout_seconds,
                        )
            except asyncio.TimeoutError as exc:
                raise SttStreamError(
                    "STT_FINAL_TIMEOUT",
                    "STT finalization timed out",
                    retryable=True,
                ) from exc
            elapsed_ms = max(0.0, self._clock() - started) * 1000
            provider = _metric_provider(final.provider)
            voice_metrics.observe("asa_voice_stt_final_latency_ms", elapsed_ms)
            voice_metrics.observe(
                "asa_voice_stt_decode_duration_ms",
                elapsed_ms,
                provider=provider,
                profile="final",
            )
            if final.audio_dropped_ms > 0:
                voice_metrics.increment(
                    "asa_voice_stt_audio_dropped_ms_total",
                    final.audio_dropped_ms,
                    reason="pressure",
                )
            if final.finality not in {"provider_final", "local_recovered_final"}:
                voice_metrics.increment(
                    "asa_voice_stt_degraded_finals_total",
                    finality=final.finality,
                )
            if final.fallback_used:
                voice_metrics.increment(
                    "asa_voice_stt_provider_fallback_total",
                    from_provider=_metric_provider(self.session.state.provider),
                    to_provider=provider,
                    reason="provider_error",
                )
            return final

    async def reset(self) -> None:
        async with self._terminal_lock:
            await self._settle_decoder()
            await self.session.reset()
            self._accepting_audio = True
            self._stopping_decoder = False
            self._decode_event.clear()
            self._last_decode_audio_ms = 0
            self._next_decode_at = 0.0
            self._partial_interval_ms = self._base_interval_ms
            self.start()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._accepting_audio = False
        await self._settle_decoder()
        await self.session.close()

    async def _decoder_loop(self) -> None:
        try:
            while True:
                await self._decode_event.wait()
                self._decode_event.clear()
                if self._stopping_decoder or self._closed:
                    return
                if not await self._wait_for_decode_deadline():
                    return
                if not self.session.should_decode() or self.session.is_silent():
                    continue
                started = self._clock()
                audio_before = self.session.state.duration_ms
                try:
                    if self._decode_limiter is None:
                        partial = await self.session.decode_partial()
                    else:
                        async with self._decode_limiter.slot():
                            partial = await self.session.decode_partial()
                except Exception as error:  # noqa: BLE001 - provider errors stay non-fatal here
                    await self._emit_decode_error(error)
                    self._next_decode_at = self._clock() + self._partial_interval_ms / 1000
                    continue
                elapsed = max(0.0, self._clock() - started)
                audio_delta = max(1, audio_before - self._last_decode_audio_ms) / 1000
                self._last_decode_audio_ms = audio_before
                self._real_time_factor = elapsed / audio_delta
                self._decode_count += 1
                log.debug(
                    "stt partial decode #%d elapsed_s=%.3f audio_delta_s=%.3f rtf=%.2f "
                    "interval_ms=%d",
                    self._decode_count, elapsed, audio_delta, self._real_time_factor,
                    self._partial_interval_ms,
                )
                provider = _metric_provider(self.session.state.provider)
                voice_metrics.observe("asa_voice_stt_partial_latency_ms", elapsed * 1000)
                voice_metrics.observe(
                    "asa_voice_stt_decode_duration_ms",
                    elapsed * 1000,
                    provider=provider,
                    profile="partial",
                )
                self._adapt_partial_interval()
                self._next_decode_at = self._clock() + self._partial_interval_ms / 1000
                if partial is not None and not self._closed and not self._stopping_decoder:
                    await self._emit_partial(partial)
                # Frames received during inference may already satisfy the next cadence. Signal
                # once; the Event coalesces any concurrent frame signals into this same pass.
                if self._accepting_audio and self.session.should_decode():
                    self.request_decode()
        except asyncio.CancelledError:
            raise

    async def _wait_for_decode_deadline(self) -> bool:
        """Coalesce frame wakeups without allowing them to bypass adaptive cadence."""
        self._waiting_for_deadline = True
        try:
            while True:
                if self._stopping_decoder or self._closed:
                    return False
                remaining = self._next_decode_at - self._clock()
                if remaining <= 0:
                    return True
                try:
                    await asyncio.wait_for(self._decode_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    # Re-check the injected clock instead of assuming the timeout and clock are
                    # identical. This also keeps deterministic clock tests honest.
                    continue
                self._decode_event.clear()
                # A frame signal means more work is pending, not that cadence may be skipped.
                # Loop until the deadline; stop/close signals are observed at the top immediately.
        finally:
            self._waiting_for_deadline = False

    def _adapt_partial_interval(self) -> None:
        if self._real_time_factor > self._rtf_slow_threshold:
            pressure = min(2.0, self._real_time_factor / self._rtf_slow_threshold)
            self._partial_interval_ms = min(
                self._max_interval_ms,
                max(
                    self._base_interval_ms,
                    int(self._partial_interval_ms * pressure),
                ),
            )
            return
        if (
            self._real_time_factor < self._rtf_slow_threshold * 0.75
            and self._partial_interval_ms > self._base_interval_ms
        ):
            self._partial_interval_ms = max(
                self._base_interval_ms,
                int(self._partial_interval_ms * 0.8),
            )

    async def _settle_decoder(self) -> None:
        task = self._decoder_task
        if task is None:
            return
        self._stopping_decoder = True
        self._decode_event.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self._decoder_task is task:
                self._decoder_task = None
            self._decode_event.clear()


def _metric_provider(provider: str) -> str:
    if provider in LABEL_VALUES["provider"]:
        return provider
    return "unknown"
