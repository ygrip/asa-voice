import asyncio

import pytest

from app.providers.streaming.base import SttStreamError, SttStreamState
from app.services.stt_stream_protocol import SttFinalEvent, SttPartialEvent
from app.services.stt_stream_scheduler import SttStreamScheduler
from app.services.operation_limiter import OperationLimiter


class _BoundedSlowSession:
    def __init__(self, *, max_frames: int = 2_000, decode_delay: float = 0.0) -> None:
        self.max_frames = max_frames
        self.decode_delay = decode_delay
        self.decode_started = asyncio.Event()
        self.decode_release = asyncio.Event()
        if decode_delay > 0:
            self.decode_release.set()
        self.frames: list[bytes] = []
        self.pending_frames = 0
        self.decode_calls = 0
        self.flush_calls = 0
        self.close_calls = 0
        self.reset_calls = 0
        self.final_emitted = False

    @property
    def state(self) -> SttStreamState:
        return SttStreamState(
            configured=True,
            accepting_audio=not self.final_emitted,
            closed=self.close_calls > 0,
            final_emitted=self.final_emitted,
            received_bytes=sum(len(frame) for frame in self.frames),
            duration_ms=len(self.frames) * 20,
            buffered_bytes=sum(len(frame) for frame in self.frames),
            utterance_bytes=sum(len(frame) for frame in self.frames),
            capped=False,
            sequence=self.decode_calls,
            provider="faster_whisper",
            model="slow-test",
            audio_dropped_ms=0,
            finality="provider_final" if self.final_emitted else None,
            cleanup_count=self.close_calls,
        )

    async def append_pcm(self, frame: bytes) -> None:
        if len(self.frames) >= self.max_frames:
            raise SttStreamError(
                "STT_STREAM_QUEUE_LIMIT",
                "Synthetic bounded session is overloaded",
                retryable=True,
            )
        self.frames.append(frame)
        self.pending_frames += 1

    def should_decode(self) -> bool:
        return self.pending_frames > 0 and not self.final_emitted

    def has_buffered_audio(self) -> bool:
        return bool(self.frames)

    def is_silent(self) -> bool:
        return False

    def last_frame_had_speech(self) -> bool:
        return True

    async def decode_partial(self) -> SttPartialEvent | None:
        represented = self.pending_frames
        self.decode_started.set()
        if self.decode_delay:
            await asyncio.sleep(self.decode_delay)
        else:
            await self.decode_release.wait()
        self.pending_frames = max(0, self.pending_frames - represented)
        self.decode_calls += 1
        return SttPartialEvent(
            type="partial",
            sequence=self.decode_calls,
            text="partial",
            committedText="",
            unstableText="partial",
            audioReceivedMs=len(self.frames) * 20,
        )

    async def flush(self) -> SttFinalEvent:
        if self.final_emitted:
            raise SttStreamError("STT_FINAL_ALREADY_EMITTED", "Final already emitted")
        self.flush_calls += 1
        self.final_emitted = True
        markers = {frame[:1] for frame in self.frames}
        words = [
            word
            for marker, word in ((b"B", "beginning"), (b"M", "middle"), (b"E", "end"))
            if marker in markers
        ]
        return SttFinalEvent(
            type="final",
            text=" ".join(words),
            finality="provider_final",
            provider="faster_whisper",
            model="accurate-final-profile",
            durationMs=len(self.frames) * 20,
            latencyMs=1,
            fallbackUsed=False,
            audioDroppedMs=0,
        )

    async def reset(self) -> None:
        self.reset_calls += 1
        self.frames.clear()
        self.pending_frames = 0
        self.final_emitted = False
        self.decode_started.clear()
        self.decode_release.clear()

    async def close(self) -> None:
        self.close_calls += 1


def _scheduler(session: _BoundedSlowSession, partials: list[SttPartialEvent]) -> SttStreamScheduler:
    async def emit_partial(partial: SttPartialEvent) -> None:
        partials.append(partial)

    async def emit_error(error: Exception) -> None:
        raise error

    return SttStreamScheduler(
        session,
        OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
        emit_partial,
        emit_error,
        final_limiter=OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
        base_interval_ms=1,
        max_interval_ms=20,
        rtf_slow_threshold=1.0,
    )


def test_31_8_second_slow_decode_keeps_ingestion_responsive_and_coalesced() -> None:
    async def exercise() -> None:
        session = _BoundedSlowSession(max_frames=2_000)
        partials: list[SttPartialEvent] = []
        scheduler = _scheduler(session, partials)
        scheduler.start()

        await scheduler.append_pcm(b"B" + bytes(639))
        await asyncio.wait_for(session.decode_started.wait(), timeout=1)
        # Original regression duration: 1,590 exact 20 ms frames = 31.8 seconds. The decoder is
        # deliberately saturated while the reader-side append path retains all frames.
        for index in range(1, 1_590):
            marker = b"M" if index == 795 else b"E" if index == 1_589 else b"S"
            await scheduler.append_pcm(marker + bytes(639))

        assert len(session.frames) == 1_590
        assert scheduler.metrics.active_tasks == 1
        assert scheduler.metrics.decode_signals == 1_590
        assert scheduler.metrics.coalesced_signals >= 1_588

        flush_task = asyncio.create_task(scheduler.flush())
        await asyncio.sleep(0)
        assert not flush_task.done()
        session.decode_release.set()
        final = await asyncio.wait_for(flush_task, timeout=1)

        assert final.duration_ms == 31_800
        assert final.text == "beginning middle end"
        assert session.flush_calls == 1
        with pytest.raises(SttStreamError, match="Final already emitted"):
            await scheduler.flush()
        assert session.flush_calls == 1
        await scheduler.close()
        assert scheduler.metrics.active_tasks == 0
        assert session.close_calls == 1

    asyncio.run(exercise())


def test_overload_is_explicit_and_cancel_leaves_no_decoder_task() -> None:
    async def exercise() -> None:
        session = _BoundedSlowSession(max_frames=3)
        scheduler = _scheduler(session, [])
        scheduler.start()
        for _ in range(3):
            await scheduler.append_pcm(b"S" * 640)
        await asyncio.wait_for(session.decode_started.wait(), timeout=1)

        with pytest.raises(SttStreamError) as raised:
            await scheduler.append_pcm(b"S" * 640)
        assert raised.value.code == "STT_STREAM_QUEUE_LIMIT"
        assert raised.value.retryable is True

        close_task = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        session.decode_release.set()
        await close_task
        assert scheduler.metrics.active_tasks == 0
        assert session.close_calls == 1

    asyncio.run(exercise())


def test_rtf_backs_off_partial_cadence_without_changing_final_profile() -> None:
    async def exercise() -> None:
        session = _BoundedSlowSession(decode_delay=0.04)
        scheduler = _scheduler(session, [])
        scheduler.start()
        await scheduler.append_pcm(b"S" * 640)

        for _ in range(100):
            if scheduler.metrics.decode_count:
                break
            await asyncio.sleep(0.002)
        assert scheduler.metrics.real_time_factor > 1
        assert scheduler.metrics.partial_interval_ms > 1

        final = await scheduler.flush()
        assert final.model == "accurate-final-profile"
        assert session.flush_calls == 1
        await scheduler.close()

    asyncio.run(exercise())


def test_frame_signals_coalesce_without_bypassing_partial_deadline() -> None:
    async def exercise() -> None:
        session = _BoundedSlowSession(decode_delay=0.002)
        partials: list[SttPartialEvent] = []

        async def emit_partial(partial: SttPartialEvent) -> None:
            partials.append(partial)

        async def emit_error(error: Exception) -> None:
            raise error

        scheduler = SttStreamScheduler(
            session,
            OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
            emit_partial,
            emit_error,
            final_limiter=OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
            base_interval_ms=80,
            max_interval_ms=80,
            rtf_slow_threshold=10.0,
        )
        scheduler.start()
        await scheduler.append_pcm(b"S" * 640)
        for _ in range(100):
            if scheduler.metrics.decode_count == 1:
                break
            await asyncio.sleep(0.001)
        assert scheduler.metrics.decode_count == 1

        for _ in range(20):
            await scheduler.append_pcm(b"S" * 640)
            await asyncio.sleep(0)
        await asyncio.sleep(0.025)
        assert scheduler.metrics.decode_count == 1
        assert scheduler.metrics.coalesced_signals >= 19

        for _ in range(100):
            if scheduler.metrics.decode_count == 2:
                break
            await asyncio.sleep(0.002)
        assert scheduler.metrics.decode_count == 2
        await scheduler.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal", ["flush", "close"])
def test_terminal_signal_wakes_cadence_wait_promptly(terminal: str) -> None:
    async def exercise() -> None:
        session = _BoundedSlowSession(decode_delay=0.002)

        async def emit_partial(_partial: SttPartialEvent) -> None:
            return None

        async def emit_error(error: Exception) -> None:
            raise error

        scheduler = SttStreamScheduler(
            session,
            OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
            emit_partial,
            emit_error,
            final_limiter=OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="busy"),
            base_interval_ms=1_000,
            max_interval_ms=1_000,
            rtf_slow_threshold=10.0,
        )
        scheduler.start()
        await scheduler.append_pcm(b"S" * 640)
        for _ in range(100):
            if scheduler.metrics.decode_count == 1:
                break
            await asyncio.sleep(0.001)
        assert scheduler.metrics.decode_count == 1
        await scheduler.append_pcm(b"S" * 640)
        await asyncio.sleep(0.01)

        started = asyncio.get_running_loop().time()
        if terminal == "flush":
            await scheduler.flush()
            await scheduler.close()
        else:
            await scheduler.close()
        assert asyncio.get_running_loop().time() - started < 0.1
        assert scheduler.metrics.active_tasks == 0

    asyncio.run(exercise())
