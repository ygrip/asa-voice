import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import runtime
from app.config import Settings
from app.routers import tts as tts_router
from app.schemas import TtsRequest
from app.providers.base import SttOptions, TtsOptions
from app.providers.faster_whisper import FasterWhisperAdapter
from app.providers.openai_stt import OpenAiSttAdapter
from app.providers.pocket_tts import PocketTtsAdapter
from app.providers.router import SttProviderRouter, TtsProviderRouter
from app.providers.streaming.base import SttOptions as StreamSttOptions, SttStreamState
from app.providers.streaming.openai_buffered_session import OpenAiBufferedFileSession
from app.services.operation_limiter import OperationBusyError, OperationLimiter
from app.services.stt_stream_protocol import SttFinalEvent, SttPartialEvent
from app.services.stt_stream_scheduler import SttStreamScheduler


class _BlockingLocalService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, _path, _language, _vad, _context, _mode="command"):
        self.started.set()
        self.release.wait(timeout=2)
        return {
            "provider": "faster_whisper",
            "model": "local-test",
            "text": "local",
            "language": "en",
            "durationSeconds": 1,
            "segments": [],
        }


class _BlockingHostedTranscriptions:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.maximum = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, **_kwargs):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.active == self.limit:
            self.all_started.set()
        try:
            await self.release.wait()
            return SimpleNamespace(text="hosted")
        finally:
            self.active -= 1


class _BlockingTtsService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, _text, _voice_id):
        self.started.set()
        self.release.wait(timeout=2)
        return b"RIFF-test-wave"


class _SingleFrameStreamingTtsService:
    model = SimpleNamespace(sample_rate=24_000)

    def synthesize_stream(self, _text, _voice_id):
        yield b"\x01\x00"


class _TwoFrameStreamingTtsService:
    model = SimpleNamespace(sample_rate=24_000)

    def synthesize_stream(self, _text, _voice_id):
        yield b"\x01\x00"
        yield b"\x02\x00"


class _BlockingStreamingTtsService:
    model = SimpleNamespace(sample_rate=24_000)

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize_stream(self, _text, _voice_id):
        yield b"\x01\x00"
        self.started.set()
        self.release.wait(timeout=2)
        yield b"\x02\x00"


class _FailingTtsService:
    def synthesize(self, _text, _voice_id):
        raise RuntimeError("synthesis failed")


def test_local_hosted_and_tts_limits_are_independent_with_deterministic_n_plus_one(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        runtime.local_decode_limiter = OperationLimiter(
            1, busy_code="STT_LOCAL_BUSY", busy_message="local busy"
        )
        runtime.hosted_request_limiter = OperationLimiter(
            4, busy_code="STT_HOSTED_BUSY", busy_message="hosted busy"
        )
        runtime.tts_limiter = OperationLimiter(1, busy_code="TTS_BUSY", busy_message="tts busy")

        local_service = _BlockingLocalService()
        local = FasterWhisperAdapter(local_service)
        local_task = asyncio.create_task(local.transcribe("unused.wav", SttOptions(language="en")))
        for _ in range(100):
            if local_service.started.is_set():
                break
            await asyncio.sleep(0.001)
        assert local_service.started.is_set()

        hosted_calls = _BlockingHostedTranscriptions(limit=4)
        hosted_client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=hosted_calls)
        )
        hosted = OpenAiSttAdapter(client=hosted_client)
        audio_paths = []
        for index in range(5):
            path = tmp_path / f"hosted-{index}.wav"
            path.write_bytes(b"audio")
            audio_paths.append(path)
        hosted_tasks = [
            asyncio.create_task(hosted.transcribe(str(path), SttOptions(language="en")))
            for path in audio_paths[:4]
        ]
        await asyncio.wait_for(hosted_calls.all_started.wait(), timeout=1)

        tts_service = _BlockingTtsService()
        tts = PocketTtsAdapter(tts_service)
        tts_task = asyncio.create_task(tts.synthesize("hello", TtsOptions()))
        for _ in range(100):
            if tts_service.started.is_set():
                break
            await asyncio.sleep(0.001)
        assert tts_service.started.is_set()

        assert runtime.local_decode_limiter.active == 1
        assert runtime.hosted_request_limiter.active == 4
        assert hosted_calls.maximum == 4
        assert runtime.tts_limiter.active == 1

        with pytest.raises(OperationBusyError) as hosted_busy:
            await hosted.transcribe(str(audio_paths[4]), SttOptions(language="en"))
        assert hosted_busy.value.code == "STT_HOSTED_BUSY"
        assert hosted_busy.value.retryable is True
        with pytest.raises(OperationBusyError) as local_busy:
            await local.transcribe("unused.wav", SttOptions(language="en"))
        assert local_busy.value.code == "STT_LOCAL_BUSY"
        with pytest.raises(OperationBusyError) as tts_busy:
            await tts.synthesize("second", TtsOptions())
        assert tts_busy.value.code == "TTS_BUSY"

        hosted_calls.release.set()
        local_service.release.set()
        tts_service.release.set()
        hosted_results = await asyncio.gather(*hosted_tasks)
        local_result = await local_task
        tts_result = await tts_task
        assert [result.text for result in hosted_results] == ["hosted"] * 4
        assert local_result.text == "local"
        assert runtime.hosted_request_limiter.active == 0
        assert runtime.local_decode_limiter.active == 0
        assert runtime.tts_limiter.active == 0
        os.remove(tts_result.audio_path)

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


class _ResponsiveStreamSession:
    def __init__(self) -> None:
        self.frames = 0
        self.pending = 0
        self.decode_calls = 0
        self.closed = False

    @property
    def state(self) -> SttStreamState:
        return SttStreamState(
            configured=True,
            accepting_audio=not self.closed,
            closed=self.closed,
            final_emitted=False,
            received_bytes=self.frames * 640,
            duration_ms=self.frames * 20,
            buffered_bytes=self.frames * 640,
            utterance_bytes=self.frames * 640,
            capped=False,
            sequence=self.decode_calls,
            provider="faster_whisper",
            model="test",
            audio_dropped_ms=0,
            finality=None,
            cleanup_count=0,
        )

    async def append_pcm(self, _frame: bytes) -> None:
        self.frames += 1
        self.pending += 1

    def should_decode(self) -> bool:
        return self.pending > 0

    def has_buffered_audio(self) -> bool:
        return self.frames > 0

    def is_silent(self) -> bool:
        return False

    def last_frame_had_speech(self) -> bool:
        return True

    async def decode_partial(self) -> SttPartialEvent:
        self.pending = 0
        self.decode_calls += 1
        return SttPartialEvent(
            type="partial",
            sequence=self.decode_calls,
            text="partial",
            committedText="",
            unstableText="partial",
            audioReceivedMs=self.frames * 20,
        )

    async def flush(self) -> SttFinalEvent:
        return SttFinalEvent(
            type="final",
            text="final",
            finality="provider_final",
            provider="faster_whisper",
            model="test",
            durationMs=self.frames * 20,
            latencyMs=1,
            fallbackUsed=False,
            audioDroppedMs=0,
        )

    async def reset(self) -> None:
        self.pending = 0

    async def close(self) -> None:
        self.closed = True


def test_v2_ingestion_stays_responsive_while_local_decode_slot_is_held() -> None:
    async def exercise() -> None:
        limiter = OperationLimiter(1, busy_code="STT_LOCAL_BUSY", busy_message="local busy")
        held = await limiter.acquire()
        session = _ResponsiveStreamSession()
        errors: list[Exception] = []

        async def emit_partial(_partial: SttPartialEvent) -> None:
            return None

        async def emit_error(error: Exception) -> None:
            errors.append(error)

        scheduler = SttStreamScheduler(
            session,
            limiter,
            emit_partial,
            emit_error,
            final_limiter=limiter,
            base_interval_ms=1,
            max_interval_ms=2,
            rtf_slow_threshold=1.0,
        )
        scheduler.start()
        for _ in range(100):
            await scheduler.append_pcm(b"S" * 640)
        for _ in range(100):
            if errors:
                break
            await asyncio.sleep(0.001)
        assert session.frames == 100
        assert isinstance(errors[0], OperationBusyError)
        assert errors[0].code == "STT_LOCAL_BUSY"

        await held.release()
        await scheduler.append_pcm(b"S" * 640)
        for _ in range(100):
            if session.decode_calls:
                break
            await asyncio.sleep(0.001)
        assert session.decode_calls == 1
        await scheduler.close()
        assert scheduler.metrics.active_tasks == 0

    asyncio.run(exercise())


def test_hosted_buffered_streams_share_adapter_limit_without_a_second_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        monkeypatch.setattr("app.config.settings.openai_stt_buffer_directory", str(tmp_path))
        runtime.hosted_request_limiter = OperationLimiter(
            4, busy_code="STT_HOSTED_BUSY", busy_message="hosted busy"
        )
        hosted_calls = _BlockingHostedTranscriptions(limit=4)
        adapter = OpenAiSttAdapter(
            client=SimpleNamespace(audio=SimpleNamespace(transcriptions=hosted_calls))
        )
        router = SttProviderRouter(primary=adapter)
        sessions = [OpenAiBufferedFileSession(router) for _ in range(5)]
        options = StreamSttOptions(
            protocol_version="2",
            session_id="session",
            request_id="request",
            client_id="client",
            mode="dictation",
            provider="openai",
            sample_rate=16_000,
            channels=1,
            sample_format="s16le",
            frame_duration_ms=20,
            language="en",
            prompt=None,
            hotwords=(),
            max_duration_seconds=300,
            max_frame_bytes=4_096,
            max_session_bytes=9_600_000,
            max_queue_ms=2_000,
        )
        for session in sessions:
            await session.configure(options)
            await session.append_pcm(b"\x01\x00" * 320)

        active = [asyncio.create_task(session.flush()) for session in sessions[:4]]
        await asyncio.wait_for(hosted_calls.all_started.wait(), timeout=1)
        with pytest.raises(OperationBusyError) as busy:
            await sessions[4].flush()
        assert busy.value.code == "STT_HOSTED_BUSY"
        assert runtime.hosted_request_limiter.active == 4

        hosted_calls.release.set()
        finals = await asyncio.gather(*active)
        assert [final.text for final in finals] == ["hosted"] * 4
        assert runtime.hosted_request_limiter.active == 0
        assert all(not path.exists() for path in tmp_path.iterdir())

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_tts_raw_stream_uses_shared_limit_and_rejects_before_headers() -> None:
    async def exercise() -> None:
        original_router = runtime.tts_router
        runtime.tts_router = TtsProviderRouter(
            primary=PocketTtsAdapter(_SingleFrameStreamingTtsService())
        )
        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        try:
            response = await tts_router.tts_stream(
                TtsRequest(text="hello", format="pcm"), _client_id="client"
            )
            assert runtime.tts_limiter.active == 1

            with pytest.raises(HTTPException) as busy:
                await tts_router.tts_stream(
                    TtsRequest(text="second", format="pcm"), _client_id="client"
                )
            assert busy.value.status_code == 429
            assert busy.value.detail == {
                "code": "TTS_BUSY",
                "message": "tts busy",
                "retryable": True,
            }

            chunks = [chunk async for chunk in response.body_iterator]
            assert chunks == [b"\x01\x00"]
            assert runtime.tts_limiter.active == 0
            await response.background()
            assert runtime.tts_limiter.active == 0
        finally:
            runtime.tts_router = original_router

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_tts_raw_stream_background_releases_abandoned_response() -> None:
    async def exercise() -> None:
        original_router = runtime.tts_router
        runtime.tts_router = TtsProviderRouter(
            primary=PocketTtsAdapter(_SingleFrameStreamingTtsService())
        )
        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        try:
            response = await tts_router.tts_stream(
                TtsRequest(text="hello", format="pcm"), _client_id="client"
            )
            assert runtime.tts_limiter.active == 1
            await response.background()
            assert runtime.tts_limiter.active == 0
            await response.body_iterator.aclose()
            assert runtime.tts_limiter.active == 0
        finally:
            runtime.tts_router = original_router

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_tts_raw_stream_aclose_after_partial_iteration_releases_once() -> None:
    async def exercise() -> None:
        original_router = runtime.tts_router
        runtime.tts_router = TtsProviderRouter(
            primary=PocketTtsAdapter(_TwoFrameStreamingTtsService())
        )
        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        try:
            response = await tts_router.tts_stream(
                TtsRequest(text="hello", format="pcm"), _client_id="client"
            )
            assert await response.body_iterator.__anext__() == b"\x01\x00"
            assert runtime.tts_limiter.active == 1
            await response.body_iterator.aclose()
            assert runtime.tts_limiter.active == 0
            await response.background()
            assert runtime.tts_limiter.active == 0
        finally:
            runtime.tts_router = original_router

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_tts_raw_stream_construction_failure_releases_lease() -> None:
    async def exercise() -> None:
        original_router = runtime.tts_router
        runtime.tts_router = TtsProviderRouter(
            primary=PocketTtsAdapter(
                SimpleNamespace(
                    model=SimpleNamespace(), synthesize_stream=lambda *_args: iter(())
                )
            )
        )
        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        try:
            with pytest.raises(AttributeError):
                await tts_router.tts_stream(
                    TtsRequest(text="hello", format="pcm"), _client_id="client"
                )
            assert runtime.tts_limiter.active == 0
        finally:
            runtime.tts_router = original_router

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_tts_raw_stream_cancellation_waits_for_worker_before_releasing() -> None:
    async def exercise() -> None:
        original_router = runtime.tts_router
        service = _BlockingStreamingTtsService()
        runtime.tts_router = TtsProviderRouter(primary=PocketTtsAdapter(service))
        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        try:
            response = await tts_router.tts_stream(
                TtsRequest(text="hello", format="pcm"), _client_id="client"
            )
            assert await response.body_iterator.__anext__() == b"\x01\x00"
            read = asyncio.create_task(response.body_iterator.__anext__())
            for _ in range(100):
                if service.started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert service.started.is_set()
            read.cancel()
            await asyncio.sleep(0.01)
            assert runtime.tts_limiter.active == 1
            assert not read.done()

            service.release.set()
            with pytest.raises(asyncio.CancelledError):
                await read
            assert runtime.tts_limiter.active == 0
            await response.background()
            assert runtime.tts_limiter.active == 0
        finally:
            service.release.set()
            runtime.tts_router = original_router

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_operation_limiter_releases_on_slot_cancellation_and_provider_exception() -> None:
    async def exercise() -> None:
        limiter = OperationLimiter(1, busy_code="BUSY", busy_message="busy")
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def hold_slot() -> None:
            async with limiter.slot():
                entered.set()
                await blocked.wait()

        task = asyncio.create_task(hold_slot())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert limiter.active == 0
        async with limiter.slot():
            assert limiter.active == 1
        assert limiter.active == 0

        runtime.tts_limiter = OperationLimiter(
            1, busy_code="TTS_BUSY", busy_message="tts busy"
        )
        adapter = PocketTtsAdapter(_FailingTtsService())
        with pytest.raises(RuntimeError, match="synthesis failed"):
            await adapter.synthesize("hello", TtsOptions())
        assert runtime.tts_limiter.active == 0
        lease = await runtime.tts_limiter.acquire()
        await lease.release()

    asyncio.run(exercise())
    runtime.reset_operation_limiters()


def test_preferred_limits_default_and_legacy_names_remain_compatible() -> None:
    defaults = Settings(_env_file=None)
    assert defaults.local_stt_concurrency_limit() == 1
    assert defaults.hosted_stt_max_concurrent == 4
    assert defaults.tts_concurrency_limit() == 1

    legacy = Settings(
        _env_file=None,
        local_stt_max_concurrent=None,
        tts_max_concurrent=None,
        max_concurrent_stt=3,
        max_concurrent_tts=2,
    )
    assert legacy.local_stt_concurrency_limit() == 3
    assert legacy.tts_concurrency_limit() == 2
