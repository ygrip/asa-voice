import asyncio
import errno
import os
import time
import tracemalloc
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth, main, runtime
from app.config import settings
from app.providers.base import SttResult
from app.providers.errors import SttFailLoudError, SttFallbackEligibleError
from app.providers.router import SttProviderRouter
from app.providers.streaming.base import SttOptions, SttStreamError
from app.providers.streaming.factory import StreamingSttSessionFactory
from app.providers.streaming.openai_buffered_session import OpenAiBufferedFileSession
from app.routers import stt
from app.services.pcm_wav_writer import IncrementalPcmWavWriter
from app.services import pcm_wav_writer
from app.services.stt_stream_protocol import SttAudioFormat, SttStartControl
from app.services.temp_audio_cleanup import (
    OPENAI_STT_FILE_PREFIX,
    cleanup_expired_openai_stt_files,
)


class _OpenAiAdapter:
    provider_name = "openai"

    async def transcribe(self, audio_path, options) -> SttResult:
        return SttResult(
            provider="openai",
            model="gpt-test",
            text="websocket final",
            language="en",
            duration_ms=20,
            latency_ms=5,
        )


class _RecordingRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.frames = b""

    async def transcribe(self, audio_path, options, provider_override=None) -> SttResult:
        self.calls += 1
        assert provider_override == "openai"
        with wave.open(audio_path, "rb") as source:
            assert source.getnchannels() == 1
            assert source.getsampwidth() == 2
            assert source.getframerate() == 16_000
            self.frames = source.readframes(source.getnframes())
        return SttResult(
            provider="openai",
            model="gpt-test",
            text="hosted final",
            language="en",
            duration_ms=20,
            latency_ms=7,
        )


class _FailingRouter:
    async def transcribe(self, audio_path, options, provider_override=None):
        raise RuntimeError("provider failed")


class _HangingRouter:
    async def transcribe(self, audio_path, options, provider_override=None):
        await asyncio.sleep(10)


class _ClassifiedFailureRouter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def transcribe(self, audio_path, options, provider_override=None):
        raise self.error


def test_incremental_writer_creates_private_valid_wav(tmp_path: Path) -> None:
    directory = tmp_path / "stt"
    writer = IncrementalPcmWavWriter(
        directory,
        sample_rate=16_000,
        channels=1,
        max_file_bytes=10_000,
    )
    path = writer.path
    first = b"\x01\x00" * 320
    second = b"\x02\x00" * 320
    writer.append_pcm(first)
    writer.append_pcm(second)
    writer.finalize()

    assert path.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    with wave.open(str(path), "rb") as source:
        assert source.getparams()[:3] == (1, 2, 16_000)
        assert source.readframes(source.getnframes()) == first + second
    writer.close_and_delete()
    assert not path.exists()


def test_incremental_writer_setup_failure_closes_descriptor_and_unlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: dict[str, object] = {}
    real_mkstemp = pcm_wav_writer.tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        descriptor, path = real_mkstemp(*args, **kwargs)
        created.update(descriptor=descriptor, path=Path(path))
        return descriptor, path

    monkeypatch.setattr(pcm_wav_writer.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(
        pcm_wav_writer.os,
        "fchmod",
        lambda descriptor, mode: (_ for _ in ()).throw(OSError("fchmod failed")),
    )

    with pytest.raises(OSError, match="fchmod failed"):
        IncrementalPcmWavWriter(
            tmp_path,
            sample_rate=16_000,
            channels=1,
            max_file_bytes=10_000,
        )

    with pytest.raises(OSError) as closed:
        os.fstat(created["descriptor"])
    assert closed.value.errno == errno.EBADF
    assert not created["path"].exists()


def test_incremental_writer_header_failure_closes_handle_and_unlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: dict[str, object] = {}
    real_mkstemp = pcm_wav_writer.tempfile.mkstemp
    real_fdopen = pcm_wav_writer.os.fdopen

    class FailingHeaderHandle:
        def __init__(self, descriptor: int) -> None:
            self._file = real_fdopen(descriptor, "w+b", buffering=0)

        def write(self, value: bytes) -> int:
            raise OSError("header write failed")

        def close(self) -> None:
            self._file.close()

    def recording_mkstemp(*args, **kwargs):
        descriptor, path = real_mkstemp(*args, **kwargs)
        created.update(descriptor=descriptor, path=Path(path))
        return descriptor, path

    monkeypatch.setattr(pcm_wav_writer.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(
        pcm_wav_writer.os,
        "fdopen",
        lambda descriptor, *args, **kwargs: FailingHeaderHandle(descriptor),
    )

    with pytest.raises(OSError, match="header write failed"):
        IncrementalPcmWavWriter(
            tmp_path,
            sample_rate=16_000,
            channels=1,
            max_file_bytes=10_000,
        )

    with pytest.raises(OSError) as closed:
        os.fstat(created["descriptor"])
    assert closed.value.errno == errno.EBADF
    assert not created["path"].exists()


def test_incremental_writer_close_failure_still_unlinks_and_is_idempotent(
    tmp_path: Path,
) -> None:
    writer = IncrementalPcmWavWriter(
        tmp_path,
        sample_rate=16_000,
        channels=1,
        max_file_bytes=10_000,
    )
    path = writer.path
    real_file = writer._file
    descriptor = real_file.fileno()

    class FailingCloseHandle:
        def close(self) -> None:
            real_file.close()
            raise OSError("close failed")

    writer._file = FailingCloseHandle()
    with pytest.raises(OSError, match="close failed"):
        writer.close_and_delete()
    assert not path.exists()
    with pytest.raises(OSError) as closed:
        os.fstat(descriptor)
    assert closed.value.errno == errno.EBADF
    writer.close_and_delete()


def test_factory_builds_openai_buffered_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "stt_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    runtime.stt_router = SttProviderRouter(primary=_OpenAiAdapter())
    session, options = StreamingSttSessionFactory().create(_start(), "client-1")
    assert isinstance(session, OpenAiBufferedFileSession)
    assert options.provider == "openai"
    runtime.reset_components()


def test_openai_session_emits_only_one_final_and_cleans_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    router = _RecordingRouter()
    session = OpenAiBufferedFileSession(router)
    frame = b"\x01\x00" * 320

    async def exercise() -> None:
        ready = await session.configure(_options())
        assert ready.supports_partials is False
        await session.append_pcm(frame)
        assert session.should_decode() is False
        assert await session.decode_partial() is None
        assert session.state.buffered_bytes == 0
        assert session.state.utterance_bytes == 0
        final = await session.flush()
        assert final.text == "hosted final"
        assert final.finality == "provider_final"
        assert final.provider == "openai"
        assert session.state.cleanup_count == 1
        with pytest.raises(SttStreamError) as duplicate:
            await session.flush()
        assert duplicate.value.code == "STT_FINAL_ALREADY_EMITTED"

    asyncio.run(exercise())
    assert router.calls == 1
    assert router.frames == frame
    assert list(tmp_path.iterdir()) == []


def test_websocket_openai_stream_emits_ready_then_final_without_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    monkeypatch.setattr(auth, "_clients", {})
    monkeypatch.setattr("app.services.audio_service.probe_duration_seconds", lambda path: 0.02)
    runtime.stt_router = SttProviderRouter(primary=_OpenAiAdapter())
    app = FastAPI()
    app.include_router(stt.router)

    with TestClient(app).websocket_connect("/stt/stream") as socket:
        socket.send_json(_start().model_dump(by_alias=True))
        ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert ready["supportsPartials"] is False
        socket.send_bytes(b"\x01\x00" * 320)
        socket.send_json({"type": "flush", "reason": "user_stop"})
        final = socket.receive_json()
        assert final["type"] == "final"
        assert final["text"] == "websocket final"
        assert final["finality"] == "provider_final"

    runtime.reset_components()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("router", [_FailingRouter(), _HangingRouter()])
def test_openai_session_cleans_provider_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    router,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    monkeypatch.setattr(settings, "openai_stt_timeout_seconds", 0.01)
    session = OpenAiBufferedFileSession(router)

    async def exercise() -> None:
        await session.configure(_options())
        await session.append_pcm(b"\x01\x00" * 320)
        with pytest.raises((RuntimeError, SttStreamError)):
            await session.flush()
        assert session.state.cleanup_count == 1

    asyncio.run(exercise())
    assert list(tmp_path.iterdir()) == []


def test_openai_session_cleans_cancel_reset_and_limit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    monkeypatch.setattr(settings, "openai_stt_max_temp_bytes", 100)
    session = OpenAiBufferedFileSession(_RecordingRouter())

    async def exercise() -> None:
        await session.configure(_options())
        await session.reset()
        assert session.state.cleanup_count == 1
        with pytest.raises(SttStreamError) as raised:
            await session.append_pcm(b"\x01\x00" * 320)
        assert raised.value.code == "STT_TEMP_FILE_LIMIT"
        assert session.state.cleanup_count == 2
        await session.reset()
        await session.append_pcm(b"\x01\x00")
        await session.close()
        await session.close()
        assert session.state.cleanup_count == 3

    asyncio.run(exercise())
    assert list(tmp_path.iterdir()) == []


def test_openai_session_task_cancellation_cleans_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    session = OpenAiBufferedFileSession(_HangingRouter())

    async def exercise() -> None:
        await session.configure(_options())
        await session.append_pcm(b"\x01\x00" * 320)
        finalization = asyncio.create_task(session.flush())
        await asyncio.sleep(0)
        finalization.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finalization
        assert session.state.cleanup_count == 1

    asyncio.run(exercise())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (SttFailLoudError("invalid API key"), "STT_PROVIDER_REJECTED", False),
        (SttFallbackEligibleError("provider unavailable"), "STT_PROVIDER_UNAVAILABLE", True),
    ],
)
def test_openai_session_preserves_provider_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    code: str,
    retryable: bool,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    session = OpenAiBufferedFileSession(_ClassifiedFailureRouter(error))

    async def exercise() -> None:
        await session.configure(_options())
        await session.append_pcm(b"\x01\x00" * 320)
        with pytest.raises(SttStreamError) as raised:
            await session.flush()
        assert raised.value.code == code
        assert raised.value.retryable is retryable
        assert session.state.cleanup_count == 1

    asyncio.run(exercise())
    assert list(tmp_path.iterdir()) == []


def test_openai_session_rejects_symlink_buffer_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "stt"
    symlink.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(symlink))
    session = OpenAiBufferedFileSession(_RecordingRouter())

    async def exercise() -> None:
        with pytest.raises(SttStreamError) as raised:
            await session.configure(_options())
        assert raised.value.code == "STT_TEMP_DIRECTORY_UNAVAILABLE"

    asyncio.run(exercise())
    assert list(target.iterdir()) == []


def test_disk_backed_session_heap_does_not_scale_with_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "openai_stt_buffer_directory", str(tmp_path))
    monkeypatch.setattr(settings, "openai_stt_max_temp_bytes", 5_000_000)
    session = OpenAiBufferedFileSession(_RecordingRouter())
    options = replace(
        _options(),
        mode="dictation",
        max_duration_seconds=300,
        max_session_bytes=5_000_000,
    )
    frame = b"\x01\x00" * 320

    async def exercise() -> None:
        await session.configure(options)
        tracemalloc.start()
        for _ in range(5_000):
            await session.append_pcm(frame)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 1_000_000
        assert session.state.received_bytes == 3_200_000
        assert session.state.buffered_bytes == 0
        await session.close()

    asyncio.run(exercise())


def test_orphan_cleanup_removes_only_expired_owned_regular_files(tmp_path: Path) -> None:
    directory = tmp_path / "stt"
    directory.mkdir()
    expired = directory / f"{OPENAI_STT_FILE_PREFIX}expired.wav"
    fresh = directory / f"{OPENAI_STT_FILE_PREFIX}fresh.wav"
    foreign = directory / "foreign.wav"
    target = directory / "target"
    symlink = directory / f"{OPENAI_STT_FILE_PREFIX}link.wav"
    for path in (expired, fresh, foreign, target):
        path.write_bytes(b"x")
    symlink.symlink_to(target)
    now = time.time()
    os.utime(expired, (now - 100, now - 100))

    assert cleanup_expired_openai_stt_files(directory, 50, now=now) == 1
    assert not expired.exists()
    assert fresh.exists() and foreign.exists() and symlink.is_symlink()


def test_cleanup_does_not_create_missing_directory(tmp_path: Path) -> None:
    directory = tmp_path / "missing"
    assert cleanup_expired_openai_stt_files(directory, 60) == 0
    assert not directory.exists()


def test_local_only_lifespan_never_runs_hosted_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "faster_whisper")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    called = False

    def unexpected_cleanup(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("local startup must not touch hosted temporary storage")

    monkeypatch.setattr(main, "cleanup_expired_openai_stt_files", unexpected_cleanup)
    monkeypatch.setattr(runtime, "load_local_stt_service", lambda: object())
    monkeypatch.setattr(runtime, "load_local_tts_service", lambda: object())
    monkeypatch.setattr(runtime, "build_routers", lambda: None)

    async def exercise() -> None:
        async with main.lifespan(FastAPI()):
            assert called is False

    asyncio.run(exercise())


def test_hosted_lifespan_attempts_cleanup_but_survives_io_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "openai")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    calls = 0

    def failing_cleanup(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError("read-only filesystem")

    monkeypatch.setattr(main, "cleanup_expired_openai_stt_files", failing_cleanup)
    monkeypatch.setattr(runtime, "load_local_tts_service", lambda: object())
    monkeypatch.setattr(runtime, "build_routers", lambda: None)

    async def exercise() -> None:
        async with main.lifespan(FastAPI()):
            assert calls == 1

    asyncio.run(exercise())


def _options() -> SttOptions:
    return SttOptions(
        protocol_version="2",
        session_id="session-1",
        request_id="request-1",
        client_id="client-1",
        mode="command",
        provider="openai",
        sample_rate=16_000,
        channels=1,
        sample_format="s16le",
        frame_duration_ms=20,
        language="en",
        prompt="ASA command",
        hotwords=("ASA",),
        max_duration_seconds=15,
        max_frame_bytes=4_096,
        max_session_bytes=9_600_000,
        max_queue_ms=2_000,
    )


def _start() -> SttStartControl:
    return SttStartControl(
        type="start",
        protocolVersion="2",
        sessionId="session-1",
        requestId="request-1",
        mode="command",
        provider="auto",
        audio=SttAudioFormat(
            sampleRate=16_000,
            channels=1,
            sampleFormat="s16le",
            frameDurationMs=20,
        ),
        language="en",
        prompt="ASA command",
        hotwords=["ASA"],
        maxDurationSeconds=15,
    )
