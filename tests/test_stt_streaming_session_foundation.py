import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth, runtime
from app.config import settings
from app.providers.router import SttProviderRouter
from app.providers.streaming.base import SttOptions, SttStreamError
from app.providers.streaming.factory import StreamingSttSessionFactory
from app.providers.streaming.faster_whisper_session import FasterWhisperRollingSession
from app.routers import stt
from app.services.stt_stream_protocol import SttAudioFormat, SttStartControl


class _FakeSttService:
    def decode_words(self, audio: np.ndarray, context) -> list[dict]:
        if audio.size == 0:
            return []
        duration = audio.size / 16_000
        return [{"w": "hello", "start": 0.0, "end": max(0.001, duration * 0.8)}]

    def transcribe_array_final(self, audio: np.ndarray, context) -> dict:
        return {"text": "hello" if audio.size else ""}


class _MarkerSttService:
    _WORDS = {1000: "beginning", 2000: "middle", 3000: "ending"}

    def decode_words(self, audio: np.ndarray, context) -> list[dict]:
        if audio.size == 0:
            return []
        marker = int(round(float(audio[-1]) * 32768))
        duration = audio.size / 16_000
        return [
            {
                "w": self._WORDS[marker],
                "start": 0.0,
                "end": max(0.001, duration * 0.8),
            }
        ]

    def transcribe_array_final(self, audio: np.ndarray, context) -> dict:
        if audio.size == 0:
            return {"text": ""}
        marker = int(round(float(audio[-1]) * 32768))
        return {"text": self._WORDS[marker]}


class _LocalAdapter:
    provider_name = "faster_whisper"


class _BlockingPartialService(_FakeSttService):
    def __init__(self) -> None:
        self.decode_started = threading.Event()
        self.decode_release = threading.Event()

    def decode_words(self, audio: np.ndarray, context) -> list[dict]:
        self.decode_started.set()
        self.decode_release.wait(timeout=2)
        return super().decode_words(audio, context)


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch: pytest.MonkeyPatch):
    runtime.reset_components()
    monkeypatch.setattr(auth, "_clients", {})
    yield
    runtime.reset_components()


def test_factory_builds_the_local_provider_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "stt_provider", "faster_whisper")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    runtime.stt_service = _FakeSttService()
    runtime.stt_router = SttProviderRouter(primary=_LocalAdapter())

    session, options = StreamingSttSessionFactory().create(_start(max_duration=10), "client-1")

    assert isinstance(session, FasterWhisperRollingSession)
    assert options.provider == "faster_whisper"
    assert options.client_id == "client-1"
    assert options.mode == "command"


def test_factory_rejects_mode_duration_above_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "faster_whisper")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    monkeypatch.setattr(settings, "stt_command_max_seconds", 5)
    runtime.stt_service = _FakeSttService()
    runtime.stt_router = SttProviderRouter(primary=_LocalAdapter())

    with pytest.raises(SttStreamError) as raised:
        StreamingSttSessionFactory().create(_start(max_duration=6), "client-1")

    assert raised.value.code == "STT_SESSION_DURATION_LIMIT"


def test_lifecycle_flushes_once_then_reset_and_close_clean_state() -> None:
    session = FasterWhisperRollingSession(_FakeSttService())
    options = _options(max_duration_seconds=10)

    async def exercise() -> None:
        ready = await session.configure(options)
        assert ready.provider == "faster_whisper"
        await session.append_pcm(_pcm_frame(1000))
        partial = await session.decode_partial()
        assert partial is not None
        assert partial.sequence == 1
        assert session.state.received_bytes == len(_pcm_frame(1000))
        assert session.state.duration_ms == 20
        assert session.state.provider == "faster_whisper"
        assert session.state.model == settings.stt_model

        final = await session.flush()
        assert final.text == "hello"
        assert final.finality == "provider_final"
        assert session.state.final_emitted is True
        assert session.state.finality == "provider_final"
        assert session.state.buffered_bytes == 0
        assert session.state.utterance_bytes == 0
        assert session.state.cleanup_count == 1
        with pytest.raises(SttStreamError) as duplicate:
            await session.flush()
        assert duplicate.value.code == "STT_FINAL_ALREADY_EMITTED"
        assert session.state.cleanup_count == 1

        await session.reset()
        assert session.state.final_emitted is False
        assert session.state.received_bytes == 0
        assert session.state.sequence == 0
        assert session.state.accepting_audio is True
        await session.append_pcm(_pcm_frame(1000))
        await session.close()
        await session.close()
        assert session.state.closed is True
        assert session.state.finality == "cancelled"
        assert session.state.buffered_bytes == 0
        assert session.state.utterance_bytes == 0
        assert session.state.cleanup_count == 2

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("option_changes", "frames", "expected_code"),
    [
        ({"max_frame_bytes": 4}, [b"\x00\x01" * 3], "STT_FRAME_TOO_LARGE"),
        ({}, [b"\x00"], "STT_INVALID_AUDIO_FRAME"),
        (
            {"max_session_bytes": 4},
            [b"\x01\x00" * 2, b"\x01\x00"],
            "STT_SESSION_BYTE_LIMIT",
        ),
        (
            {"max_duration_seconds": 1, "max_frame_bytes": 40_000, "max_queue_ms": 2_000},
            [b"\xd0\x07" * 16_000, b"\xd0\x07"],
            "STT_SESSION_DURATION_LIMIT",
        ),
        (
            {"max_queue_ms": 20},
            [b"\xd0\x07" * 320, b"\xd0\x07" * 320],
            "STT_STREAM_QUEUE_LIMIT",
        ),
    ],
)
def test_limits_fail_with_explicit_codes(
    option_changes: dict,
    frames: list[bytes],
    expected_code: str,
) -> None:
    session = FasterWhisperRollingSession(_FakeSttService())
    options = _options(**option_changes)

    async def exercise() -> None:
        await session.configure(options)
        with pytest.raises(SttStreamError) as raised:
            for frame in frames:
                await session.append_pcm(frame)
        assert raised.value.code == expected_code

    asyncio.run(exercise())


def test_long_capped_stream_preserves_beginning_middle_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_audio_seconds", 1)
    session = FasterWhisperRollingSession(_MarkerSttService())
    options = _options(
        mode="dictation",
        max_duration_seconds=10,
        max_queue_ms=5_000,
        max_session_bytes=1_000_000,
    )

    async def exercise() -> None:
        await session.configure(options)
        for marker in (1000, 2000):
            for _ in range(4):
                await session.append_pcm(_pcm_frame(marker, samples=1_600))
            await session.decode_partial()
            await session.decode_partial()
        for _ in range(4):
            await session.append_pcm(_pcm_frame(3000, samples=1_600))

        state = session.state
        assert state.capped is True
        assert state.buffered_bytes <= 16_000 * 2
        assert state.utterance_bytes <= 16_000 * 2
        assert state.audio_dropped_ms > 0

        final = await session.flush()
        assert final.text == "beginning middle ending"
        assert final.audio_dropped_ms > 0
        assert session.state.buffered_bytes == 0

    asyncio.run(exercise())


def test_v2_router_uses_factory_and_never_emits_a_second_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "faster_whisper")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    runtime.stt_service = _FakeSttService()
    runtime.stt_router = SttProviderRouter(primary=_LocalAdapter())
    app = FastAPI()
    app.include_router(stt.router)

    with TestClient(app).websocket_connect("/stt/stream") as socket:
        socket.send_json(_start(max_duration=1).model_dump(by_alias=True))
        ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert ready["protocolVersion"] == "2"

        socket.send_bytes(_pcm_frame(1000))
        socket.send_json({"type": "flush", "reason": "user_stop"})
        final = socket.receive_json()
        assert final["type"] == "final"
        assert final["text"] == "hello"
        assert final["finality"] == "provider_final"

        socket.send_json({"type": "flush", "reason": "user_stop"})
        duplicate = socket.receive_json()
        assert duplicate["type"] == "error"
        assert duplicate["code"] == "STT_FINAL_ALREADY_EMITTED"


def test_reader_services_control_while_partial_decode_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "stt_provider", "faster_whisper")
    monkeypatch.setattr(settings, "stt_fallback_provider", "none")
    monkeypatch.setattr(settings, "stt_stream_interval_ms", 20)
    monkeypatch.setattr(settings, "stt_stream_max_partial_interval_ms", 80)
    service = _BlockingPartialService()
    runtime.stt_service = service
    runtime.stt_router = SttProviderRouter(primary=_LocalAdapter())
    app = FastAPI()
    app.include_router(stt.router)

    with TestClient(app).websocket_connect("/stt/stream") as socket:
        socket.send_json(_start(max_duration=1).model_dump(by_alias=True))
        assert socket.receive_json()["type"] == "ready"
        socket.send_bytes(_pcm_frame(1000))
        assert service.decode_started.wait(timeout=1)

        # The invalid control error must arrive before slow inference is released. In the original
        # 1011 regression the receive loop awaited decode and this future timed out instead.
        socket.send_text("{")
        with ThreadPoolExecutor(max_workers=1) as executor:
            response_future = executor.submit(socket.receive_json)
            try:
                response = response_future.result(timeout=1)
            finally:
                service.decode_release.set()
        assert response["type"] == "error"
        assert response["code"] == "STT_INVALID_JSON"

        socket.send_json({"type": "flush", "reason": "user_stop"})
        while True:
            event = socket.receive_json()
            if event["type"] == "final":
                break
        assert event["text"] == "hello"


def _options(**changes) -> SttOptions:
    base = SttOptions(
        protocol_version="2",
        session_id="session-1",
        request_id="request-1",
        client_id="client-1",
        mode="command",
        provider="faster_whisper",
        sample_rate=16_000,
        channels=1,
        sample_format="s16le",
        frame_duration_ms=20,
        language="en",
        prompt="Voice command for ASA.",
        hotwords=("ASA", "Setara"),
        max_duration_seconds=10,
        max_frame_bytes=4_096,
        max_session_bytes=9_600_000,
        max_queue_ms=2_000,
    )
    return replace(base, **changes)


def _start(max_duration: int) -> SttStartControl:
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
        prompt="Voice command for ASA.",
        hotwords=["ASA", "Setara"],
        maxDurationSeconds=max_duration,
    )


def _pcm_frame(marker: int, *, samples: int = 320) -> bytes:
    return np.full(samples, marker, dtype="<i2").tobytes()
