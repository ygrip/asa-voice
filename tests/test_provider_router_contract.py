"""Unit tests for SttProviderRouter/TtsProviderRouter (setara-s94o.4).

Acceptance criteria covered here:
- router calls primary and returns its result when no fallback is configured
- router raises primary's exception when fallback is None
"""
import asyncio

import pytest

from app.providers.base import SttOptions, SttResult, TtsOptions, TtsResult
from app.providers.router import SttProviderRouter, TtsProviderRouter


class _FakeSttAdapter:
    def __init__(self, result: SttResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls = 0

    async def transcribe(self, audio_path, options):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


def _stt_result(provider: str = "faster_whisper") -> SttResult:
    return SttResult(
        provider=provider, model="test", text="hello", language="en",
        duration_ms=100, latency_ms=10, segments=[],
    )


def test_stt_router_calls_primary_and_returns_its_result_with_no_fallback() -> None:
    primary = _FakeSttAdapter(result=_stt_result())
    router = SttProviderRouter(primary=primary)

    result = asyncio.run(router.transcribe("audio.wav", SttOptions()))

    assert primary.calls == 1
    assert result.provider == "faster_whisper"
    assert result.fallback_used is False


def test_stt_router_raises_primary_error_when_fallback_is_none() -> None:
    primary = _FakeSttAdapter(error=RuntimeError("primary failed"))
    router = SttProviderRouter(primary=primary)

    with pytest.raises(RuntimeError, match="primary failed"):
        asyncio.run(router.transcribe("audio.wav", SttOptions()))


def test_stt_router_falls_back_and_marks_fallback_used() -> None:
    primary = _FakeSttAdapter(error=RuntimeError("primary failed"))
    fallback = _FakeSttAdapter(result=_stt_result(provider="faster_whisper_fallback"))
    router = SttProviderRouter(primary=primary, fallback=fallback)

    result = asyncio.run(router.transcribe("audio.wav", SttOptions()))

    assert result.provider == "faster_whisper_fallback"
    assert result.fallback_used is True


class _FakeTtsAdapter:
    def __init__(self, result: TtsResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls = 0

    async def synthesize(self, text, options):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


def _tts_result(provider: str = "pocket_tts") -> TtsResult:
    return TtsResult(provider=provider, model="test", audio_path="/tmp/x.wav", content_type="audio/wav", latency_ms=5)


def test_tts_router_calls_primary_and_returns_its_result_with_no_fallback() -> None:
    primary = _FakeTtsAdapter(result=_tts_result())
    router = TtsProviderRouter(primary=primary)

    result = asyncio.run(router.synthesize("hello", TtsOptions()))

    assert primary.calls == 1
    assert result.provider == "pocket_tts"


def test_tts_router_raises_primary_error_when_fallback_is_none() -> None:
    primary = _FakeTtsAdapter(error=RuntimeError("primary failed"))
    router = TtsProviderRouter(primary=primary)

    with pytest.raises(RuntimeError, match="primary failed"):
        asyncio.run(router.synthesize("hello", TtsOptions()))
