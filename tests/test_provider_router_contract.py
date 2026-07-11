"""Unit tests for SttProviderRouter/TtsProviderRouter (setara-s94o.4).

Acceptance criteria covered here:
- router calls primary and returns its result when no fallback is configured
- router raises primary's exception when fallback is None
"""
import asyncio

import pytest

from app.providers.base import SttOptions, SttResult, TtsOptions, TtsResult
from app.providers.errors import SttFallbackEligibleError
from app.providers.router import SttProviderRouter, TtsProviderRouter


class _FakeSttAdapter:
    def __init__(self, provider_name="faster_whisper", result: SttResult | None = None, error: Exception | None = None):
        self.provider_name = provider_name
        self._result = result
        self._error = error
        self.calls = 0

    async def transcribe(self, audio_path, options):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result

    async def transcribe_array(self, audio, options):
        return await self.transcribe("<in-memory>", options)


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


def test_stt_router_falls_back_only_on_fallback_eligible_error() -> None:
    primary = _FakeSttAdapter(
        provider_name="openai", error=SttFallbackEligibleError("transient failure")
    )
    fallback = _FakeSttAdapter(
        provider_name="faster_whisper", result=_stt_result(provider="faster_whisper")
    )
    router = SttProviderRouter(primary=primary, fallback=fallback)

    result = asyncio.run(router.transcribe("audio.wav", SttOptions(client_id="test")))

    assert result.provider == "faster_whisper"
    assert result.fallback_used is True


def test_stt_router_does_not_fall_back_on_non_fallback_eligible_error() -> None:
    # e.g. a fail-loud auth/billing error must never be silently masked by a fallback.
    primary = _FakeSttAdapter(provider_name="openai", error=RuntimeError("bad api key"))
    fallback = _FakeSttAdapter(provider_name="faster_whisper", result=_stt_result())
    router = SttProviderRouter(primary=primary, fallback=fallback)

    with pytest.raises(RuntimeError, match="bad api key"):
        asyncio.run(router.transcribe("audio.wav", SttOptions(client_id="test")))
    assert fallback.calls == 0


def test_stt_router_resolve_provider_supports_override() -> None:
    primary = _FakeSttAdapter(provider_name="openai", result=_stt_result(provider="openai"))
    fallback = _FakeSttAdapter(
        provider_name="faster_whisper", result=_stt_result(provider="faster_whisper")
    )
    router = SttProviderRouter(primary=primary, fallback=fallback)

    result = asyncio.run(
        router.transcribe("audio.wav", SttOptions(client_id="test"), provider_override="faster_whisper")
    )

    assert result.provider == "faster_whisper"
    assert primary.calls == 0
    assert fallback.calls == 1


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
