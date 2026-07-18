"""Unit tests for OpenAiTtsAdapter + OpenAI TTS error classification (setara-nx07.2)."""
import asyncio
import os
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import runtime
from app.config import settings
from app.providers.base import TtsOptions
from app.providers.errors import TtsFailLoudError, TtsFallbackEligibleError
from app.providers.openai_tts import OpenAiTtsAdapter, classify_openai_tts_error
from app.providers.policy import InMemoryDailyQuotaStore

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")


def _status_error(cls, status_code: int, body=None):
    response = httpx.Response(status_code, request=_REQUEST)
    return cls(f"error {status_code}", response=response, body=body)


@pytest.fixture(autouse=True)
def isolated_tts_char_quota_store():
    original = runtime.tts_char_quota_store
    runtime.tts_char_quota_store = InMemoryDailyQuotaStore()
    yield
    runtime.tts_char_quota_store = original


# --- error classification -----------------------------------------------------------------


def test_timeout_is_fallback_eligible() -> None:
    exc = openai.APITimeoutError(request=_REQUEST)
    assert isinstance(classify_openai_tts_error(exc), TtsFallbackEligibleError)


def test_connection_error_is_fallback_eligible() -> None:
    exc = openai.APIConnectionError(request=_REQUEST)
    assert isinstance(classify_openai_tts_error(exc), TtsFallbackEligibleError)


def test_server_error_is_fallback_eligible() -> None:
    exc = _status_error(openai.InternalServerError, 500)
    assert isinstance(classify_openai_tts_error(exc), TtsFallbackEligibleError)


def test_plain_rate_limit_is_fallback_eligible() -> None:
    exc = _status_error(openai.RateLimitError, 429, body={"code": "rate_limit_exceeded", "type": "requests"})
    assert isinstance(classify_openai_tts_error(exc), TtsFallbackEligibleError)


def test_quota_exhausted_rate_limit_is_fail_loud() -> None:
    exc = _status_error(
        openai.RateLimitError, 429, body={"code": "insufficient_quota", "type": "insufficient_quota"}
    )
    assert isinstance(classify_openai_tts_error(exc), TtsFailLoudError)


def test_invalid_api_key_is_fail_loud() -> None:
    exc = _status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"})
    assert isinstance(classify_openai_tts_error(exc), TtsFailLoudError)


def test_unsupported_voice_bad_request_is_fail_loud() -> None:
    exc = _status_error(openai.BadRequestError, 400)
    assert isinstance(classify_openai_tts_error(exc), TtsFailLoudError)


def test_unknown_error_defaults_fail_loud() -> None:
    assert isinstance(classify_openai_tts_error(ValueError("weird")), TtsFailLoudError)


def test_already_classified_errors_pass_through_unchanged() -> None:
    fallback = TtsFallbackEligibleError("x")
    fail_loud = TtsFailLoudError("y")
    assert classify_openai_tts_error(fallback) is fallback
    assert classify_openai_tts_error(fail_loud) is fail_loud


# --- adapter: completed synthesis -----------------------------------------------------------


class _FakeSpeech:
    def __init__(self, content=b"RIFFfake", capture=None, error=None, fail_once=False):
        self._content = content
        self._capture = capture
        self._error = error
        self._fail_once = fail_once
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._capture is not None:
            self._capture.append(dict(kwargs))
        if self._error and (not self._fail_once or self.calls == 1):
            raise self._error
        return SimpleNamespace(content=self._content)


class _FakeStreamCM:
    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error

    async def __aenter__(self):
        if self._error:
            raise self._error
        return SimpleNamespace(iter_bytes=self._iter_bytes)

    async def __aexit__(self, *_args):
        return False

    async def _iter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamingResponse:
    def __init__(self, chunks=None, error=None, capture=None):
        self._chunks = chunks or []
        self._error = error
        self._capture = capture

    def create(self, **kwargs):
        if self._capture is not None:
            self._capture.append(dict(kwargs))
        return _FakeStreamCM(self._chunks, self._error)


def _fake_client(speech: _FakeSpeech | None = None, streaming: _FakeStreamingResponse | None = None):
    speech = speech or _FakeSpeech()
    speech.with_streaming_response = streaming or _FakeStreamingResponse()
    return SimpleNamespace(audio=SimpleNamespace(speech=speech))


def test_synthesize_writes_temp_file_and_returns_result() -> None:
    adapter = OpenAiTtsAdapter(client=_fake_client())

    result = asyncio.run(adapter.synthesize("hello", TtsOptions(client_id="c1")))

    try:
        assert result.provider == "openai"
        assert result.content_type == "audio/wav"
        assert os.path.getsize(result.audio_path) > 0
        assert result.latency_ms >= 0
    finally:
        os.remove(result.audio_path)


def test_synthesize_resolves_stable_voice_id_to_provider_voice() -> None:
    captured: list = []
    adapter = OpenAiTtsAdapter(client=_fake_client(_FakeSpeech(capture=captured)))

    result = asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1", voice_id="asa_bright")))
    os.remove(result.audio_path)

    assert captured[0]["voice"] == "shimmer"


def test_synthesize_omits_instructions_for_a_model_that_does_not_support_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_tts_model", "tts-1")
    monkeypatch.setattr(settings, "openai_tts_instructions", "speak warmly")
    captured: list = []
    adapter = OpenAiTtsAdapter(client=_fake_client(_FakeSpeech(capture=captured)))

    result = asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    os.remove(result.audio_path)

    assert "instructions" not in captured[0]


def test_synthesize_includes_instructions_for_a_model_that_supports_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_tts_model", "gpt-4o-mini-tts")
    monkeypatch.setattr(settings, "openai_tts_instructions", "speak warmly")
    captured: list = []
    adapter = OpenAiTtsAdapter(client=_fake_client(_FakeSpeech(capture=captured)))

    result = asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    os.remove(result.audio_path)

    assert captured[0]["instructions"] == "speak warmly"


def test_synthesize_rejects_text_over_the_configured_length_without_calling_the_provider() -> None:
    speech = _FakeSpeech()
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))

    long_text = "x" * (settings.openai_tts_max_text_chars + 1)
    with pytest.raises(TtsFailLoudError, match="max is"):
        asyncio.run(adapter.synthesize(long_text, TtsOptions(client_id="c1")))
    assert speech.calls == 0


def test_synthesize_retries_once_on_fallback_eligible_error_then_succeeds() -> None:
    error = openai.APITimeoutError(request=_REQUEST)
    speech = _FakeSpeech(error=error, fail_once=True)
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))

    result = asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    os.remove(result.audio_path)

    assert speech.calls == 2


def test_synthesize_does_not_retry_a_fail_loud_error() -> None:
    error = _status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"})
    speech = _FakeSpeech(error=error)
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))

    with pytest.raises(TtsFailLoudError):
        asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    assert speech.calls == 1


def test_synthesize_gives_up_after_one_retry() -> None:
    error = openai.APITimeoutError(request=_REQUEST)
    speech = _FakeSpeech(error=error, fail_once=False)
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))

    with pytest.raises(TtsFallbackEligibleError):
        asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    assert speech.calls == 2


def test_synthesize_rejects_once_the_daily_character_quota_is_exhausted() -> None:
    speech = _FakeSpeech()
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))
    runtime.tts_char_quota_store.record("c1", settings.max_tts_chars_per_client_per_day)

    with pytest.raises(TtsFailLoudError, match="quota exceeded"):
        asyncio.run(adapter.synthesize("hi", TtsOptions(client_id="c1")))
    assert speech.calls == 0


def test_synthesize_records_character_usage_only_after_success() -> None:
    speech = _FakeSpeech()
    adapter = OpenAiTtsAdapter(client=_fake_client(speech))

    result = asyncio.run(adapter.synthesize("hello", TtsOptions(client_id="c1")))
    os.remove(result.audio_path)

    assert runtime.tts_char_quota_store.used_seconds("c1") == len("hello")


def test_synthesize_does_not_record_usage_on_fail_loud_error() -> None:
    error = _status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"})
    adapter = OpenAiTtsAdapter(client=_fake_client(_FakeSpeech(error=error)))

    with pytest.raises(TtsFailLoudError):
        asyncio.run(adapter.synthesize("hello", TtsOptions(client_id="c1")))

    assert runtime.tts_char_quota_store.used_seconds("c1") == 0.0


# --- adapter: streaming synthesis -----------------------------------------------------------


def test_synthesize_stream_yields_chunks_with_verified_metadata() -> None:
    streaming = _FakeStreamingResponse(chunks=[b"a", b"b"])
    adapter = OpenAiTtsAdapter(client=_fake_client(streaming=streaming))

    async def exercise():
        stream = await adapter.synthesize_stream("hi", TtsOptions(client_id="c1"))
        assert stream.provider == "openai"
        assert stream.metadata.content_type == "audio/l16"
        assert stream.metadata.sample_rate == 24_000
        assert stream.metadata.channels == 1
        assert stream.metadata.sample_format == "s16le"
        chunks = [chunk async for chunk in stream.chunks]
        assert chunks == [b"a", b"b"]

    asyncio.run(exercise())


def test_synthesize_stream_raises_classified_error_before_first_chunk() -> None:
    error = openai.APITimeoutError(request=_REQUEST)
    streaming = _FakeStreamingResponse(error=error)
    adapter = OpenAiTtsAdapter(client=_fake_client(streaming=streaming))

    async def exercise():
        stream = await adapter.synthesize_stream("hi", TtsOptions(client_id="c1"))
        with pytest.raises(TtsFallbackEligibleError):
            _ = [chunk async for chunk in stream.chunks]

    asyncio.run(exercise())


def test_synthesize_stream_rejects_once_the_daily_character_quota_is_exhausted() -> None:
    adapter = OpenAiTtsAdapter(client=_fake_client())
    runtime.tts_char_quota_store.record("c1", settings.max_tts_chars_per_client_per_day)

    async def exercise():
        with pytest.raises(TtsFailLoudError, match="quota exceeded"):
            await adapter.synthesize_stream("hi", TtsOptions(client_id="c1"))

    asyncio.run(exercise())
