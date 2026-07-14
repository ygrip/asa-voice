"""Unit tests for OpenAiSttAdapter + OpenAI error classification (setara-s94o.6/.7)."""
import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.providers.base import SttOptions
from app.providers.errors import SttFailLoudError, SttFallbackEligibleError
from app.providers.openai_stt import OpenAiSttAdapter, classify_openai_stt_error, _is_prompt_echo

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


def _status_error(cls, status_code: int, body=None):
    response = httpx.Response(status_code, request=_REQUEST)
    return cls(f"error {status_code}", response=response, body=body)


# --- error classification -----------------------------------------------------------------


def test_timeout_is_fallback_eligible() -> None:
    exc = openai.APITimeoutError(request=_REQUEST)
    assert isinstance(classify_openai_stt_error(exc), SttFallbackEligibleError)


def test_connection_error_is_fallback_eligible() -> None:
    exc = openai.APIConnectionError(request=_REQUEST)
    assert isinstance(classify_openai_stt_error(exc), SttFallbackEligibleError)


def test_server_error_is_fallback_eligible() -> None:
    exc = _status_error(openai.InternalServerError, 500)
    assert isinstance(classify_openai_stt_error(exc), SttFallbackEligibleError)


def test_plain_rate_limit_is_fallback_eligible() -> None:
    exc = _status_error(openai.RateLimitError, 429, body={"code": "rate_limit_exceeded", "type": "requests"})
    assert isinstance(classify_openai_stt_error(exc), SttFallbackEligibleError)


def test_quota_exhausted_rate_limit_is_fail_loud() -> None:
    exc = _status_error(
        openai.RateLimitError, 429, body={"code": "insufficient_quota", "type": "insufficient_quota"}
    )
    assert isinstance(classify_openai_stt_error(exc), SttFailLoudError)


def test_invalid_api_key_is_fail_loud() -> None:
    exc = _status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"})
    assert isinstance(classify_openai_stt_error(exc), SttFailLoudError)


def test_permission_denied_is_fail_loud() -> None:
    exc = _status_error(openai.PermissionDeniedError, 403)
    assert isinstance(classify_openai_stt_error(exc), SttFailLoudError)


def test_bad_request_unsupported_format_is_fail_loud() -> None:
    exc = _status_error(openai.BadRequestError, 400)
    assert isinstance(classify_openai_stt_error(exc), SttFailLoudError)


def test_unknown_error_defaults_fail_loud() -> None:
    # Never silently fall back on an unrecognized error shape.
    assert isinstance(classify_openai_stt_error(ValueError("weird")), SttFailLoudError)


# --- adapter normalization -----------------------------------------------------------------


class _FakeTranscriptions:
    def __init__(self, text: str = "hello world", capture: dict | None = None, error: Exception | None = None):
        self._text = text
        self._capture = capture
        self._error = error

    async def create(self, **kwargs):
        if self._capture is not None:
            self._capture.update(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(text=self._text)


def _fake_client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=_FakeTranscriptions(**kwargs)))


def test_adapter_normalizes_result(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(text="create a release plan"))

    result = asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))

    assert result.provider == "openai"
    assert result.text == "create a release plan"
    assert result.latency_ms >= 0
    assert result.fallback_used is False


def test_adapter_uses_default_domain_prompt_when_options_prompt_missing(tmp_path) -> None:
    from app.config import settings

    captured: dict = {}
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(capture=captured))

    asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))

    assert captured["prompt"] == settings.openai_stt_prompt


def test_adapter_prefers_explicit_prompt_over_default(tmp_path) -> None:
    captured: dict = {}
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(capture=captured))

    asyncio.run(
        adapter.transcribe(str(audio_path), SttOptions(client_id="test", prompt="custom prompt"))
    )

    assert captured["prompt"] == "custom prompt"


def test_adapter_raises_fallback_eligible_on_timeout(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(error=openai.APITimeoutError(request=_REQUEST)))

    with pytest.raises(SttFallbackEligibleError):
        asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))


def test_adapter_raises_fail_loud_on_bad_api_key(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    error = _status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"})
    adapter = OpenAiSttAdapter(client=_fake_client(error=error))

    with pytest.raises(SttFailLoudError):
        asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))


# --- prompt-echo hallucination guard --------------------------------------------------------
# gpt-4o-mini-transcribe (and other Whisper-family hosted STT) can hallucinate the injected
# vocabulary-bias prompt back as the transcript on short/quiet audio - a real report: the user
# said "yes" and got back the full OPENAI_STT_PROMPT sentence verbatim.


def test_is_prompt_echo_detects_verbatim_match() -> None:
    prompt = "Common product terms: Setara, Raksara, scenario, test case."
    assert _is_prompt_echo(prompt, prompt) is True


def test_is_prompt_echo_ignores_case_and_punctuation() -> None:
    prompt = "Common product terms: Setara, Raksara."
    echoed = "common product terms setara raksara"
    assert _is_prompt_echo(echoed, prompt) is True


def test_is_prompt_echo_is_false_for_a_real_short_answer() -> None:
    prompt = "Common product terms: Setara, Raksara, scenario, test case."
    assert _is_prompt_echo("yes", prompt) is False


def test_adapter_blanks_out_a_hallucinated_prompt_echo(tmp_path) -> None:
    from app.config import settings

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(text=settings.openai_stt_prompt))

    result = asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))

    assert result.text == ""


def test_adapter_keeps_a_real_transcript_that_happens_to_share_a_word(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    adapter = OpenAiSttAdapter(client=_fake_client(text="show me the scenario coverage"))

    result = asyncio.run(adapter.transcribe(str(audio_path), SttOptions(client_id="test")))

    assert result.text == "show me the scenario coverage"
