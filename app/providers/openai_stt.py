"""OpenAiSttAdapter: OpenAI hosted STT conforming to the SttAdapter protocol (setara-s94o.6), plus
error classification into fallback-eligible vs fail-loud (setara-s94o.7).

Plan reference: asa-local-openai-hosted-mode-plan.md §8 (OpenAI Hosted STT Implementation).
"""
import re
import time

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.providers.base import SttOptions, SttResult
from app.providers.errors import SttFailLoudError, SttFallbackEligibleError
from app.services.operation_limiter import OperationBusyError


def _normalize_for_echo_check(text: str) -> str:
    collapsed = re.sub(r"[^\w\s]+", " ", text.strip().lower())
    return re.sub(r"\s+", " ", collapsed).strip()


def _is_prompt_echo(text: str, prompt: str) -> bool:
    """Whisper-family hosted STT (gpt-4o-mini-transcribe included) can hallucinate the injected
    vocabulary-bias `prompt` back as the transcript on short/quiet/ambiguous audio - a known
    failure mode, not a real utterance. A one-word "yes" should never coincidentally produce our
    entire hotwords sentence verbatim."""
    if not text or not prompt:
        return False
    normalized_text = _normalize_for_echo_check(text)
    normalized_prompt = _normalize_for_echo_check(prompt)
    return normalized_text == normalized_prompt or normalized_prompt in normalized_text


class OpenAiSttAdapter:
    """SttAdapter for OpenAI hosted STT. gpt-4o-mini-transcribe (and gpt-4o-transcribe) only
    support the `json`/`text` response formats — no segment timestamps or audio duration are
    available from the API for these models, so `duration_ms` is always 0 here; callers that need
    audio duration (e.g. the policy layer's quota accounting) derive it from the input audio
    instead of trusting the provider's response."""

    provider_name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None):
        # The openai SDK raises immediately if api_key is falsy and OPENAI_API_KEY isn't set in
        # the environment. Construction must succeed even with no key configured (this adapter
        # can be built as an unused hosted/hybrid fallback candidate, or the key may only be
        # configured later) — a real request without a valid key then fails naturally as
        # AuthenticationError, correctly classified as SttFailLoudError below.
        self.client = client or AsyncOpenAI(
            api_key=settings.openai_api_key or "sk-not-configured",
            timeout=settings.openai_stt_timeout_seconds,
        )
        self.model = settings.openai_stt_model

    async def transcribe(self, audio_path: str, options: SttOptions) -> SttResult:
        from app import runtime

        started = time.monotonic()
        prompt = options.prompt or settings.openai_stt_prompt
        try:
            async with runtime.hosted_request_limiter.slot():
                with open(audio_path, "rb") as audio_file:
                    result = await self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=options.language,
                        prompt=prompt,
                    )
        except OperationBusyError:
            raise
        except Exception as exc:  # noqa: BLE001 - reclassified below, never swallowed
            raise classify_openai_stt_error(exc) from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        text = "" if _is_prompt_echo(result.text, prompt) else result.text
        return SttResult(
            provider="openai",
            model=self.model,
            text=text,
            language=options.language,
            duration_ms=0,
            latency_ms=latency_ms,
            segments=[],
            fallback_used=False,
        )


def classify_openai_stt_error(exc: Exception) -> Exception:
    """Map an openai SDK exception to SttFallbackEligibleError or SttFailLoudError (plan §8.3).

    Fallback-eligible: network timeout, connection error, provider 5xx, transient rate limiting.
    Fail-loud (never falls back silently): invalid API key, billing/permission errors, quota
    exhausted, and any request the provider rejected as malformed (bad format, too large, etc).
    """
    if isinstance(exc, openai.APITimeoutError):
        return SttFallbackEligibleError(f"OpenAI STT timed out: {exc}")
    if isinstance(exc, openai.RateLimitError):
        if _is_quota_exhausted(exc):
            return SttFailLoudError(f"OpenAI quota/billing exhausted: {exc}")
        return SttFallbackEligibleError(f"OpenAI STT rate limited: {exc}")
    if isinstance(exc, openai.AuthenticationError):
        return SttFailLoudError(f"OpenAI authentication failed (check OPENAI_API_KEY): {exc}")
    if isinstance(exc, openai.PermissionDeniedError):
        return SttFailLoudError(f"OpenAI billing/permission error: {exc}")
    if isinstance(exc, openai.BadRequestError):
        return SttFailLoudError(f"OpenAI rejected the request (format/size?): {exc}")
    if isinstance(exc, openai.InternalServerError):
        return SttFallbackEligibleError(f"OpenAI STT server error: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return SttFallbackEligibleError(f"OpenAI STT connection error: {exc}")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return SttFallbackEligibleError(str(exc))
        return SttFailLoudError(str(exc))
    # Unknown error shape: default fail-loud. Silently falling back on an unrecognized error is
    # exactly the failure mode this classification exists to prevent.
    return SttFailLoudError(str(exc))


def _is_quota_exhausted(exc: "openai.RateLimitError") -> bool:
    """OpenAI reports both transient rate limiting AND exhausted billing quota as HTTP 429
    (RateLimitError) - the error body's `code`/`type` is the only way to tell them apart."""
    code = (getattr(exc, "code", None) or "").lower()
    err_type = (getattr(exc, "type", None) or "").lower()
    return "insufficient_quota" in (code, err_type)
