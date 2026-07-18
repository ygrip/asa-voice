"""OpenAiTtsAdapter: OpenAI hosted TTS conforming to the TtsAdapter protocol (setara-nx07.2), plus
error classification into fallback-eligible vs fail-loud (plan §7).

Plan reference: asa-hosted-tts-and-cue-migration-plan.md §7 (Implement OpenAiTtsAdapter).
"""
import time

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.providers.base import TtsAudioMetadata, TtsOptions, TtsResult, TtsStreamResult
from app.providers.errors import TtsFailLoudError, TtsFallbackEligibleError
from app.services import audio_service, voice_catalog
from app.services.operation_limiter import OperationBusyError

# OpenAI's `response_format=pcm` stream is documented as 24kHz mono 16-bit signed little-endian for
# tts-1/tts-1-hd. Not guessed at random - but plan §7 requires this be confirmed by a live
# integration test against the real API before hosted release (tracked under setara-nx07.8).
_STREAM_SAMPLE_RATE_HZ = 24_000

# Only gpt-4o-mini-tts accepts `instructions` (voice-acting direction); tts-1/tts-1-hd (the default
# lane, plan §4) reject the param outright. Never send it to a model that doesn't support it.
_MODELS_SUPPORTING_INSTRUCTIONS = {"gpt-4o-mini-tts"}


def _resolve_voice_ref(voice_id: str | None) -> str:
    try:
        return voice_catalog.resolve_voice_ref(voice_id, "openai", settings.tts_default_voice)
    except voice_catalog.UnknownVoiceError:
        return settings.openai_tts_voice


class OpenAiTtsAdapter:
    """TtsAdapter for OpenAI hosted TTS (`/tts` complete synthesis and `/tts/stream` PCM
    streaming). Construction must succeed even with no API key configured (this adapter can be
    built as an unused hosted/hybrid fallback candidate) - a real request without a valid key then
    fails naturally as AuthenticationError, correctly classified as TtsFailLoudError below."""

    provider_name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=settings.openai_api_key or "sk-not-configured",
            timeout=settings.openai_tts_timeout_seconds,
        )
        self.model = settings.openai_tts_model

    async def synthesize(self, text: str, options: TtsOptions) -> TtsResult:
        from app import runtime

        _validate_text_length(text)
        _check_daily_char_quota(options.client_id)
        voice_ref = _resolve_voice_ref(options.voice_id)
        speed = options.speed if options.speed is not None else settings.openai_tts_speed
        instructions = _resolve_instructions(options.instructions)
        started = time.monotonic()

        async def _call():
            return await self.client.audio.speech.create(
                model=self.model,
                voice=voice_ref,
                input=text,
                response_format=settings.openai_tts_complete_format,
                speed=speed,
                **({"instructions": instructions} if instructions else {}),
            )

        async with runtime.hosted_tts_limiter.slot():
            response = await _call_with_one_retry(_call)
        _record_daily_char_usage(options.client_id, len(text))

        content_type = f"audio/{settings.openai_tts_complete_format}"
        path = audio_service.write_temp(response.content, f".{settings.openai_tts_complete_format}")
        latency_ms = int((time.monotonic() - started) * 1000)
        return TtsResult(
            provider="openai",
            model=self.model,
            audio_path=path,
            content_type=content_type,
            latency_ms=latency_ms,
        )

    async def synthesize_stream(self, text: str, options: TtsOptions) -> TtsStreamResult:
        from app import runtime

        _validate_text_length(text)
        _check_daily_char_quota(options.client_id)
        voice_ref = _resolve_voice_ref(options.voice_id)
        speed = options.speed if options.speed is not None else settings.openai_tts_speed
        instructions = _resolve_instructions(options.instructions)

        async def _chunks():
            # Unlike `synthesize`, no same-provider retry here: a streaming call is only safe to
            # retry before the first byte, and TtsProviderRouter already provides a strictly
            # stronger recovery for that window (an entirely different fallback provider), so a
            # same-provider retry would just be redundant complexity.
            async with runtime.hosted_tts_limiter.slot():
                try:
                    async with self.client.audio.speech.with_streaming_response.create(
                        model=self.model,
                        voice=voice_ref,
                        input=text,
                        response_format=settings.openai_tts_stream_format,
                        speed=speed,
                        **({"instructions": instructions} if instructions else {}),
                    ) as response:
                        # The request was accepted by the provider - count it now rather than
                        # after full drain, matching the "committed once bytes start" boundary the
                        # router itself uses for fallback eligibility.
                        _record_daily_char_usage(options.client_id, len(text))
                        async for chunk in response.iter_bytes():
                            yield chunk
                except OperationBusyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reclassified below, never swallowed
                    raise classify_openai_tts_error(exc) from exc

        metadata = TtsAudioMetadata(
            content_type="audio/l16",
            sample_rate=_STREAM_SAMPLE_RATE_HZ,
            channels=1,
            sample_format="s16le",
            response_format=settings.openai_tts_stream_format,
        )
        return TtsStreamResult(
            provider="openai",
            model=self.model,
            voice_id=options.voice_id,
            metadata=metadata,
            chunks=_chunks(),
        )

    def list_voices(self) -> list[dict]:
        return voice_catalog.list_voices_for_provider("openai", self.model)


def _validate_text_length(text: str) -> None:
    if len(text) > settings.openai_tts_max_text_chars:
        raise TtsFailLoudError(
            f"Text is {len(text)} chars; max is {settings.openai_tts_max_text_chars} per request"
        )


def _resolve_instructions(requested: str | None) -> str | None:
    if settings.openai_tts_model not in _MODELS_SUPPORTING_INSTRUCTIONS:
        return None
    return requested or settings.openai_tts_instructions or None


def _check_daily_char_quota(client_id: str | None) -> None:
    """Reject before any provider call once a client's daily hosted-TTS character budget is
    exhausted (plan §7 cost policy). Not fallback-eligible: quota exhaustion is a policy decision,
    not a provider health signal, so it must surface to the caller rather than silently spend a
    fallback provider's (e.g. local Pocket TTS) resources instead."""
    from app import runtime

    if not client_id:
        return
    used = runtime.tts_char_quota_store.used_seconds(client_id)
    if used >= settings.max_tts_chars_per_client_per_day:
        raise TtsFailLoudError(
            f"Daily hosted TTS character quota exceeded for client '{client_id}' "
            f"({settings.max_tts_chars_per_client_per_day} chars/day)"
        )


def _record_daily_char_usage(client_id: str | None, char_count: int) -> None:
    """Only call after a request actually succeeded - never for a rejected/failed one, and only
    once even if `_call_with_one_retry` retried internally."""
    from app import runtime

    if not client_id:
        return
    runtime.tts_char_quota_store.record(client_id, char_count)


async def _call_with_one_retry(call):
    """At most one same-provider retry, and only for a fallback-eligible-classified failure - a
    cheap transient-blip recovery distinct from TtsProviderRouter's separate-provider fallback
    (plan §7 cost policy: "at most one controlled retry before bytes are emitted")."""
    try:
        return await call()
    except OperationBusyError:
        raise
    except Exception as exc:  # noqa: BLE001 - reclassified below, never swallowed
        classified = classify_openai_tts_error(exc)
        if not isinstance(classified, TtsFallbackEligibleError):
            raise classified from exc
        try:
            return await call()
        except OperationBusyError:
            raise
        except Exception as retry_exc:  # noqa: BLE001 - reclassified below, never swallowed
            raise classify_openai_tts_error(retry_exc) from retry_exc


def classify_openai_tts_error(exc: Exception) -> Exception:
    """Map an openai SDK exception to TtsFallbackEligibleError or TtsFailLoudError (plan §7).

    Fallback-eligible: network timeout, connection error, provider 5xx, transient rate limiting.
    Fail-loud (never falls back silently): invalid API key, billing/permission errors, quota
    exhausted, unsupported voice/model, and any request the provider rejected as malformed.
    """
    if isinstance(exc, (TtsFallbackEligibleError, TtsFailLoudError)):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return TtsFallbackEligibleError(f"OpenAI TTS timed out: {exc}")
    if isinstance(exc, openai.RateLimitError):
        if _is_quota_exhausted(exc):
            return TtsFailLoudError(f"OpenAI quota/billing exhausted: {exc}")
        return TtsFallbackEligibleError(f"OpenAI TTS rate limited: {exc}")
    if isinstance(exc, openai.AuthenticationError):
        return TtsFailLoudError(f"OpenAI authentication failed (check OPENAI_API_KEY): {exc}")
    if isinstance(exc, openai.PermissionDeniedError):
        return TtsFailLoudError(f"OpenAI billing/permission error: {exc}")
    if isinstance(exc, openai.BadRequestError):
        return TtsFailLoudError(f"OpenAI rejected the request (voice/model/text?): {exc}")
    if isinstance(exc, openai.InternalServerError):
        return TtsFallbackEligibleError(f"OpenAI TTS server error: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return TtsFallbackEligibleError(f"OpenAI TTS connection error: {exc}")
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return TtsFallbackEligibleError(str(exc))
        return TtsFailLoudError(str(exc))
    # Unknown error shape: default fail-loud. Silently falling back on an unrecognized error is
    # exactly the failure mode this classification exists to prevent.
    return TtsFailLoudError(str(exc))


def _is_quota_exhausted(exc: "openai.RateLimitError") -> bool:
    """OpenAI reports both transient rate limiting AND exhausted billing quota as HTTP 429
    (RateLimitError) - the error body's `code`/`type` is the only way to tell them apart."""
    code = (getattr(exc, "code", None) or "").lower()
    err_type = (getattr(exc, "type", None) or "").lower()
    return "insufficient_quota" in (code, err_type)
