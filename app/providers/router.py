"""Provider routers: primary + optional fallback selection for STT/TTS. This is the single call
site routers/stt.py and routers/tts.py go through — no router ever calls a provider adapter
directly, so adding a provider (Phase 2 OpenAI, Phase 6 hosted TTS) only means constructing a new
adapter and passing it in here.

Plan reference: asa-local-openai-hosted-mode-plan.md §6 (Provider Router).
"""
from typing import AsyncIterator, Optional, Protocol

import numpy as np

from app.providers.base import (
    IN_MEMORY_AUDIO_MARKER, SttAdapter, SttOptions, SttResult, TtsAdapter, TtsOptions, TtsResult,
    TtsStreamResult,
)
from app.providers.errors import SttFallbackEligibleError, TtsFallbackEligibleError
from app.services.operation_limiter import OperationBusyError

STT_STREAM_SAMPLE_RATE = 16000  # PCM16 mono @16kHz streaming contract


class SttPolicy(Protocol):
    def validate_audio(
        self, audio_path: str, options: SttOptions, duration_seconds: Optional[float]
    ) -> None: ...

    def record_usage(self, options: SttOptions, duration_seconds: Optional[float]) -> None: ...


class NoopSttPolicy:
    """Phase 1 stub — real request validation + quota enforcement lands in setara-s94o.9
    (policy layer v1, app/providers/policy.py:RequestValidationPolicy). Exists so the router's
    call site never has to change shape when the real policy is wired in."""

    def validate_audio(
        self, audio_path: str, options: SttOptions, duration_seconds: Optional[float]
    ) -> None:
        return None

    def record_usage(self, options: SttOptions, duration_seconds: Optional[float]) -> None:
        return None


class SttProviderRouter:
    """Selects a primary STT adapter, falling back to a secondary one (if configured) when the
    primary raises SttFallbackEligibleError. `fallback` is None in local-only configs — every
    existing caller works unchanged with a single provider.

    Also supports an explicit per-request provider override (setara-s94o.8) for trusted clients —
    routers/stt.py decides *whether* an override is allowed (trust tier); this class only resolves
    the requested provider name to an adapter, it does not itself enforce any trust policy.
    """

    def __init__(
        self,
        primary: SttAdapter,
        fallback: Optional[SttAdapter] = None,
        policy: Optional[SttPolicy] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.policy = policy or NoopSttPolicy()
        self._by_name: dict[str, SttAdapter] = {}
        for adapter in (primary, fallback):
            if adapter is not None:
                self._by_name[getattr(adapter, "provider_name", "")] = adapter

    def resolve_provider(self, provider_name: str) -> Optional[SttAdapter]:
        """Look up a named adapter for an explicit provider= override. Returns None if the
        provider isn't wired into this router (caller decides how to handle that: ignore the
        override and use the default primary, or reject the request)."""
        return self._by_name.get(provider_name)

    async def transcribe(
        self, audio_path: str, options: SttOptions, provider_override: Optional[str] = None
    ) -> SttResult:
        primary = self.resolve_provider(provider_override) if provider_override else self.primary
        duration_seconds = _probe_file_duration_seconds(audio_path)
        self.policy.validate_audio(audio_path, options, duration_seconds)
        try:
            result = await primary.transcribe(audio_path, options)
        except SttFallbackEligibleError:
            if not self.fallback or self.fallback is primary:
                raise
            result = await self.fallback.transcribe(audio_path, options)
            result.fallback_used = True
        self.policy.record_usage(options, duration_seconds)
        return result

    async def transcribe_array(
        self,
        audio: "np.ndarray",
        options: SttOptions,
        provider_override: Optional[str] = None,
    ) -> SttResult:
        """File-free variant for /stt/raw and streaming session flush — local-only fast path;
        an adapter is not required to implement it (hosted providers only get `transcribe`)."""
        primary = self.resolve_provider(provider_override) if provider_override else self.primary
        duration_seconds = audio.shape[0] / STT_STREAM_SAMPLE_RATE if audio is not None else None
        self.policy.validate_audio(IN_MEMORY_AUDIO_MARKER, options, duration_seconds)
        try:
            result = await primary.transcribe_array(audio, options)
        except SttFallbackEligibleError:
            if not self.fallback or self.fallback is primary or not hasattr(
                self.fallback, "transcribe_array"
            ):
                raise
            result = await self.fallback.transcribe_array(audio, options)
            result.fallback_used = True
        self.policy.record_usage(options, duration_seconds)
        return result


def _probe_file_duration_seconds(audio_path: str) -> Optional[float]:
    if audio_path == IN_MEMORY_AUDIO_MARKER:
        return None
    from app.services import audio_service  # local import: keeps router.py free of a hard,
    # module-level dependency on the ffprobe-backed helper for callers that never touch files.
    return audio_service.probe_duration_seconds(audio_path)


class TtsProviderRouter:
    """Selects a primary TTS adapter, falling back to a secondary one (if configured) when the
    primary raises TtsFallbackEligibleError. Any other exception (including TtsFailLoudError,
    e.g. a bad API key or exhausted billing) propagates immediately — an unclassified TTS failure
    must never be silently masked by a fallback attempt (plan §7 error classification)."""

    def __init__(self, primary: TtsAdapter, fallback: Optional[TtsAdapter] = None):
        self.primary = primary
        self.fallback = fallback
        self._by_name: dict[str, TtsAdapter] = {}
        for adapter in (primary, fallback):
            if adapter is not None:
                self._by_name[getattr(adapter, "provider_name", "")] = adapter

    def resolve_provider(self, provider_name: str) -> Optional[TtsAdapter]:
        return self._by_name.get(provider_name)

    async def synthesize(self, text: str, options: TtsOptions) -> TtsResult:
        try:
            return await self.primary.synthesize(text, options)
        except OperationBusyError:
            raise
        except TtsFallbackEligibleError as primary_error:
            if not self.fallback or self.fallback is self.primary:
                raise
            return await self.fallback.synthesize(text, options)

    async def synthesize_stream(self, text: str, options: TtsOptions) -> TtsStreamResult:
        """Fallback is only allowed before the first audio chunk is produced — once bytes are
        committed to the caller, a mid-stream failure propagates instead of switching providers
        (plan §6.5: never concatenate two provider voices in one response)."""
        try:
            return await self._synthesize_stream_primed(self.primary, text, options)
        except OperationBusyError:
            raise
        except TtsFallbackEligibleError as primary_error:
            if not self.fallback or self.fallback is self.primary:
                raise
            return await self._synthesize_stream_primed(self.fallback, text, options)

    @staticmethod
    async def _synthesize_stream_primed(
        adapter: TtsAdapter, text: str, options: TtsOptions
    ) -> TtsStreamResult:
        """Fetch the adapter's stream result and force its first chunk now, so any failure that
        happens before real audio bytes exist still raises here — where the caller above can still
        fall back — instead of surfacing once the client is already mid-stream."""
        result = await adapter.synthesize_stream(text, options)
        result.chunks = await _prime_first_chunk(result.chunks)
        return result


async def _prime_first_chunk(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    iterator = chunks.__aiter__()
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        first = None
    return _PrimedAsyncIterator(iterator, first)


class _PrimedAsyncIterator:
    """Replays an already-fetched `first_chunk` before resuming `rest`. `aclose()` always forwards
    to `rest` — even if this wrapper's `__anext__` was never called — so a caller that discards an
    unconsumed stream still deterministically triggers `rest`'s own cleanup (e.g. releasing a
    concurrency lease held by an adapter's generator). A plain wrapper generator can't guarantee
    this: aclose() on a generator that never started running never executes its body/finally, and
    would instead depend on the event loop's async-generator GC finalizer to clean up `rest`."""

    def __init__(self, rest: AsyncIterator[bytes], first_chunk: Optional[bytes]):
        self._rest = rest
        self._first_chunk = first_chunk
        self._first_consumed = first_chunk is None

    def __aiter__(self) -> "AsyncIterator[bytes]":
        return self

    async def __anext__(self) -> bytes:
        if not self._first_consumed:
            self._first_consumed = True
            return self._first_chunk
        return await self._rest.__anext__()

    async def aclose(self) -> None:
        await self._rest.aclose()
