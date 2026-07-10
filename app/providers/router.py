"""Provider routers: primary + optional fallback selection for STT/TTS. This is the single call
site routers/stt.py and routers/tts.py go through — no router ever calls a provider adapter
directly, so adding a provider (Phase 2 OpenAI, Phase 6 hosted TTS) only means constructing a new
adapter and passing it in here.

Plan reference: asa-local-openai-hosted-mode-plan.md §6 (Provider Router).
"""
from typing import Optional, Protocol

import numpy as np

from app.providers.base import (
    SttAdapter, SttOptions, SttResult, TtsAdapter, TtsOptions, TtsResult,
)


class SttPolicy(Protocol):
    def validate_audio(self, audio_path: str, options: SttOptions) -> None: ...


class NoopSttPolicy:
    """Phase 1 stub — real request validation + quota enforcement lands in setara-s94o.9
    (policy layer v1). Exists now so the router's call site never has to change shape."""

    def validate_audio(self, audio_path: str, options: SttOptions) -> None:
        return None


class SttProviderRouter:
    """Selects a primary STT adapter, falling back to a secondary one (if configured) when the
    primary raises. `fallback` is None in local-only Phase 1 configs — every existing caller works
    unchanged with a single provider; Phase 2 only needs to pass a second adapter in."""

    def __init__(
        self,
        primary: SttAdapter,
        fallback: Optional[SttAdapter] = None,
        policy: Optional[SttPolicy] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.policy = policy or NoopSttPolicy()

    async def transcribe(self, audio_path: str, options: SttOptions) -> SttResult:
        self.policy.validate_audio(audio_path, options)
        try:
            return await self.primary.transcribe(audio_path, options)
        except Exception as primary_error:
            if not self.fallback:
                raise primary_error
            result = await self.fallback.transcribe(audio_path, options)
            result.fallback_used = True
            return result

    async def transcribe_array(self, audio: "np.ndarray", options: SttOptions) -> SttResult:
        """File-free variant for /stt/raw and streaming session flush — local-only fast path;
        no adapter is required to implement it (hosted providers only get `transcribe`)."""
        self.policy.validate_audio("<in-memory>", options)
        try:
            return await self.primary.transcribe_array(audio, options)
        except Exception as primary_error:
            if not self.fallback or not hasattr(self.fallback, "transcribe_array"):
                raise primary_error
            result = await self.fallback.transcribe_array(audio, options)
            result.fallback_used = True
            return result


class TtsProviderRouter:
    """Selects a primary TTS adapter, falling back to a secondary one (if configured) when the
    primary raises."""

    def __init__(self, primary: TtsAdapter, fallback: Optional[TtsAdapter] = None):
        self.primary = primary
        self.fallback = fallback

    async def synthesize(self, text: str, options: TtsOptions) -> TtsResult:
        try:
            return await self.primary.synthesize(text, options)
        except Exception as primary_error:
            if not self.fallback:
                raise primary_error
            return await self.fallback.synthesize(text, options)
