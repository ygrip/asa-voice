"""PocketTtsAdapter: wraps the existing Pocket TTS (Kyutai) TtsService behind the provider-agnostic
TtsAdapter protocol. Synthesis itself (model loading, voice_id resolution) is untouched in
tts_service.py; this module only adapts the result shape and writes the synthesized audio to a
temp file via the shared audio_service boundary, so the future OpenAiTtsAdapter (Phase 6) can
reuse the same file-handoff convention instead of duplicating it per adapter.

Plan reference: asa-local-openai-hosted-mode-plan.md §5.2 (TtsAdapter).
"""
import time

from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.providers.base import TtsOptions, TtsResult
from app.services import audio_service
from app.services.tts_service import TtsService


class PocketTtsAdapter:
    """TtsAdapter for the local Pocket TTS engine."""

    provider_name = "pocket_tts"

    def __init__(self, service: TtsService):
        self.service = service

    async def synthesize(self, text: str, options: TtsOptions) -> TtsResult:
        from app import runtime

        started = time.monotonic()
        async with runtime.tts_limiter.slot():
            wav_bytes = await run_in_threadpool(self.service.synthesize, text, options.voice_id)
        path = audio_service.write_temp(wav_bytes, ".wav")
        latency_ms = int((time.monotonic() - started) * 1000)
        return TtsResult(
            provider="pocket_tts",
            model=settings.tts_default_model,
            audio_path=path,
            content_type="audio/wav",
            latency_ms=latency_ms,
        )
