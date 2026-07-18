"""PocketTtsAdapter: wraps the existing Pocket TTS (Kyutai) TtsService behind the provider-agnostic
TtsAdapter protocol. Synthesis itself (model loading, voice_id resolution) is untouched in
tts_service.py; this module only adapts the result shape and writes the synthesized audio to a
temp file via the shared audio_service boundary, so the future OpenAiTtsAdapter (Phase 6) can
reuse the same file-handoff convention instead of duplicating it per adapter.

Plan reference: asa-local-openai-hosted-mode-plan.md §5.2 (TtsAdapter).
"""
import asyncio
import time

from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.providers.base import TtsAudioMetadata, TtsOptions, TtsResult, TtsStreamResult
from app.services import audio_service
from app.services.tts_service import TtsService, TtsSynthesisError

_STREAM_END = object()  # sentinel: sync generator exhausted


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

    async def synthesize_stream(self, text: str, options: TtsOptions) -> TtsStreamResult:
        from app import runtime

        async def pcm_frames():
            async with runtime.tts_limiter.slot():
                gen = self.service.synthesize_stream(text, options.voice_id)
                while True:
                    try:
                        chunk = await _next_stream_chunk(gen)
                    except TtsSynthesisError:
                        break
                    if chunk is _STREAM_END:
                        break
                    yield chunk

        metadata = TtsAudioMetadata(
            content_type="audio/l16",
            sample_rate=int(self.service.model.sample_rate),
            channels=1,
            sample_format="s16le",
            response_format="pcm",
        )
        return TtsStreamResult(
            provider="pocket_tts",
            model=settings.tts_default_model,
            voice_id=options.voice_id,
            metadata=metadata,
            chunks=pcm_frames(),
        )

    def list_voices(self) -> list[dict]:
        return self.service.list_voices()


async def _next_stream_chunk(generator):
    """Do not release a synthesis lease while its non-cancellable worker is still running."""
    worker = asyncio.create_task(run_in_threadpool(lambda: next(generator, _STREAM_END)))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except BaseException:
            pass
        raise
