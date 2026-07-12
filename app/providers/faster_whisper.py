"""FasterWhisperAdapter: wraps the existing faster-whisper SttService behind the provider-agnostic
SttAdapter protocol. This is plumbing only — VAD filtering, segment handling, and the rolling
streaming/buffer machinery in app/services/stt_service.py are untouched; this module only adapts
the result shape.

Plan reference: asa-local-openai-hosted-mode-plan.md §9.1 (Faster Whisper Adapter).
"""
import time

import numpy as np
from fastapi.concurrency import run_in_threadpool

from app.providers.base import SttOptions, SttResult, SttSegment
from app.services.stt_context import SttDecodeContext
from app.services.stt_service import SttService


class FasterWhisperAdapter:
    """SttAdapter for the local faster-whisper engine. Holds the (expensive to load) SttService
    instance and exposes both the protocol's file-based `transcribe()` and an additional
    `transcribe_array()` for the file-free in-memory path used by /stt/raw and streaming — a local
    -only fast path that hosted providers (which require an uploaded file) cannot offer."""

    provider_name = "faster_whisper"

    def __init__(self, service: SttService):
        self.service = service

    async def transcribe(self, audio_path: str, options: SttOptions) -> SttResult:
        from app import runtime

        context = _context_from_options(options)
        started = time.monotonic()
        async with runtime.local_decode_limiter.slot():
            result = await run_in_threadpool(
                self.service.transcribe, audio_path, options.language, None, context
            )
        return _to_stt_result(result, started)

    async def transcribe_array(self, audio: np.ndarray, options: SttOptions) -> SttResult:
        from app import runtime

        context = _context_from_options(options)
        started = time.monotonic()
        async with runtime.local_decode_limiter.slot():
            result = await run_in_threadpool(self.service.transcribe_array_final, audio, context)
        return _to_stt_result(result, started)


def _context_from_options(options: SttOptions) -> SttDecodeContext | None:
    if not (options.prompt or options.hotwords or options.request_id):
        return None
    return SttDecodeContext(
        language=options.language,
        prompt=options.prompt,
        hotwords=options.hotwords or [],
        request_id=options.request_id,
    )


def _to_stt_result(result: dict, started: float) -> SttResult:
    latency_ms = int((time.monotonic() - started) * 1000)
    segments = [
        SttSegment(start=seg["start"], end=seg["end"], text=seg["text"]) for seg in result["segments"]
    ]
    return SttResult(
        provider="faster_whisper",
        model=result["model"],
        text=result["text"],
        language=result.get("language"),
        duration_ms=int(result.get("durationSeconds", 0) * 1000),
        latency_ms=latency_ms,
        segments=segments,
        fallback_used=False,
    )
