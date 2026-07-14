from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.providers.streaming.base import SttOptions, SttStreamError, SttStreamState
from app.services.stt_context import SttDecodeContext
from app.services.stt_service import collapse_repeats
from app.services.stt_stream_protocol import SttFinalEvent, SttPartialEvent, SttReadyEvent

if TYPE_CHECKING:
    from app.services.stt_service import SttService


STREAM_SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2


class FasterWhisperRollingSession:
    """Provider-owned lifecycle around the existing LocalAgreement rolling transcription."""

    def __init__(self, service: SttService) -> None:
        self._service = service
        self._options: SttOptions | None = None
        self._context: SttDecodeContext | None = None
        self._rolling: _RollingTranscription | None = None
        self._accepting_audio = False
        self._closed = False
        self._finalizing = False
        self._final_emitted = False
        self._received_bytes = 0
        self._received_samples = 0
        self._sequence = 0
        self._capped = False
        self._audio_dropped_ms = 0
        self._finality: str | None = None
        self._audio_cleaned = True
        self._cleanup_count = 0
        self._last_frame_had_speech = False

    @property
    def state(self) -> SttStreamState:
        rolling = self._rolling
        buffered_bytes = 0
        utterance_bytes = 0
        if rolling is not None:
            buffered_bytes = rolling.total_samples * SAMPLE_WIDTH_BYTES
            utterance_bytes = rolling.utterance_samples * SAMPLE_WIDTH_BYTES
        return SttStreamState(
            configured=self._options is not None,
            accepting_audio=self._accepting_audio,
            closed=self._closed,
            final_emitted=self._final_emitted,
            received_bytes=self._received_bytes,
            duration_ms=self._duration_ms(),
            buffered_bytes=buffered_bytes,
            utterance_bytes=utterance_bytes,
            capped=self._capped,
            sequence=self._sequence,
            provider=self._options.provider if self._options else None,
            model=settings.stt_model if self._options else None,
            audio_dropped_ms=self._audio_dropped_ms,
            finality=self._finality,
            cleanup_count=self._cleanup_count,
        )

    async def configure(self, options: SttOptions) -> SttReadyEvent:
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if self._options is not None:
            raise SttStreamError("STT_SESSION_ALREADY_CONFIGURED", "STT session is already configured")
        if options.provider != "faster_whisper":
            raise SttStreamError("STT_STREAM_PROVIDER_UNAVAILABLE", "Expected faster_whisper provider")
        if (
            options.sample_rate != STREAM_SAMPLE_RATE
            or options.channels != 1
            or options.sample_format != "s16le"
        ):
            raise SttStreamError("STT_INVALID_AUDIO_FORMAT", "Expected PCM16 mono 16kHz s16le")
        # Sized from whichever is smaller: the already-negotiated per-mode duration
        # (command=15s/hands_free=30s/dictation=300s via Start.maxDurationSeconds) or the global
        # max_audio_seconds resource guard. Previously this used max_audio_seconds alone (default
        # 20s), which silently truncated every hands_free/dictation session onto the fragile
        # committed-text-plus-tail fallback well before the client's advertised limit (setara-s94o
        # STT quality incident) - max_audio_seconds' default is raised accordingly so it no longer
        # clips any current mode, while still bounding a session against a future/misconfigured
        # max_duration_seconds. The window still self-bounds in the common case regardless:
        # _trim_committed_audio() evicts already-agreed audio after each decode, so retained PCM
        # tracks the unstable tail, not the full utterance - this cap only matters when
        # LocalAgreement never commits (persistently ambiguous/noisy audio).
        max_window_samples = min(options.max_duration_seconds, settings.max_audio_seconds) * options.sample_rate
        if max_window_samples <= 0:
            raise SttStreamError("STT_STREAM_CONFIG_INVALID", "Rolling audio window must be positive")

        self._options = options
        self._context = options.decode_context()
        self._rolling = _RollingTranscription(
            service=self._service,
            context=self._context,
            max_window_samples=max_window_samples,
        )
        self._audio_cleaned = False
        self._accepting_audio = True
        return SttReadyEvent(
            type="ready",
            protocolVersion=options.protocol_version,
            provider=options.provider,
            model=settings.stt_model,
            supportsPartials=True,
            maxDurationSeconds=options.max_duration_seconds,
        )

    async def append_pcm(self, frame: bytes) -> None:
        options, rolling = self._active_session()
        if not frame:
            raise SttStreamError("STT_INVALID_AUDIO_FRAME", "PCM frame must not be empty")
        if len(frame) > options.max_frame_bytes:
            raise SttStreamError(
                "STT_FRAME_TOO_LARGE",
                f"PCM frame exceeds {options.max_frame_bytes} bytes",
            )
        if len(frame) % SAMPLE_WIDTH_BYTES != 0:
            raise SttStreamError(
                "STT_INVALID_AUDIO_FRAME",
                "PCM16 frame must contain complete samples",
            )
        next_bytes = self._received_bytes + len(frame)
        if next_bytes > options.max_session_bytes:
            raise SttStreamError(
                "STT_SESSION_BYTE_LIMIT",
                f"STT session exceeds {options.max_session_bytes} bytes",
            )

        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        next_samples = self._received_samples + samples.size
        if next_samples > options.max_duration_seconds * options.sample_rate:
            raise SttStreamError(
                "STT_SESSION_DURATION_LIMIT",
                f"STT session exceeds {options.max_duration_seconds} seconds",
            )
        # No undecoded-speech queue rejection here (setara-s94o STT recovery, RC-01): command mode
        # deliberately never runs a partial decode, so a counter that only drains on partial decode
        # climbed monotonically and false-rejected valid 15s command audio well before its advertised
        # limit. Optional caption latency must never discard authoritative final audio - duration and
        # byte limits above are the only legitimate ingestion caps.
        frame_had_speech = _contains_speech(samples)

        rolling.append_samples(samples)
        self._received_bytes = next_bytes
        self._received_samples = next_samples
        self._last_frame_had_speech = frame_had_speech
        self._capped = self._capped or rolling.utterance_capped
        self._audio_dropped_ms = rolling.dropped_samples * 1000 // options.sample_rate

    def should_decode(self) -> bool:
        if not self._accepting_audio or self._rolling is None:
            return False
        return self._rolling.should_decode()

    def has_buffered_audio(self) -> bool:
        return self._rolling is not None and self._rolling.has_buffered_audio()

    def is_silent(self) -> bool:
        if self._rolling is None:
            return False
        return self._rolling.is_silent()

    def last_frame_had_speech(self) -> bool:
        return self._last_frame_had_speech

    async def decode_partial(self) -> SttPartialEvent | None:
        options, rolling = self._active_session()
        if not rolling.has_buffered_audio():
            return None
        audio, buffer_offset_seconds = rolling.partial_decode_input()
        words = await run_in_threadpool(self._service.decode_words, audio, self._context)
        result = rolling.apply_partial_decode(words, buffer_offset_seconds)
        self._sequence += 1
        text = collapse_repeats((rolling.committed_text + " " + result["partial"]).strip())
        return SttPartialEvent(
            type="partial",
            sequence=self._sequence,
            text=text,
            committedText=rolling.committed_text,
            unstableText=result["partial"],
            audioReceivedMs=self._received_samples * 1000 // options.sample_rate,
        )

    async def flush(self) -> SttFinalEvent:
        options, rolling = self._configured_session()
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if self._final_emitted or self._finalizing:
            raise SttStreamError("STT_FINAL_ALREADY_EMITTED", "Final transcript was already emitted")

        self._finalizing = True
        self._accepting_audio = False
        started = time.monotonic()
        try:
            text = await run_in_threadpool(rolling.final_text, options.mode)
            finality = "provider_final"
            self._finality = finality
            self._final_emitted = True
            return SttFinalEvent(
                type="final",
                text=text,
                finality=finality,
                provider=options.provider,
                model=settings.stt_model,
                durationMs=self._duration_ms(),
                latencyMs=int((time.monotonic() - started) * 1000),
                fallbackUsed=False,
                audioDroppedMs=self._audio_dropped_ms,
            )
        finally:
            self._cleanup_audio()
            self._finalizing = False

    async def reset(self) -> None:
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        options, _ = self._configured_session()
        self._cleanup_audio()
        self._rolling = _RollingTranscription(
            service=self._service,
            context=self._context,
            max_window_samples=min(options.max_duration_seconds, settings.max_audio_seconds)
            * options.sample_rate,
        )
        self._audio_cleaned = False
        self._accepting_audio = True
        self._finalizing = False
        self._final_emitted = False
        self._received_bytes = 0
        self._received_samples = 0
        self._sequence = 0
        self._capped = False
        self._audio_dropped_ms = 0
        self._finality = None
        self._last_frame_had_speech = False

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting_audio = False
        if not self._final_emitted:
            self._finality = "cancelled"
        self._cleanup_audio()
        self._closed = True

    def _duration_ms(self) -> int:
        if self._options is None:
            return 0
        return self._received_samples * 1000 // self._options.sample_rate

    def _active_session(self) -> tuple[SttOptions, _RollingTranscription]:
        options, rolling = self._configured_session()
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if not self._accepting_audio:
            raise SttStreamError("STT_SESSION_FINALIZED", "STT session is not accepting audio")
        return options, rolling

    def _configured_session(self) -> tuple[SttOptions, _RollingTranscription]:
        if self._options is None or self._rolling is None:
            raise SttStreamError("STT_SESSION_NOT_CONFIGURED", "STT session is not configured")
        return self._options, self._rolling

    def _cleanup_audio(self) -> None:
        if self._audio_cleaned:
            return
        if self._rolling is not None:
            self._rolling.reset()
        self._audio_cleaned = True
        self._cleanup_count += 1


class _RollingTranscription:
    """Extracted LocalAgreement-2 rolling buffer from the original StreamingSttSession."""

    def __init__(
        self,
        service: SttService,
        context: SttDecodeContext | None,
        max_window_samples: int,
    ) -> None:
        self._service = service
        self._context = context
        self._max_window_samples = max_window_samples
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._buffer_offset_seconds = 0.0
        self._committed: list[str] = []
        self._previous: list[dict] = []
        self._agreed_length = 0
        self._samples_since_decode = 0
        self._utterance: deque[np.ndarray] = deque()
        self._utterance_samples = 0
        self._utterance_capped = False
        self._dropped_samples = 0

    @property
    def committed_text(self) -> str:
        return " ".join(self._committed).strip()

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def utterance_samples(self) -> int:
        return self._utterance_samples

    @property
    def utterance_capped(self) -> bool:
        return self._utterance_capped

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    def reset(self) -> None:
        self._chunks.clear()
        self._total_samples = 0
        self._buffer_offset_seconds = 0.0
        self._committed = []
        self._previous = []
        self._agreed_length = 0
        self._samples_since_decode = 0
        self._utterance.clear()
        self._utterance_samples = 0
        self._utterance_capped = False

    def append_samples(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)
        self._total_samples += samples.size
        self._samples_since_decode += samples.size
        self._utterance.append(samples)
        self._utterance_samples += samples.size
        if self._total_samples > self._max_window_samples:
            drop = self._total_samples - self._max_window_samples
            self._drop_rolling_samples(drop)
            self._buffer_offset_seconds += drop / STREAM_SAMPLE_RATE
        if self._utterance_samples > self._max_window_samples:
            self._utterance_capped = True
            drop = self._utterance_samples - self._max_window_samples
            self._drop_utterance_samples(drop)
            self._dropped_samples += drop

    def should_decode(self) -> bool:
        interval_samples = settings.stt_stream_interval_ms * STREAM_SAMPLE_RATE // 1000
        if self._samples_since_decode < interval_samples or self._total_samples == 0:
            return False
        if self.is_silent():
            return False
        return True

    def has_buffered_audio(self) -> bool:
        return self._total_samples > 0

    def is_silent(self) -> bool:
        if settings.stt_stream_energy_threshold <= 0 or self._total_samples == 0:
            return False
        audio = self._rolling_audio()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms < settings.stt_stream_energy_threshold

    def partial_decode_input(self) -> tuple[np.ndarray, float]:
        """Snapshot inference input without holding or mutating the rolling deque in a worker."""
        self._samples_since_decode = 0
        return self._rolling_audio(), self._buffer_offset_seconds

    def apply_partial_decode(
        self,
        words: list[dict],
        buffer_offset_seconds: float,
    ) -> dict:
        """Apply one completed decode on the event loop while ingestion is between frames."""
        for word in words:
            word["start"] += buffer_offset_seconds
            word["end"] += buffer_offset_seconds

        newly_committed = self._commit_agreement(words)
        partial = " ".join(word["w"] for word in words[self._agreed_length :])
        self._previous = words
        self._trim_committed_audio()
        return {"committed": newly_committed, "partial": partial.strip()}

    def final_text(self, mode: str) -> str:
        if not self._utterance_capped:
            audio = self._utterance_audio()
            final = (
                self._service.transcribe_array_final(audio, self._context, mode)["text"]
                if audio.size
                else ""
            )
        else:
            tail_audio = self._rolling_audio()
            tail = (
                self._service.transcribe_array_final(tail_audio, self._context, mode)["text"]
                if tail_audio.size
                else ""
            )
            final = (self.committed_text + " " + tail).strip()
        return collapse_repeats(final)

    def _commit_agreement(self, current: list[dict]) -> str:
        agreed = 0
        while (
            agreed < len(current)
            and agreed < len(self._previous)
            and _normalize_word(current[agreed]["w"]) == _normalize_word(self._previous[agreed]["w"])
        ):
            agreed += 1
        newly_committed = [word["w"] for word in current[self._agreed_length : agreed]]
        self._committed.extend(newly_committed)
        self._agreed_length = agreed
        return " ".join(newly_committed).strip()

    def _trim_committed_audio(self) -> None:
        if self._agreed_length == 0:
            return
        cut_seconds = (
            self._previous[self._agreed_length - 1]["end"] - self._buffer_offset_seconds
        )
        cut_samples = int(max(0.0, cut_seconds) * STREAM_SAMPLE_RATE)
        if cut_samples > 0 and cut_samples < self._total_samples:
            self._drop_rolling_samples(cut_samples)
            self._buffer_offset_seconds += cut_samples / STREAM_SAMPLE_RATE
            self._previous = self._previous[self._agreed_length :]
            self._agreed_length = 0

    def _rolling_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(tuple(self._chunks))

    def _utterance_audio(self) -> np.ndarray:
        if not self._utterance:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(tuple(self._utterance))

    def _drop_utterance_samples(self, count: int) -> None:
        remaining = count
        while remaining > 0 and self._utterance:
            chunk = self._utterance[0]
            if chunk.size <= remaining:
                self._utterance.popleft()
                removed = chunk.size
            else:
                self._utterance[0] = chunk[remaining:]
                removed = remaining
            remaining -= removed
            self._utterance_samples -= removed

    def _drop_rolling_samples(self, count: int) -> None:
        remaining = count
        while remaining > 0 and self._chunks:
            chunk = self._chunks[0]
            if chunk.size <= remaining:
                self._chunks.popleft()
                removed = chunk.size
            else:
                self._chunks[0] = chunk[remaining:]
                removed = remaining
            remaining -= removed
            self._total_samples -= removed


def _contains_speech(samples: np.ndarray) -> bool:
    if samples.size == 0:
        return False
    if settings.stt_stream_energy_threshold <= 0:
        return True
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return rms >= settings.stt_stream_energy_threshold


def _normalize_word(word: str) -> str:
    return "".join(character for character in word.lower() if character.isalnum())
