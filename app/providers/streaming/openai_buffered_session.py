from __future__ import annotations

import asyncio
import time

from app.config import settings
from app.providers.base import SttOptions as BatchSttOptions
from app.providers.errors import (
    SttFailLoudError,
    SttFallbackEligibleError,
    SttPolicyRejectedError,
)
from app.providers.router import SttProviderRouter
from app.providers.streaming.base import SttOptions, SttStreamError, SttStreamState
from app.services.pcm_wav_writer import (
    PCM16_SAMPLE_WIDTH_BYTES,
    IncrementalPcmWavWriter,
    TempAudioLimitExceeded,
)
from app.services.stt_stream_protocol import SttFinalEvent, SttPartialEvent, SttReadyEvent
from app.services.voice_metrics import voice_metrics


STREAM_SAMPLE_RATE = 16_000


class OpenAiBufferedFileSession:
    """Disk-backed hosted stream that uploads one WAV only when the client flushes."""

    def __init__(self, router: SttProviderRouter) -> None:
        self._router = router
        self._options: SttOptions | None = None
        self._writer: IncrementalPcmWavWriter | None = None
        self._accepting_audio = False
        self._closed = False
        self._finalizing = False
        self._final_emitted = False
        self._received_bytes = 0
        self._received_samples = 0
        self._finality: str | None = None
        self._cleanup_count = 0
        self._metric_file_active = False
        self._metric_file_bytes = 0

    @property
    def state(self) -> SttStreamState:
        return SttStreamState(
            configured=self._options is not None,
            accepting_audio=self._accepting_audio,
            closed=self._closed,
            final_emitted=self._final_emitted,
            received_bytes=self._received_bytes,
            duration_ms=self._duration_ms(),
            buffered_bytes=0,
            utterance_bytes=0,
            capped=False,
            sequence=0,
            provider=self._options.provider if self._options else None,
            model=settings.openai_stt_model if self._options else None,
            audio_dropped_ms=0,
            finality=self._finality,
            cleanup_count=self._cleanup_count,
        )

    async def configure(self, options: SttOptions) -> SttReadyEvent:
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if self._options is not None:
            raise SttStreamError("STT_SESSION_ALREADY_CONFIGURED", "STT session is already configured")
        if options.provider != "openai":
            raise SttStreamError("STT_STREAM_PROVIDER_UNAVAILABLE", "Expected openai provider")
        if (
            options.sample_rate != STREAM_SAMPLE_RATE
            or options.channels != 1
            or options.sample_format != "s16le"
        ):
            raise SttStreamError("STT_INVALID_AUDIO_FORMAT", "Expected PCM16 mono 16kHz s16le")
        if settings.openai_stt_max_temp_bytes <= 44:
            raise SttStreamError("STT_STREAM_CONFIG_INVALID", "OpenAI STT temp byte limit is invalid")
        try:
            writer = self._new_writer(options)
        except Exception as exc:
            raise SttStreamError(
                "STT_TEMP_DIRECTORY_UNAVAILABLE",
                "OpenAI STT temporary storage is unavailable",
                retryable=False,
            ) from exc
        self._options = options
        self._writer = writer
        self._track_writer_created()
        self._accepting_audio = True
        return SttReadyEvent(
            type="ready",
            protocolVersion=options.protocol_version,
            provider=options.provider,
            model=settings.openai_stt_model,
            supportsPartials=False,
            maxDurationSeconds=options.max_duration_seconds,
        )

    async def append_pcm(self, frame: bytes) -> None:
        options, writer = self._active_session()
        try:
            if not frame:
                raise SttStreamError("STT_INVALID_AUDIO_FRAME", "PCM frame must not be empty")
            if len(frame) > options.max_frame_bytes:
                raise SttStreamError(
                    "STT_FRAME_TOO_LARGE",
                    f"PCM frame exceeds {options.max_frame_bytes} bytes",
                )
            if len(frame) % PCM16_SAMPLE_WIDTH_BYTES != 0:
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
            next_samples = self._received_samples + len(frame) // PCM16_SAMPLE_WIDTH_BYTES
            if next_samples > options.max_duration_seconds * options.sample_rate:
                raise SttStreamError(
                    "STT_SESSION_DURATION_LIMIT",
                    f"STT session exceeds {options.max_duration_seconds} seconds",
                )
            writer.append_pcm(frame)
            self._metric_file_bytes += len(frame)
            voice_metrics.add_gauge("asa_voice_stt_temp_file_bytes", len(frame))
            self._received_bytes = next_bytes
            self._received_samples = next_samples
        except TempAudioLimitExceeded as exc:
            self._fail_and_cleanup()
            raise SttStreamError("STT_TEMP_FILE_LIMIT", str(exc), retryable=False) from exc
        except SttStreamError:
            self._fail_and_cleanup()
            raise
        except Exception as exc:
            self._fail_and_cleanup()
            raise SttStreamError(
                "STT_TEMP_FILE_WRITE_FAILED",
                "Unable to persist streaming STT audio",
                retryable=True,
            ) from exc

    def should_decode(self) -> bool:
        return False

    def has_buffered_audio(self) -> bool:
        return self._received_bytes > 0

    def is_silent(self) -> bool:
        return False

    def last_frame_had_speech(self) -> bool:
        # Hosted buffered sessions intentionally do not run local VAD or partial inference.
        return True

    async def decode_partial(self) -> SttPartialEvent | None:
        return None

    async def flush(self) -> SttFinalEvent:
        options = self._configured_options()
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if self._final_emitted or self._finalizing:
            raise SttStreamError("STT_FINAL_ALREADY_EMITTED", "Final transcript was already emitted")
        if self._writer is None:
            raise SttStreamError("STT_SESSION_FINALIZED", "STT session is not accepting audio")
        writer = self._writer
        if self._received_bytes == 0:
            self._fail_and_cleanup()
            raise SttStreamError("STT_EMPTY_AUDIO", "Cannot finalize an empty STT session")

        self._finalizing = True
        self._accepting_audio = False
        started = time.monotonic()
        try:
            audio_path = writer.finalize()
            result = await asyncio.wait_for(
                self._router.transcribe(
                    str(audio_path),
                    BatchSttOptions(
                        language=options.language,
                        prompt=options.prompt,
                        hotwords=list(options.hotwords),
                        request_id=options.request_id,
                        client_id=options.client_id,
                    ),
                    provider_override="openai",
                ),
                timeout=settings.openai_stt_timeout_seconds,
            )
            self._finality = "provider_final"
            self._final_emitted = True
            return SttFinalEvent(
                type="final",
                text=result.text,
                finality="provider_final",
                provider=result.provider,
                model=result.model,
                durationMs=self._duration_ms(),
                latencyMs=result.latency_ms or int((time.monotonic() - started) * 1000),
                fallbackUsed=result.fallback_used,
                audioDroppedMs=0,
            )
        except asyncio.TimeoutError as exc:
            raise SttStreamError(
                "STT_FINAL_TIMEOUT",
                "Hosted STT finalization timed out",
                retryable=True,
            ) from exc
        except SttFailLoudError as exc:
            raise SttStreamError("STT_PROVIDER_REJECTED", str(exc), retryable=False) from exc
        except SttFallbackEligibleError as exc:
            raise SttStreamError("STT_PROVIDER_UNAVAILABLE", str(exc), retryable=True) from exc
        except SttPolicyRejectedError as exc:
            raise SttStreamError("STT_POLICY_REJECTED", exc.detail, retryable=False) from exc
        finally:
            self._cleanup_audio()
            self._finalizing = False

    async def reset(self) -> None:
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        options = self._configured_options()
        self._cleanup_audio()
        try:
            self._writer = self._new_writer(options)
            self._track_writer_created()
        except Exception as exc:
            self._accepting_audio = False
            raise SttStreamError(
                "STT_TEMP_DIRECTORY_UNAVAILABLE",
                "OpenAI STT temporary storage is unavailable",
                retryable=False,
            ) from exc
        self._accepting_audio = True
        self._finalizing = False
        self._final_emitted = False
        self._received_bytes = 0
        self._received_samples = 0
        self._finality = None

    async def close(self) -> None:
        if self._closed:
            return
        self._accepting_audio = False
        if not self._final_emitted:
            self._finality = "cancelled"
        self._cleanup_audio()
        self._closed = True

    def _new_writer(self, options: SttOptions) -> IncrementalPcmWavWriter:
        return IncrementalPcmWavWriter(
            settings.openai_stt_buffer_directory,
            sample_rate=options.sample_rate,
            channels=options.channels,
            max_file_bytes=settings.openai_stt_max_temp_bytes,
        )

    def _duration_ms(self) -> int:
        if self._options is None:
            return 0
        return self._received_samples * 1000 // self._options.sample_rate

    def _active_session(self) -> tuple[SttOptions, IncrementalPcmWavWriter]:
        options, writer = self._configured_session()
        if self._closed:
            raise SttStreamError("STT_SESSION_CLOSED", "STT session is closed")
        if not self._accepting_audio:
            raise SttStreamError("STT_SESSION_FINALIZED", "STT session is not accepting audio")
        return options, writer

    def _configured_session(self) -> tuple[SttOptions, IncrementalPcmWavWriter]:
        options = self._configured_options()
        if self._writer is None:
            raise SttStreamError("STT_SESSION_NOT_CONFIGURED", "STT session is not configured")
        return options, self._writer

    def _configured_options(self) -> SttOptions:
        if self._options is None:
            raise SttStreamError("STT_SESSION_NOT_CONFIGURED", "STT session is not configured")
        return self._options

    def _fail_and_cleanup(self) -> None:
        self._accepting_audio = False
        self._cleanup_audio()

    def _cleanup_audio(self) -> None:
        if self._writer is None:
            return
        self._writer.close_and_delete()
        self._writer = None
        if self._metric_file_active:
            voice_metrics.add_gauge("asa_voice_stt_temp_files_active", -1)
            voice_metrics.add_gauge(
                "asa_voice_stt_temp_file_bytes", -self._metric_file_bytes
            )
            self._metric_file_active = False
            self._metric_file_bytes = 0
        self._cleanup_count += 1

    def _track_writer_created(self) -> None:
        self._metric_file_active = True
        self._metric_file_bytes = 0
        voice_metrics.add_gauge("asa_voice_stt_temp_files_active", 1)
