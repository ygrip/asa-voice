from app import runtime
from app.config import settings
from app.providers.streaming.base import SttOptions, SttStreamError, StreamingSttSession
from app.services.stt_stream_protocol import SttStartControl


class StreamingSttSessionFactory:
    def create(self, start: SttStartControl, client_id: str) -> tuple[StreamingSttSession, SttOptions]:
        provider = settings.stt_provider if start.provider == "auto" else start.provider
        options = self._options(start, client_id, provider)
        if provider == "faster_whisper":
            if runtime.stt_service is None or not runtime.has_stt_adapter("faster_whisper"):
                raise SttStreamError(
                    "STT_STREAM_PROVIDER_UNAVAILABLE",
                    "Faster-whisper streaming is not ready",
                    retryable=True,
                )
            from app.providers.streaming.faster_whisper_session import FasterWhisperRollingSession

            return FasterWhisperRollingSession(runtime.stt_service), options
        if provider == "openai":
            if runtime.stt_router is None or not runtime.has_stt_adapter("openai"):
                raise SttStreamError(
                    "STT_STREAM_PROVIDER_UNAVAILABLE",
                    "OpenAI buffered streaming is not ready",
                    retryable=True,
                )
            if not runtime.hosted_stt_config_usable():
                raise SttStreamError(
                    "STT_STREAM_PROVIDER_UNAVAILABLE",
                    "OpenAI STT configuration is incomplete",
                    retryable=False,
                )
            from app.providers.streaming.openai_buffered_session import OpenAiBufferedFileSession

            return OpenAiBufferedFileSession(runtime.stt_router), options
        raise SttStreamError(
            "STT_STREAM_PROVIDER_UNAVAILABLE",
            f"Streaming STT provider {provider} is unsupported",
            retryable=False,
        )

    def _options(self, start: SttStartControl, client_id: str, provider: str) -> SttOptions:
        if start.protocol_version != settings.stt_stream_protocol_version:
            raise SttStreamError(
                "STT_PROTOCOL_VERSION_UNSUPPORTED",
                f"Expected STT protocol version {settings.stt_stream_protocol_version}",
            )
        if start.audio.frame_duration_ms != settings.stt_frame_duration_ms:
            raise SttStreamError(
                "STT_INVALID_AUDIO_FORMAT",
                f"Expected {settings.stt_frame_duration_ms} ms PCM frames",
            )
        mode_limit = _mode_duration_limit(start.mode)
        if start.max_duration_seconds > mode_limit:
            raise SttStreamError(
                "STT_SESSION_DURATION_LIMIT",
                f"{start.mode} streams are limited to {mode_limit} seconds",
                retryable=False,
            )
        if settings.stt_stream_max_frame_bytes <= 0:
            raise SttStreamError("STT_STREAM_CONFIG_INVALID", "Frame byte limit must be positive")
        if settings.stt_stream_max_session_bytes <= 0:
            raise SttStreamError("STT_STREAM_CONFIG_INVALID", "Session byte limit must be positive")
        if settings.stt_stream_queue_max_ms <= 0:
            raise SttStreamError("STT_STREAM_CONFIG_INVALID", "Queue limit must be positive")
        if (
            settings.stt_stream_interval_ms <= 0
            or settings.stt_stream_max_partial_interval_ms < settings.stt_stream_interval_ms
            or settings.stt_stream_rtf_slow_threshold <= 0
        ):
            raise SttStreamError(
                "STT_STREAM_CONFIG_INVALID",
                "Partial cadence and RTF threshold must be positive and ordered",
            )
        return SttOptions(
            protocol_version=start.protocol_version,
            session_id=start.session_id,
            request_id=start.request_id,
            client_id=client_id,
            mode=start.mode,
            provider=provider,
            sample_rate=start.audio.sample_rate,
            channels=start.audio.channels,
            sample_format=start.audio.sample_format,
            frame_duration_ms=start.audio.frame_duration_ms,
            language=start.language,
            prompt=start.prompt,
            hotwords=tuple(start.hotwords),
            max_duration_seconds=start.max_duration_seconds,
            max_frame_bytes=settings.stt_stream_max_frame_bytes,
            max_session_bytes=settings.stt_stream_max_session_bytes,
            max_queue_ms=settings.stt_stream_queue_max_ms,
        )


def _mode_duration_limit(mode: str) -> int:
    if mode == "command":
        return settings.stt_command_max_seconds
    if mode == "hands_free":
        return settings.stt_handsfree_max_seconds
    if mode == "dictation":
        return settings.stt_dictation_max_seconds
    raise SttStreamError("STT_INVALID_MESSAGE", f"Unsupported STT mode {mode}")
