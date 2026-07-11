"""Policy layer v1 (setara-s94o.9): request validation before any provider is invoked, plus an
in-memory per-client daily usage quota. This is the policy object SttProviderRouter calls
validate_audio()/record_usage() on - Phase 1 wired a no-op stub (NoopSttPolicy in router.py); this
module is the real implementation.

In-memory is intentionally the local-dev/single-instance tier - setara-s94o.14 swaps the storage
backend for Redis, not the validation logic.

Plan reference: asa-local-openai-hosted-mode-plan.md §10 (Policy Layer / Guardrails).
"""
import os
import threading
from datetime import date, datetime, timezone

from app.config import settings
from app.providers.base import IN_MEMORY_AUDIO_MARKER, SttOptions
from app.providers.errors import SttPolicyRejectedError

# Audio formats faster-whisper (via ffmpeg) and OpenAI's transcription API both accept. Rejecting
# anything else here means a garbage upload never reaches a provider at all.
ALLOWED_AUDIO_EXTENSIONS = {"wav", "webm", "mp3", "mp4", "m4a", "ogg", "oga", "flac", "mpga", "mpeg"}


class InMemoryDailyQuotaStore:
    """Per-client daily STT usage counter, in seconds. Resets when the UTC calendar day rolls
    over. Not persisted across restarts and not shared across instances - see class docstring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage: dict[str, tuple[str, float]] = {}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def used_seconds(self, client_id: str) -> float:
        today = self._today()
        with self._lock:
            day, used = self._usage.get(client_id, (today, 0.0))
            return used if day == today else 0.0

    def record(self, client_id: str, seconds: float) -> None:
        if seconds <= 0:
            return
        today = self._today()
        with self._lock:
            day, used = self._usage.get(client_id, (today, 0.0))
            used = used if day == today else 0.0
            self._usage[client_id] = (today, used + seconds)


class RequestValidationPolicy:
    """Validates a request before any STT provider is invoked, and tracks per-client daily usage.
    Reused as the `policy` passed into SttProviderRouter - never bypass it by calling an adapter
    directly."""

    def __init__(self, quota_store: InMemoryDailyQuotaStore | None = None):
        self.quota_store = quota_store or InMemoryDailyQuotaStore()

    def validate_audio(
        self, audio_path: str, options: SttOptions, duration_seconds: float | None
    ) -> None:
        if not options.client_id:
            raise SttPolicyRejectedError(401, "Unknown or missing client")

        if audio_path != IN_MEMORY_AUDIO_MARKER:
            self._validate_file(audio_path)

        if duration_seconds is not None and duration_seconds > settings.max_stt_seconds_per_request:
            raise SttPolicyRejectedError(
                413,
                f"Audio is {duration_seconds:.1f}s; max is "
                f"{settings.max_stt_seconds_per_request}s per request",
            )

        used = self.quota_store.used_seconds(options.client_id)
        if used >= settings.max_stt_seconds_per_client_per_day:
            raise SttPolicyRejectedError(
                429,
                f"Daily STT quota exceeded for client '{options.client_id}' "
                f"({settings.max_stt_seconds_per_client_per_day}s/day)",
            )

    def record_usage(self, options: SttOptions, duration_seconds: float | None) -> None:
        """Increment the client's daily counter. Only call this after a request actually
        completed (success or fallback) - never for a rejected/invalid request."""
        if not options.client_id or duration_seconds is None:
            return
        self.quota_store.record(options.client_id, duration_seconds)

    def _validate_file(self, audio_path: str) -> None:
        suffix = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else ""
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise SttPolicyRejectedError(415, f"Unsupported audio format: .{suffix or 'unknown'}")

        try:
            size = os.path.getsize(audio_path)
        except OSError:
            return  # file already gone/unreadable - let the provider surface the real error
        limit = settings.max_upload_mb * 1024 * 1024
        if size > limit:
            raise SttPolicyRejectedError(413, f"Audio exceeds {settings.max_upload_mb} MB")
