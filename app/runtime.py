"""Shared singletons + concurrency gates, initialized at app startup (see main.lifespan)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import settings
from app.providers.policy import InMemoryDailyQuotaStore, RequestValidationPolicy
from app.providers.router import SttProviderRouter, TtsProviderRouter
from app.services.cue_service import CueService
from app.services.operation_limiter import OperationLimiter

if TYPE_CHECKING:
    from app.services.stt_service import SttService
    from app.services.tts_service import TtsService

# Engine singletons — kept directly for the streaming session (rolling-window WS decode) and the
# /tts/stream + voice-catalog paths, which sit outside the provider-router abstraction (Phase 1 only
# has one provider per role; Phase 2's OpenAI realtime STT is a distinct code path, see setara-s94o.18).
stt_service: SttService | None = None
tts_service: TtsService | None = None

# The single call site routers/stt.py and routers/tts.py go through for provider selection.
stt_router: SttProviderRouter | None = None
tts_router: TtsProviderRouter | None = None

# Shared for the process lifetime so the daily quota counter (setara-s94o.9) actually accumulates
# across requests instead of resetting every time build_routers() runs.
stt_policy = RequestValidationPolicy()

# Hosted TTS daily character quota (setara-nx07.2 / plan §7 cost policy). Reuses
# InMemoryDailyQuotaStore's day-rollover counter for characters instead of seconds - the mechanism
# is unit-agnostic despite the STT-era method name (used_seconds).
tts_char_quota_store = InMemoryDailyQuotaStore()

# Shared for the process lifetime so the in-memory/disk cue cache tiers (setara-nx07.3) persist
# across requests instead of resetting every time build_routers() runs.
cue_service = CueService()


def _build_operation_limiters() -> tuple[
    OperationLimiter, OperationLimiter, OperationLimiter, OperationLimiter
]:
    return (
        OperationLimiter(
            settings.local_stt_max_concurrent
            if settings.local_stt_max_concurrent is not None
            else settings.max_concurrent_stt,
            busy_code="STT_LOCAL_BUSY",
            busy_message="Local STT decode is busy - retry shortly",
        ),
        OperationLimiter(
            settings.hosted_stt_max_concurrent,
            busy_code="STT_HOSTED_BUSY",
            busy_message="Hosted STT request limit reached - retry shortly",
        ),
        OperationLimiter(
            settings.tts_max_concurrent
            if settings.tts_max_concurrent is not None
            else settings.max_concurrent_tts,
            busy_code="TTS_BUSY",
            busy_message="TTS synthesis is busy - retry shortly",
        ),
        OperationLimiter(
            settings.hosted_tts_max_concurrent,
            busy_code="TTS_HOSTED_BUSY",
            busy_message="Hosted TTS request limit reached - retry shortly",
        ),
    )


local_decode_limiter, hosted_request_limiter, tts_limiter, hosted_tts_limiter = (
    _build_operation_limiters()
)


def reset_operation_limiters() -> None:
    global local_decode_limiter, hosted_request_limiter, tts_limiter, hosted_tts_limiter
    local_decode_limiter, hosted_request_limiter, tts_limiter, hosted_tts_limiter = (
        _build_operation_limiters()
    )

SUPPORTED_STT_PROVIDERS = {"faster_whisper", "openai"}
SUPPORTED_STT_FALLBACK_PROVIDERS = {"none", "faster_whisper", "openai"}
SUPPORTED_TTS_PROVIDERS = {"pocket_tts", "openai"}
SUPPORTED_TTS_FALLBACK_PROVIDERS = {"none", "pocket_tts", "openai"}


class UnsupportedProviderError(RuntimeError):
    """Raised at boot when a configured provider has no adapter yet — fail fast and loud instead
    of silently no-op'ing into a broken mode (plan §7.1 / setara-s94o.4)."""


def validate_provider_config() -> None:
    if settings.stt_provider not in SUPPORTED_STT_PROVIDERS:
        raise UnsupportedProviderError(
            f"STT_PROVIDER={settings.stt_provider!r} has no adapter yet "
            f"(supported: {sorted(SUPPORTED_STT_PROVIDERS)})"
        )
    if settings.stt_fallback_provider not in SUPPORTED_STT_FALLBACK_PROVIDERS:
        raise UnsupportedProviderError(
            f"STT_FALLBACK_PROVIDER={settings.stt_fallback_provider!r} has no adapter yet "
            f"(supported: {sorted(SUPPORTED_STT_FALLBACK_PROVIDERS)})"
        )
    if settings.tts_provider not in SUPPORTED_TTS_PROVIDERS:
        raise UnsupportedProviderError(
            f"TTS_PROVIDER={settings.tts_provider!r} has no adapter yet "
            f"(supported: {sorted(SUPPORTED_TTS_PROVIDERS)})"
        )
    if settings.tts_fallback_provider not in SUPPORTED_TTS_FALLBACK_PROVIDERS:
        raise UnsupportedProviderError(
            f"TTS_FALLBACK_PROVIDER={settings.tts_fallback_provider!r} has no adapter yet "
            f"(supported: {sorted(SUPPORTED_TTS_FALLBACK_PROVIDERS)})"
        )


def needs_local_stt() -> bool:
    return (
        settings.stt_provider == "faster_whisper"
        or settings.stt_fallback_provider == "faster_whisper"
    )


def needs_hosted_stt() -> bool:
    return settings.stt_provider == "openai" or settings.stt_fallback_provider == "openai"


def hosted_stt_config_usable() -> bool:
    return bool(
        settings.openai_api_key.strip()
        and settings.openai_stt_model.strip()
        and settings.openai_stt_timeout_seconds > 0
    )


def needs_local_tts() -> bool:
    return settings.tts_provider == "pocket_tts" or settings.tts_fallback_provider == "pocket_tts"


def needs_hosted_tts() -> bool:
    return settings.tts_provider == "openai" or settings.tts_fallback_provider == "openai"


def hosted_tts_config_usable() -> bool:
    return bool(
        settings.openai_api_key.strip()
        and settings.openai_tts_model.strip()
        and settings.openai_tts_timeout_seconds > 0
    )


def load_local_stt_service() -> SttService:
    """Import the local engine only when provider selection requires it."""
    from app.services.stt_service import SttService

    return SttService()


def load_local_tts_service() -> TtsService:
    """Import Pocket TTS only when provider selection requires it."""
    from app.services.tts_service import TtsService

    return TtsService()


def build_stt_adapter(provider: str, service: SttService | None):
    if provider == "none":
        return None
    if provider == "faster_whisper":
        if service is None:
            return None
        from app.providers.faster_whisper import FasterWhisperAdapter

        return FasterWhisperAdapter(service)
    if provider == "openai":
        from app.providers.openai_stt import OpenAiSttAdapter

        return OpenAiSttAdapter()
    raise UnsupportedProviderError(f"STT provider {provider!r} has no adapter yet")


def build_tts_adapter(provider: str, service: TtsService | None):
    if provider == "none":
        return None
    if provider == "pocket_tts":
        if service is None:
            return None
        from app.providers.pocket_tts import PocketTtsAdapter

        return PocketTtsAdapter(service)
    if provider == "openai":
        from app.providers.openai_tts import OpenAiTtsAdapter

        return OpenAiTtsAdapter()
    raise UnsupportedProviderError(f"TTS provider {provider!r} has no adapter yet")


def build_routers() -> None:
    """(Re)build the provider routers from the currently loaded engine singletons. Call once both
    stt_service/tts_service have been loaded (or attempted) at startup."""
    global stt_router, tts_router

    stt_primary = build_stt_adapter(settings.stt_provider, stt_service)
    stt_fallback_name = settings.stt_fallback_provider
    if stt_fallback_name == settings.stt_provider:
        stt_fallback_name = "none"
    stt_fallback = build_stt_adapter(stt_fallback_name, stt_service)
    stt_router = (
        SttProviderRouter(primary=stt_primary, fallback=stt_fallback, policy=stt_policy)
        if stt_primary else None
    )

    tts_primary = build_tts_adapter(settings.tts_provider, tts_service)
    tts_fallback_name = settings.tts_fallback_provider
    if tts_fallback_name == settings.tts_provider:
        tts_fallback_name = "none"
    tts_fallback = build_tts_adapter(tts_fallback_name, tts_service)
    tts_router = TtsProviderRouter(primary=tts_primary, fallback=tts_fallback) if tts_primary else None


def has_stt_adapter(provider: str) -> bool:
    return stt_router is not None and stt_router.resolve_provider(provider) is not None


def has_tts_adapter(provider: str) -> bool:
    return tts_router is not None and tts_router.resolve_provider(provider) is not None


def reset_components() -> None:
    global stt_service, tts_service, stt_router, tts_router

    stt_service = None
    tts_service = None
    stt_router = None
    tts_router = None
    reset_operation_limiters()
