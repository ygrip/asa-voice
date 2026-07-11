"""Shared singletons + concurrency gates, initialized at app startup (see main.lifespan)."""
import asyncio

from app.config import settings
from app.providers.faster_whisper import FasterWhisperAdapter
from app.providers.openai_stt import OpenAiSttAdapter
from app.providers.pocket_tts import PocketTtsAdapter
from app.providers.policy import RequestValidationPolicy
from app.providers.router import SttProviderRouter, TtsProviderRouter
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

# One job at a time by default — protects the capped container from OOM/CPU contention.
stt_semaphore = asyncio.Semaphore(settings.max_concurrent_stt)
tts_semaphore = asyncio.Semaphore(settings.max_concurrent_tts)

SUPPORTED_STT_PROVIDERS = {"faster_whisper", "openai"}
SUPPORTED_STT_FALLBACK_PROVIDERS = {"none", "faster_whisper", "openai"}
SUPPORTED_TTS_PROVIDERS = {"pocket_tts"}
SUPPORTED_TTS_FALLBACK_PROVIDERS = {"none", "pocket_tts"}


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


def build_stt_adapter(provider: str, service: SttService | None):
    if provider == "none":
        return None
    if provider == "faster_whisper":
        return FasterWhisperAdapter(service) if service is not None else None
    if provider == "openai":
        return OpenAiSttAdapter()
    raise UnsupportedProviderError(f"STT provider {provider!r} has no adapter yet")


def build_tts_adapter(provider: str, service: TtsService | None):
    if provider == "none" or service is None:
        return None
    if provider == "pocket_tts":
        return PocketTtsAdapter(service)
    raise UnsupportedProviderError(f"TTS provider {provider!r} has no adapter yet")


def build_routers() -> None:
    """(Re)build the provider routers from the currently loaded engine singletons. Call once both
    stt_service/tts_service have been loaded (or attempted) at startup."""
    global stt_router, tts_router

    stt_primary = build_stt_adapter(settings.stt_provider, stt_service)
    stt_fallback_name = settings.stt_fallback_provider if settings.stt_fallback_provider != settings.stt_provider else "none"
    stt_fallback = build_stt_adapter(stt_fallback_name, stt_service)
    stt_router = (
        SttProviderRouter(primary=stt_primary, fallback=stt_fallback, policy=stt_policy)
        if stt_primary else None
    )

    tts_primary = build_tts_adapter(settings.tts_provider, tts_service)
    tts_fallback_name = settings.tts_fallback_provider if settings.tts_fallback_provider != settings.tts_provider else "none"
    tts_fallback = build_tts_adapter(tts_fallback_name, tts_service)
    tts_router = TtsProviderRouter(primary=tts_primary, fallback=tts_fallback) if tts_primary else None

