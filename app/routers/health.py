from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Response, status

from app import runtime
from app.config import settings
from app.schemas import (
    HealthResponse,
    HealthSttInfo,
    HealthTtsInfo,
    ModelLimits,
    ModelsResponse,
    SttInfo,
    TtsInfo,
)

router = APIRouter()


@dataclass(frozen=True)
class ComponentReadiness:
    artifact_ready: bool | None
    local_stt_ready: bool
    hosted_stt_ready: bool
    stt_primary_ready: bool
    stt_fallback_ready: bool | None
    local_tts_ready: bool
    hosted_tts_ready: bool
    tts_ready: bool
    tts_fallback_ready: bool | None
    stt_warning: str | None
    tts_warning: str | None

    def summary(self) -> str:
        messages = [message for message in (self.stt_warning, self.tts_warning) if message]
        if messages:
            return "; ".join(messages)
        if self.stt_primary_ready:
            return "all configured components ready"
        return f"primary STT provider {settings.stt_provider} is unavailable"


def component_readiness() -> ComponentReadiness:
    artifact_ready = _stt_artifact_ready()
    local_ready = (
        runtime.stt_service is not None
        and runtime.has_stt_adapter("faster_whisper")
        and artifact_ready is not False
    )
    hosted_ready = runtime.hosted_stt_config_usable() and runtime.has_stt_adapter("openai")
    primary_ready = _provider_ready(settings.stt_provider, local_ready, hosted_ready)

    fallback_provider = _fallback_provider()
    fallback_ready = None
    stt_warning = None
    if fallback_provider is not None:
        fallback_ready = _provider_ready(fallback_provider, local_ready, hosted_ready)
        if not fallback_ready:
            stt_warning = f"Configured STT fallback {fallback_provider} is unavailable"
    if settings.stt_provider == "openai" and not runtime.hosted_stt_config_usable():
        stt_warning = "OpenAI STT configuration is incomplete"

    # TTS readiness is defined per-provider (plan §8), not by `runtime.tts_service != null` -
    # a hosted-only OpenAI config never constructs tts_service at all.
    local_tts_ready = runtime.tts_service is not None and runtime.has_tts_adapter("pocket_tts")
    hosted_tts_ready = runtime.hosted_tts_config_usable() and runtime.has_tts_adapter("openai")
    tts_primary_ready = _tts_provider_ready(settings.tts_provider, local_tts_ready, hosted_tts_ready)

    tts_fallback_provider = _tts_fallback_provider()
    tts_fallback_ready = None
    tts_warning = None
    if tts_fallback_provider is not None:
        tts_fallback_ready = _tts_provider_ready(
            tts_fallback_provider, local_tts_ready, hosted_tts_ready
        )
        if not tts_fallback_ready:
            tts_warning = f"Configured TTS fallback {tts_fallback_provider} is unavailable"
    if settings.tts_provider == "openai" and not runtime.hosted_tts_config_usable():
        tts_warning = "OpenAI TTS configuration is incomplete"
    if not tts_primary_ready and tts_warning is None:
        tts_warning = f"Configured TTS provider {settings.tts_provider} is unavailable"

    return ComponentReadiness(
        artifact_ready=artifact_ready,
        local_stt_ready=local_ready,
        hosted_stt_ready=hosted_ready,
        stt_primary_ready=primary_ready,
        stt_fallback_ready=fallback_ready,
        local_tts_ready=local_tts_ready,
        hosted_tts_ready=hosted_tts_ready,
        tts_ready=tts_primary_ready,
        tts_fallback_ready=tts_fallback_ready,
        stt_warning=stt_warning,
        tts_warning=tts_warning,
    )


@router.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """STT readiness gate with independently reported local, hosted, fallback, and TTS state."""
    readiness = component_readiness()
    artifact_ready = readiness.artifact_ready
    if not readiness.stt_primary_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    health_status = "ok"
    if not readiness.stt_primary_ready:
        health_status = "loading"
    elif readiness.stt_warning is not None or readiness.tts_warning is not None:
        health_status = "degraded"

    fallback_provider = _fallback_provider()
    tts_fallback_provider = _tts_fallback_provider()
    return HealthResponse(
        status=health_status,
        mode=settings.asa_voice_mode,
        sttLoaded=readiness.stt_primary_ready,
        ttsLoaded=readiness.tts_ready,
        stt=HealthSttInfo(
            model=_stt_model(settings.stt_provider),
            device=_local_value(settings.stt_device, settings.stt_provider),
            computeType=settings.stt_compute_type if settings.stt_provider == "faster_whisper" else None,
            artifactReady=artifact_ready,
            provider=settings.stt_provider,
            fallbackProvider=fallback_provider,
            fallbackModel=_stt_model(fallback_provider) if fallback_provider else None,
            localReady=readiness.local_stt_ready,
            hostedReady=readiness.hosted_stt_ready,
            fallbackReady=readiness.stt_fallback_ready,
            ready=readiness.stt_primary_ready,
            warning=readiness.stt_warning,
        ),
        tts=HealthTtsInfo(
            engine=_tts_engine(settings.tts_provider),
            model=_tts_model(settings.tts_provider),
            sampleRate=settings.tts_sample_rate if settings.tts_provider == "pocket_tts" else None,
            provider=settings.tts_provider,
            fallbackProvider=tts_fallback_provider,
            fallbackModel=_tts_model(tts_fallback_provider) if tts_fallback_provider else None,
            localReady=readiness.local_tts_ready,
            hostedReady=readiness.hosted_tts_ready,
            fallbackReady=readiness.tts_fallback_ready,
            ready=readiness.tts_ready,
            warning=readiness.tts_warning,
        ),
    )


def _stt_artifact_ready() -> bool | None:
    if not runtime.needs_local_stt():
        return None
    model_path = Path(settings.stt_model)
    if not model_path.is_absolute():
        return None
    return (model_path / ".asa_model_ready").is_file()


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    readiness = component_readiness()
    voices = _tts_active_adapter_voices()
    fallback_provider = _fallback_provider()
    tts_fallback_provider = _tts_fallback_provider()
    available_stt = [
        provider
        for provider in sorted(runtime.SUPPORTED_STT_PROVIDERS)
        if _provider_ready(provider, readiness.local_stt_ready, readiness.hosted_stt_ready)
    ]
    available_tts = [
        provider
        for provider in sorted(runtime.SUPPORTED_TTS_PROVIDERS)
        if _tts_provider_ready(provider, readiness.local_tts_ready, readiness.hosted_tts_ready)
    ]

    return ModelsResponse(
        mode=settings.asa_voice_mode,
        limits=ModelLimits(
            cpu=(
                settings.local_stt_max_concurrent
                if settings.local_stt_max_concurrent is not None
                else settings.max_concurrent_stt
            )
            + (
                settings.tts_max_concurrent
                if settings.tts_max_concurrent is not None
                else settings.max_concurrent_tts
            ),
            memoryMb=2048,
            maxAudioSeconds=settings.max_audio_seconds,
            maxUploadMb=settings.max_upload_mb,
        ),
        stt=SttInfo(
            engine=_stt_engine(settings.stt_provider),
            model=_stt_model(settings.stt_provider),
            device=_local_value(settings.stt_device, settings.stt_provider),
            computeType=_local_value(settings.stt_compute_type, settings.stt_provider),
            activeProvider=settings.stt_provider,
            activeModel=_stt_model(settings.stt_provider),
            activeLoaded=readiness.stt_primary_ready,
            fallbackProvider=fallback_provider,
            fallbackModel=_stt_model(fallback_provider) if fallback_provider else None,
            fallbackLoaded=readiness.stt_fallback_ready,
            localLoaded=readiness.local_stt_ready,
            hostedConfigured=runtime.hosted_stt_config_usable(),
            availableProviders=available_stt,
            supportedProviders=sorted(runtime.SUPPORTED_STT_PROVIDERS),
        ),
        tts=TtsInfo(
            engine=_tts_engine(settings.tts_provider),
            activeModel=_tts_model(settings.tts_provider),
            loaded=readiness.tts_ready,
            defaultVoice=settings.tts_default_voice,
            voices=voices,
            activeProvider=settings.tts_provider,
            fallbackProvider=tts_fallback_provider,
            fallbackModel=_tts_model(tts_fallback_provider) if tts_fallback_provider else None,
            fallbackLoaded=readiness.tts_fallback_ready,
            localLoaded=readiness.local_tts_ready,
            hostedConfigured=runtime.hosted_tts_config_usable(),
            availableProviders=available_tts,
            supportedProviders=sorted(runtime.SUPPORTED_TTS_PROVIDERS),
        ),
    )


def _provider_ready(provider: str, local_ready: bool, hosted_ready: bool) -> bool:
    if provider == "faster_whisper":
        return local_ready
    if provider == "openai":
        return hosted_ready
    return False


def _fallback_provider() -> str | None:
    provider = settings.stt_fallback_provider
    if provider == "none" or provider == settings.stt_provider:
        return None
    return provider


def _stt_model(provider: str | None) -> str:
    if provider == "openai":
        return settings.openai_stt_model
    if provider == "faster_whisper":
        return settings.stt_model
    return ""


def _stt_engine(provider: str) -> str:
    if provider == "faster_whisper":
        return "faster-whisper"
    return provider


def _local_value(value: str, provider: str) -> str | None:
    if provider == "faster_whisper":
        return value
    return None


def _tts_provider_ready(provider: str, local_ready: bool, hosted_ready: bool) -> bool:
    if provider == "pocket_tts":
        return local_ready
    if provider == "openai":
        return hosted_ready
    return False


def _tts_fallback_provider() -> str | None:
    provider = settings.tts_fallback_provider
    if provider == "none" or provider == settings.tts_provider:
        return None
    return provider


def _tts_model(provider: str | None) -> str:
    if provider == "openai":
        return settings.openai_tts_model
    if provider == "pocket_tts":
        return settings.tts_default_model
    return ""


def _tts_engine(provider: str) -> str:
    if provider == "pocket_tts":
        return "pocket-tts"
    return provider


def _tts_active_adapter_voices() -> list[dict]:
    if runtime.tts_router is None:
        return []
    adapter = runtime.tts_router.resolve_provider(settings.tts_provider) or runtime.tts_router.primary
    return adapter.list_voices() if adapter else []
