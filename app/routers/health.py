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
    tts_ready: bool
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

    tts_ready = runtime.tts_service is not None and runtime.tts_router is not None
    tts_warning = None
    if not tts_ready:
        tts_warning = f"Configured TTS provider {settings.tts_provider} is unavailable"

    return ComponentReadiness(
        artifact_ready=artifact_ready,
        local_stt_ready=local_ready,
        hosted_stt_ready=hosted_ready,
        stt_primary_ready=primary_ready,
        stt_fallback_ready=fallback_ready,
        tts_ready=tts_ready,
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
            engine=settings.tts_engine,
            model=settings.tts_default_model,
            sampleRate=settings.tts_sample_rate,
            provider=settings.tts_provider,
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
    voices = runtime.tts_service.list_voices() if runtime.tts_service else []
    fallback_provider = _fallback_provider()
    available_stt = [
        provider
        for provider in sorted(runtime.SUPPORTED_STT_PROVIDERS)
        if _provider_ready(provider, readiness.local_stt_ready, readiness.hosted_stt_ready)
    ]
    available_tts = [settings.tts_provider] if readiness.tts_ready else []

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
            engine=settings.tts_engine,
            activeModel=settings.tts_default_model,
            loaded=readiness.tts_ready,
            defaultVoice=settings.tts_default_voice,
            voices=voices,
            activeProvider=settings.tts_provider,
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
