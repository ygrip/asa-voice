from pathlib import Path

from fastapi import APIRouter, Response, status

from app import runtime
from app.config import settings
from app.schemas import (
    HealthResponse, HealthSttInfo, HealthTtsInfo, ModelLimits, ModelsResponse, SttInfo, TtsInfo,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Readiness gate: 200 only when BOTH models are loaded and the service can serve traffic;
    503 otherwise. The Docker healthcheck (`curl -f`) treats non-2xx as unhealthy, so the
    container stays 'starting/unhealthy' until STT+TTS are actually ready."""
    stt_ready = runtime.stt_service is not None and runtime.stt_router is not None
    artifact_ready = _stt_artifact_ready()
    if artifact_ready is False:
        stt_ready = False
    tts_ready = runtime.tts_service is not None and runtime.tts_router is not None
    ready = stt_ready and tts_ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    fallback_provider = (
        settings.stt_fallback_provider if settings.stt_fallback_provider != "none" else None
    )
    return HealthResponse(
        status="ok" if ready else "loading",
        mode=settings.asa_voice_mode,
        sttLoaded=stt_ready,
        ttsLoaded=tts_ready,
        stt=HealthSttInfo(
            model=settings.stt_model,
            device=settings.stt_device,
            computeType=settings.stt_compute_type,
            artifactReady=artifact_ready,
            provider=settings.stt_provider,
            fallbackProvider=fallback_provider,
            ready=stt_ready,
        ),
        tts=HealthTtsInfo(
            engine=settings.tts_engine,
            sampleRate=settings.tts_sample_rate,
            provider=settings.tts_provider,
            ready=tts_ready,
        ),
    )


def _stt_artifact_ready() -> bool | None:
    model_path = Path(settings.stt_model)
    if not model_path.is_absolute():
        return None
    return (model_path / ".asa_model_ready").is_file()


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    voices = runtime.tts_service.list_voices() if runtime.tts_service else []
    fallback_provider = (
        settings.stt_fallback_provider if settings.stt_fallback_provider != "none" else None
    )
    return ModelsResponse(
        mode=settings.asa_voice_mode,
        limits=ModelLimits(
            cpu=settings.max_concurrent_stt + settings.max_concurrent_tts,
            memoryMb=2048,
            maxAudioSeconds=settings.max_audio_seconds,
            maxUploadMb=settings.max_upload_mb,
        ),
        stt=SttInfo(
            engine="faster-whisper",
            model=settings.stt_model,
            device=settings.stt_device,
            computeType=settings.stt_compute_type,
            activeProvider=settings.stt_provider,
            activeModel=settings.stt_model,
            fallbackProvider=fallback_provider,
            fallbackModel=settings.stt_model if fallback_provider else None,
            availableProviders=sorted(runtime.SUPPORTED_STT_PROVIDERS),
        ),
        tts=TtsInfo(
            engine=settings.tts_engine,
            defaultVoice=settings.tts_default_voice,
            voices=voices,
            activeProvider=settings.tts_provider,
            availableProviders=sorted(runtime.SUPPORTED_TTS_PROVIDERS),
        ),
    )

