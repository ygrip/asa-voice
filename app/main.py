import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import runtime
from app.auth import _get_clients
from app.config import settings
from app.routers import health, stt, tts
from app.services.temp_audio_cleanup import cleanup_expired_openai_stt_files

# LOG_LEVEL=DEBUG surfaces the per-session STT timeline/decode diagnostics added for setara-s94o
# (asa.stt / asa.stt.scheduler / asa.stt.service loggers) without needing a code change.
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("asa-voice")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.reset_components()
    clients = _get_clients()
    if clients:
        log.info("Auth enabled - %d client(s): %s", len(clients), ", ".join(clients))
    else:
        log.warning("Auth DISABLED - ALLOWED_CLIENTS not set; all requests accepted")

    # Fail fast on an unimplemented provider selection instead of booting into a silently broken
    # mode (plan §7.1 / setara-s94o.4). Distinct from a model load failure below, which degrades
    # gracefully - this is a config error and should stop the process.
    runtime.validate_provider_config()
    if runtime.needs_hosted_stt():
        try:
            removed_orphans = cleanup_expired_openai_stt_files(
                settings.openai_stt_buffer_directory,
                settings.openai_stt_orphan_ttl_seconds,
            )
            if removed_orphans:
                log.info("Removed %d expired hosted STT temporary file(s)", removed_orphans)
        except (OSError, ValueError, RuntimeError):
            log.warning("Hosted STT orphan cleanup could not complete safely")
    log.info(
        "ASA_VOICE_MODE=%s STT_PROVIDER=%s (fallback=%s) TTS_PROVIDER=%s (fallback=%s)",
        settings.asa_voice_mode, settings.stt_provider, settings.stt_fallback_provider,
        settings.tts_provider, settings.tts_fallback_provider,
    )

    # Loading the local engine imports CTranslate2 and can download model artifacts, so do not even
    # construct that path unless the selected primary or fallback requires faster-whisper.
    if runtime.needs_local_stt():
        try:
            log.info(
                "Loading STT model %s (%s/%s)",
                settings.stt_model,
                settings.stt_device,
                settings.stt_compute_type,
            )
            runtime.stt_service = runtime.load_local_stt_service()
            log.info("Local STT ready")
        except Exception:
            log.exception("Local STT model failed to load")
    else:
        log.info("Skipping local STT model; configured STT providers are hosted-only")

    # Same reasoning as STT above, applied to Pocket TTS: its constructor imports torch and loads
    # model weights, so hosted-only TTS configs must never construct it (plan §8).
    if runtime.needs_local_tts():
        try:
            log.info("Loading TTS engine %s", settings.tts_engine)
            runtime.tts_service = runtime.load_local_tts_service()
            log.info("Local TTS ready")
        except Exception:
            log.exception("Local TTS engine failed to load")
    else:
        log.info("Skipping Pocket TTS; configured TTS providers are hosted-only")

    runtime.build_routers()

    readiness = health.component_readiness()
    if readiness.stt_primary_ready and readiness.tts_ready and readiness.stt_warning is None:
        log.info("ASA voice sidecar ready")
    elif readiness.stt_primary_ready:
        log.warning("ASA voice sidecar started DEGRADED: %s", readiness.summary())
    else:
        log.warning("ASA voice sidecar STT is not ready: %s", readiness.summary())
    try:
        yield
    finally:
        runtime.reset_components()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(stt.router)
app.include_router(tts.router)
