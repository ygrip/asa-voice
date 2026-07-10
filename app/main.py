import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import runtime
from app.auth import _get_clients
from app.config import settings
from app.routers import health, stt, tts
from app.services.stt_service import SttService
from app.services.tts_service import TtsService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asa-voice")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    clients = _get_clients()
    if clients:
        log.info("Auth enabled — %d client(s): %s", len(clients), ", ".join(clients))
    else:
        log.warning("Auth DISABLED — ALLOWED_CLIENTS not set; all requests accepted")

    # Fail fast on an unimplemented provider selection instead of booting into a silently broken
    # mode (plan §7.1 / setara-s94o.4). Distinct from a model *load* failure below, which degrades
    # gracefully — this is a config error and should stop the process.
    runtime.validate_provider_config()
    log.info(
        "ASA_VOICE_MODE=%s STT_PROVIDER=%s (fallback=%s) TTS_PROVIDER=%s (fallback=%s)",
        settings.asa_voice_mode, settings.stt_provider, settings.stt_fallback_provider,
        settings.tts_provider, settings.tts_fallback_provider,
    )

    # Preload models once at startup. Load each independently and never crash the process on
    # failure — the container stays up serving /health, which returns 503 until BOTH are loaded.
    # This gives a clear readiness signal instead of a crash-loop.
    try:
        log.info("Loading STT model %s (%s/%s)…", settings.stt_model, settings.stt_device, settings.stt_compute_type)
        runtime.stt_service = SttService()
        log.info("STT ready")
    except Exception:
        log.exception("STT model failed to load — /health will report not ready")
    try:
        log.info("Loading TTS engine %s…", settings.tts_engine)
        runtime.tts_service = TtsService()
        log.info("TTS ready")
    except Exception:
        log.exception("TTS engine failed to load — /health will report not ready")

    runtime.build_routers()

    if runtime.stt_service and runtime.tts_service:
        log.info("ASA voice sidecar ready")
    else:
        log.warning("ASA voice sidecar started DEGRADED (a model failed to load); /health = 503")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(stt.router)
app.include_router(tts.router)
