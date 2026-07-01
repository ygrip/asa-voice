import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import runtime
from app.config import settings
from app.routers import health, stt, tts
from app.services.stt_service import SttService
from app.services.tts_service import TtsService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asa-voice")


@asynccontextmanager
async def lifespan(_app: FastAPI):
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

    if runtime.stt_service and runtime.tts_service:
        log.info("ASA voice sidecar ready")
    else:
        log.warning("ASA voice sidecar started DEGRADED (a model failed to load); /health = 503")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health.router)
app.include_router(stt.router)
app.include_router(tts.router)
