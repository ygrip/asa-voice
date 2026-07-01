"""Shared singletons + concurrency gates, initialized at app startup (see main.lifespan)."""
import asyncio

from app.config import settings
from app.services.stt_service import SttService
from app.services.tts_service import TtsService

stt_service: SttService | None = None
tts_service: TtsService | None = None

# One job at a time by default — protects the capped container from OOM/CPU contention.
stt_semaphore = asyncio.Semaphore(settings.max_concurrent_stt)
tts_semaphore = asyncio.Semaphore(settings.max_concurrent_tts)
