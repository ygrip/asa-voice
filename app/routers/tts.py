from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from app import runtime
from app.config import settings
from app.schemas import TtsRequest
from app.services.tts_service import TtsSynthesisError

router = APIRouter()

_STREAM_END = object()  # sentinel: sync generator exhausted


@router.post("/tts")
async def tts(req: TtsRequest) -> Response:
    if runtime.tts_service is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if runtime.tts_semaphore.locked():
        raise HTTPException(status_code=429, detail="TTS busy — retry shortly")

    async with runtime.tts_semaphore:
        try:
            audio = await run_in_threadpool(runtime.tts_service.synthesize, text, req.voiceId)
        except TtsSynthesisError as exc:
            raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {exc}") from exc

    return Response(content=audio, media_type="audio/wav")


@router.post("/tts/stream")
async def tts_stream(req: TtsRequest) -> StreamingResponse:
    if runtime.tts_service is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if runtime.tts_semaphore.locked():
        raise HTTPException(status_code=429, detail="TTS busy — retry shortly")

    async def pcm_frames():
        # Hold the single TTS slot for the whole stream (generate_audio_stream isn't thread-safe).
        async with runtime.tts_semaphore:
            gen = runtime.tts_service.synthesize_stream(text, req.voiceId)
            # Pull each chunk in the threadpool so torch decode never blocks the event loop.
            while True:
                try:
                    chunk = await run_in_threadpool(lambda: next(gen, _STREAM_END))
                except TtsSynthesisError:
                    break  # stop cleanly; client already has partial audio
                if chunk is _STREAM_END:
                    break
                yield chunk

    return StreamingResponse(
        pcm_frames(),
        media_type="audio/L16",
        headers={
            "X-Sample-Rate": str(int(runtime.tts_service.model.sample_rate)),
            "X-Channels": "1",
            "Cache-Control": "no-store",
        },
    )
