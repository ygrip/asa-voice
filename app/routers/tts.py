import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from app import runtime
from app.auth import require_api_key
from app.config import settings
from app.providers.base import TtsOptions
from app.schemas import TtsRequest
from app.services.tts_service import TtsSynthesisError

router = APIRouter()

_STREAM_END = object()  # sentinel: sync generator exhausted


@router.post("/tts")
async def tts(
    req: TtsRequest,
    _client_id: str = Depends(require_api_key),
) -> Response:
    if runtime.tts_router is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if req.format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Unsupported format; expected wav")

    if runtime.tts_semaphore.locked():
        raise HTTPException(status_code=429, detail="TTS busy — retry shortly")

    options = TtsOptions(voice_id=req.voiceId, format=req.format, client_id=_client_id)
    async with runtime.tts_semaphore:
        try:
            result = await runtime.tts_router.synthesize(text, options)
        except TtsSynthesisError as exc:
            raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {exc}") from exc

    try:
        with open(result.audio_path, "rb") as f:
            audio = f.read()
    finally:
        try:
            os.remove(result.audio_path)
        except OSError:
            pass

    return Response(content=audio, media_type=result.content_type)


@router.post("/tts/stream")
async def tts_stream(
    req: TtsRequest,
    _client_id: str = Depends(require_api_key),
) -> StreamingResponse:
    if runtime.tts_service is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if req.format.lower() not in {"pcm", "l16"}:
        raise HTTPException(status_code=400, detail="Unsupported format; expected pcm or l16")

    if runtime.tts_semaphore.locked():
        raise HTTPException(status_code=429, detail="TTS busy — retry shortly")

    async def pcm_frames():
        async with runtime.tts_semaphore:
            gen = runtime.tts_service.synthesize_stream(text, req.voiceId)
            while True:
                try:
                    chunk = await run_in_threadpool(lambda: next(gen, _STREAM_END))
                except TtsSynthesisError:
                    break
                if chunk is _STREAM_END:
                    break
                yield chunk

    return StreamingResponse(
        pcm_frames(),
        media_type="audio/l16",
        headers={
            "X-Sample-Rate": str(int(runtime.tts_service.model.sample_rate)),
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
            "Cache-Control": "no-store, no-transform",
            "X-Content-Type-Options": "nosniff",
        },
    )
