import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from app import runtime
from app.auth import require_api_key
from app.providers.base import TtsOptions
from app.schemas import TtsRequest
from app.services.operation_limiter import OperationBusyError
from app.services.tts_service import TtsSynthesisError

router = APIRouter()


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

    options = TtsOptions(voice_id=req.voiceId, format=req.format, client_id=_client_id)
    try:
        result = await runtime.tts_router.synthesize(text, options)
    except OperationBusyError as exc:
        raise _busy_http_error(exc) from exc
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
    if runtime.tts_router is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if req.format.lower() not in {"pcm", "l16"}:
        raise HTTPException(status_code=400, detail="Unsupported format; expected pcm or l16")

    options = TtsOptions(voice_id=req.voiceId, format=req.format, client_id=_client_id)
    try:
        stream = await runtime.tts_router.synthesize_stream(text, options)
    except OperationBusyError as exc:
        raise _busy_http_error(exc) from exc
    except TtsSynthesisError as exc:
        raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {exc}") from exc

    return StreamingResponse(
        stream.chunks,
        media_type="audio/l16",
        headers={
            "X-Sample-Rate": str(stream.metadata.sample_rate),
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
            "Cache-Control": "no-store, no-transform",
            "X-Content-Type-Options": "nosniff",
        },
        # Belt-and-suspenders: aclose() on an abandoned/never-fully-drained response still runs
        # the adapter's own finally/lease-release cleanup; a no-op if already exhausted.
        background=BackgroundTask(stream.chunks.aclose),
    )


def _busy_http_error(error: OperationBusyError) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={"code": error.code, "message": str(error), "retryable": error.retryable},
    )
