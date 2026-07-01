import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app import runtime
from app.config import settings
from app.schemas import SttResponse
from app.services import audio_service
from app.services.stt_service import StreamingSttSession, collapse_repeats

router = APIRouter()
log = logging.getLogger("asa.stt")


@router.post("/stt", response_model=SttResponse)
async def stt(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    vad: bool | None = Form(default=None),
) -> SttResponse:
    if runtime.stt_service is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    # Reject before doing expensive work if a job is already running (cap protects RAM/CPU).
    if runtime.stt_semaphore.locked():
        raise HTTPException(status_code=429, detail="STT busy — retry shortly")

    audio_bytes = await file.read()
    try:
        audio_service.check_upload_size(len(audio_bytes))
    except audio_service.AudioTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    suffix = ".wav"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower()

    path = audio_service.write_temp(audio_bytes, suffix)
    try:
        audio_service.enforce_duration(path)
    except audio_service.AudioTooLong as exc:
        import os
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    async with runtime.stt_semaphore:
        # transcribe() is blocking (CTranslate2) and removes the temp file when done.
        result = await run_in_threadpool(runtime.stt_service.transcribe, path, language, vad)
    return SttResponse(**result)


@router.websocket("/stt/stream")
async def stt_stream(ws: WebSocket) -> None:
    """Rolling-window streaming STT. Client sends binary PCM16 mono @16kHz frames; server emits
    {type:"partial", text} as words stabilize and {type:"final", text} on a flush text message
    ("flush" = end of utterance, e.g. VAD silence). Decodes are serialized through the STT slot."""
    await ws.accept()
    if runtime.stt_service is None:
        await ws.send_json({"type": "error", "detail": "STT model not loaded"})
        await ws.close()
        return

    session = StreamingSttSession(runtime.stt_service)
    last_speech_time: list[float] = [time.monotonic()]  # mutable cell for the closure
    closed = False

    async def send_json(message: dict) -> bool:
        nonlocal closed
        if closed:
            return False
        try:
            await ws.send_json(message)
            return True
        except (WebSocketDisconnect, RuntimeError):
            closed = True
            return False

    async def run_decode() -> None:
        # Serialize decodes on the single STT slot; skip (don't queue) if a job is already running.
        if closed or runtime.stt_semaphore.locked():
            return
        try:
            async with runtime.stt_semaphore:
                result = await run_in_threadpool(session.decode)
            text = collapse_repeats((session.committed_text + " " + result["partial"]).strip())
            await send_json({"type": "partial", "text": text})
        except Exception:  # noqa: BLE001 - decode errors must not close the WS
            if closed:
                return
            log.exception("stt decode error (non-fatal)")

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                closed = True
                break
            data = msg.get("bytes")
            if data is not None:
                session.add_pcm(data)
                if not session.is_silent():
                    last_speech_time[0] = time.monotonic()
                if session.should_decode():
                    await run_decode()
                elif (
                    settings.stt_stream_silence_flush_s > 0
                    and session.committed_text
                    and session.is_silent()
                    and time.monotonic() - last_speech_time[0] > settings.stt_stream_silence_flush_s
                ):
                    # Server-side silence flush: committed text + sustained silence, then auto-finalize.
                    try:
                        async with runtime.stt_semaphore:
                            if session.has_buffered_audio():
                                await run_in_threadpool(session.decode)
                            final = await run_in_threadpool(session.flush)
                        await send_json({"type": "final", "text": final["final"]})
                        last_speech_time[0] = time.monotonic()
                    except Exception:  # noqa: BLE001
                        if closed:
                            break
                        log.exception("stt auto-flush error (non-fatal)")
                continue
            text = msg.get("text")
            if text == "flush":
                try:
                    async with runtime.stt_semaphore:
                        if session.has_buffered_audio():
                            await run_in_threadpool(session.decode)
                        final = await run_in_threadpool(session.flush)
                    await send_json({"type": "final", "text": final["final"]})
                except Exception:  # noqa: BLE001
                    if closed:
                        break
                    log.exception("stt flush error (non-fatal)")
                    await send_json({"type": "final", "text": ""})
    except WebSocketDisconnect:
        closed = True
        pass
    except Exception:  # noqa: BLE001 - never crash the worker on a bad stream
        log.exception("stt stream error")
        await send_json({"type": "error", "detail": "stream failed"})
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
