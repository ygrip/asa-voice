import base64
import binascii
import json
import logging
import os
import time
import uuid

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool

from app import runtime
from app.auth import require_api_key, validate_key
from app.config import settings
from app.providers.base import SttOptions
from app.schemas import SttResponse, stt_response
from app.services import audio_service
from app.services.stt_service import StreamingSttSession, collapse_repeats
from app.services.stt_context import SttDecodeContext

router = APIRouter()
log = logging.getLogger("asa.stt")


def _options_from_context(
    context: SttDecodeContext | None,
    client_id: str,
    request_id: str | None,
    language: str | None = None,
) -> SttOptions:
    return SttOptions(
        language=language or (context.language if context else None),
        prompt=context.prompt if context else None,
        hotwords=context.hotwords if context else None,
        request_id=request_id or (context.request_id if context else None) or str(uuid.uuid4()),
        client_id=client_id,
    )


@router.post("/stt/raw", response_model=SttResponse)
async def stt_raw(
    request: Request,
    x_sample_rate: int = Header(default=16000),
    x_channels: int = Header(default=1),
    x_sample_format: str = Header(default="s16le"),
    x_stt_context: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
    _client_id: str = Depends(require_api_key),
) -> SttResponse:
    if runtime.stt_router is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "audio/l16":
        raise HTTPException(status_code=415, detail="Expected Content-Type audio/l16")
    if x_sample_rate != 16000 or x_channels != 1 or x_sample_format.lower() != "s16le":
        raise HTTPException(status_code=415, detail="Expected PCM16 mono 16kHz s16le")
    if runtime.stt_semaphore.locked():
        raise HTTPException(status_code=429, detail="STT busy - retry shortly")

    body = await request.body()
    try:
        audio_service.check_upload_size(len(body))
    except audio_service.AudioTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if len(body) % 2 != 0:
        raise HTTPException(status_code=400, detail="PCM16 body must contain complete samples")
    sample_count = len(body) // 2
    if sample_count > settings.max_audio_seconds * 16000:
        raise HTTPException(status_code=413, detail="Audio exceeds maximum duration")

    context = _decode_header_context(x_stt_context)
    options = _options_from_context(context, _client_id, x_request_id)
    audio = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    async with runtime.stt_semaphore:
        result = await runtime.stt_router.transcribe_array(audio, options)
    return stt_response(result, options.request_id)


def _decode_header_context(encoded: str | None) -> SttDecodeContext | None:
    if not encoded:
        return None
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded + padding)
        if len(decoded) > 4096:
            raise ValueError("context is too large")
        return SttDecodeContext.model_validate_json(decoded)
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid X-Stt-Context") from exc


@router.post("/stt", response_model=SttResponse)
async def stt(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    context: str | None = Form(default=None),
    x_request_id: str | None = Header(default=None),
    _client_id: str = Depends(require_api_key),
) -> SttResponse:
    if runtime.stt_router is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    if runtime.stt_semaphore.locked():
        raise HTTPException(status_code=429, detail="STT busy - retry shortly")

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
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        decode_context = SttDecodeContext.model_validate_json(context) if context else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid STT context") from exc

    options = _options_from_context(decode_context, _client_id, x_request_id, language)
    async with runtime.stt_semaphore:
        result = await runtime.stt_router.transcribe(path, options)
    return stt_response(result, options.request_id)


@router.websocket("/stt/stream")
async def stt_stream(
    ws: WebSocket,
    api_key: str | None = Query(default=None),
) -> None:
    """Rolling-window streaming STT. Client sends binary PCM16 mono @16kHz frames; server emits
    {type:"partial", text} as words stabilize and {type:"final", text} on flush. Auth via the
    X-API-Key header (core relay) or ?api_key=client_id:secret query param (browser fallback).
    Decodes are serialized through the STT slot."""
    await ws.accept()

    # Prefer the header (core relay uses it); fall back to the query param for direct browser clients.
    key = ws.headers.get("x-api-key") or api_key
    try:
        validate_key(key)
    except HTTPException as exc:
        await ws.send_json({"type": "error", "detail": exc.detail})
        await ws.close(code=4001)
        return

    if runtime.stt_service is None:
        await ws.send_json({"type": "error", "detail": "STT model not loaded"})
        await ws.close()
        return

    session = StreamingSttSession(runtime.stt_service)
    last_speech_time: list[float] = [time.monotonic()]
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
        if closed or runtime.stt_semaphore.locked():
            return
        try:
            async with runtime.stt_semaphore:
                result = await run_in_threadpool(session.decode)
            text = collapse_repeats((session.committed_text + " " + result["partial"]).strip())
            await send_json({"type": "partial", "text": text})
        except Exception:  # noqa: BLE001
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
            control: dict = {}
            if text:
                if text == "flush":
                    control = {"type": "flush"}
                else:
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            control = parsed
                    except json.JSONDecodeError:
                        await send_json({"type": "error", "detail": "invalid control message"})
                        continue
            control_type = control.get("type")
            if control_type == "config":
                try:
                    config_payload = {key: value for key, value in control.items() if key != "type"}
                    session.configure(SttDecodeContext.model_validate(config_payload))
                except ValueError:
                    await send_json({"type": "error", "detail": "invalid STT config"})
                continue
            if control_type == "reset":
                session.reset()
                continue
            if control_type == "flush":
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
    except Exception:  # noqa: BLE001
        log.exception("stt stream error")
        await send_json({"type": "error", "detail": "stream failed"})
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
