import asyncio
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
from app import runtime
from app.auth import provider_override_allowed, require_api_key, validate_key
from app.config import settings
from app.providers.base import SttOptions
from app.providers.errors import SttFailLoudError, SttFallbackEligibleError, SttPolicyRejectedError
from app.providers.streaming.base import SttStreamError, StreamingSttSession
from app.providers.streaming.factory import StreamingSttSessionFactory
from app.schemas import SttResponse, stt_response
from app.services import audio_service
from app.services.stt_context import SttDecodeContext
from app.services.stt_stream_protocol import (
    STT_MAX_CONTROL_BYTES,
    SttAudioFormat,
    SttErrorEvent,
    SttProtocolError,
    SttStartControl,
    parse_stt_client_control,
)
from app.services.stt_stream_scheduler import SttStreamScheduler
from app.services.operation_limiter import OperationBusyError
from app.services.voice_metrics import LABEL_VALUES, voice_metrics

router = APIRouter()
log = logging.getLogger("asa.stt")


def _options_from_context(
    context: SttDecodeContext | None,
    client_id: str,
    request_id: str | None,
    language: str | None = None,
    mode: str = "command",
) -> SttOptions:
    return SttOptions(
        language=language or (context.language if context else None),
        prompt=context.prompt if context else None,
        hotwords=context.hotwords if context else None,
        request_id=request_id or (context.request_id if context else None) or str(uuid.uuid4()),
        client_id=client_id,
        mode=mode,
    )


def _resolve_provider_override(requested: str | None, client_id: str) -> str | None:
    """Explicit provider= override (plan §6.2 / setara-s94o.8). Disabled globally unless
    STT_ALLOW_PROVIDER_OVERRIDE=true, and even then only honored for a client whose configured
    trust tier allows it (development/admin/test) - a production-tier or unknown client's
    override request is silently ignored rather than rejected, so it just gets the default
    provider instead of an error."""
    if not requested or requested == "auto":
        return None
    if not settings.stt_allow_provider_override:
        log.info("provider override '%s' requested but overrides are disabled; ignoring", requested)
        return None
    if not provider_override_allowed(client_id):
        log.info(
            "provider override '%s' requested by client '%s' whose trust tier disallows it; ignoring",
            requested, client_id,
        )
        return None
    return requested


@router.post("/stt/raw", response_model=SttResponse)
async def stt_raw(
    request: Request,
    x_sample_rate: int = Header(default=16000),
    x_channels: int = Header(default=1),
    x_sample_format: str = Header(default="s16le"),
    x_stt_context: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
    provider: str | None = Query(default=None),
    _client_id: str = Depends(require_api_key),
) -> SttResponse:
    if runtime.stt_router is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "audio/l16":
        raise HTTPException(status_code=415, detail="Expected Content-Type audio/l16")
    if x_sample_rate != 16000 or x_channels != 1 or x_sample_format.lower() != "s16le":
        raise HTTPException(status_code=415, detail="Expected PCM16 mono 16kHz s16le")
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
    provider_override = _resolve_provider_override(provider, _client_id)
    audio = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    try:
        result = await runtime.stt_router.transcribe_array(
            audio, options, provider_override=provider_override
        )
    except OperationBusyError as exc:
        raise _busy_http_error(exc) from exc
    except SttPolicyRejectedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except SttFailLoudError as exc:
        # Never falls back (setara-s94o.7) - a distinct "reason" so callers can show an
        # actionable billing/auth message instead of a generic "STT unavailable" one.
        raise HTTPException(status_code=502, detail={"reason": "stt_fail_loud", "message": str(exc)}) from exc
    except SttFallbackEligibleError as exc:
        # Reaching here means the fallback provider (if any) also failed - both are down.
        raise HTTPException(status_code=502, detail={"reason": "stt_unavailable", "message": str(exc)}) from exc
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
    provider: str | None = Form(default=None),
    mode: str = Form(default="command"),
    x_request_id: str | None = Header(default=None),
    _client_id: str = Depends(require_api_key),
) -> SttResponse:
    if runtime.stt_router is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

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
        try:
            os.remove(path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Invalid STT context") from exc

    options = _options_from_context(decode_context, _client_id, x_request_id, language, mode)
    provider_override = _resolve_provider_override(provider, _client_id)
    try:
        result = await runtime.stt_router.transcribe(
            path, options, provider_override=provider_override
        )
    except OperationBusyError as exc:
        raise _busy_http_error(exc) from exc
    except SttPolicyRejectedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except SttFailLoudError as exc:
        # Never falls back (setara-s94o.7) - a distinct "reason" so callers can show an
        # actionable billing/auth message instead of a generic "STT unavailable" one.
        raise HTTPException(status_code=502, detail={"reason": "stt_fail_loud", "message": str(exc)}) from exc
    except SttFallbackEligibleError as exc:
        # Reaching here means the fallback provider (if any) also failed - both are down.
        raise HTTPException(status_code=502, detail={"reason": "stt_unavailable", "message": str(exc)}) from exc
    finally:
        # Guaranteed cleanup regardless of which adapter handled the request - the local
        # faster-whisper engine already removes its own temp file, but a hosted provider (or a
        # provider= override picking one) never touches the filesystem path, so this must not
        # rely on the adapter to have done it.
        try:
            os.remove(path)
        except OSError:
            pass
    return stt_response(result, options.request_id)


@router.websocket("/stt/stream")
async def stt_stream(
    ws: WebSocket,
    api_key: str | None = Query(default=None),
) -> None:
    """Provider-owned v2 streaming STT with temporary compatibility for legacy config/flush."""
    await ws.accept()
    session_started = time.monotonic()

    key = ws.headers.get("x-api-key") or api_key
    try:
        client_id = validate_key(key)
    except HTTPException as exc:
        await ws.send_json({"type": "error", "detail": exc.detail})
        await ws.close(code=4001)
        return

    factory = StreamingSttSessionFactory()
    session: StreamingSttSession | None = None
    scheduler: SttStreamScheduler | None = None
    session_mode: str | None = None
    last_speech_time: list[float] = [time.monotonic()]
    last_audio_time: list[float] = [time.monotonic()]
    closed = False
    metric_session: tuple[str, str] | None = None
    send_lock = asyncio.Lock()

    async def send_json(message: dict | SttErrorEvent | object) -> bool:
        nonlocal closed
        if closed:
            return False
        payload = message
        if hasattr(message, "model_dump"):
            payload = message.model_dump(by_alias=True)
        try:
            async with send_lock:
                if closed:
                    return False
                await ws.send_json(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            closed = True
            return False

    async def send_stream_error(error: SttStreamError) -> bool:
        return await send_json(error.to_event())

    async def configure(start: SttStartControl) -> bool:
        nonlocal metric_session, scheduler, session, session_mode
        if session is not None:
            await send_stream_error(
                SttStreamError("STT_SESSION_ALREADY_CONFIGURED", "STT session is already configured")
            )
            return False
        try:
            session, options = factory.create(start, client_id)
            hotwords_count = len(options.hotwords) if options.hotwords else 0
            log.debug(
                "stt timeline: +%dms start received mode=%s provider=%s hotwords=%d prompt_len=%d",
                int((time.monotonic() - session_started) * 1000),
                options.mode,
                options.provider,
                hotwords_count,
                len(options.prompt) if options.prompt else 0,
            )
            ready = await session.configure(options)
            local_limiter = (
                runtime.local_decode_limiter if options.provider == "faster_whisper" else None
            )
            scheduler = SttStreamScheduler(
                session,
                local_limiter,
                emit_partial=lambda partial: send_json(partial),
                emit_decode_error=handle_decode_error,
                final_limiter=local_limiter,
                base_interval_ms=settings.stt_stream_interval_ms,
                max_interval_ms=settings.stt_stream_max_partial_interval_ms,
                rtf_slow_threshold=settings.stt_stream_rtf_slow_threshold,
            )
            # Command never runs a partial decoder; dictation/hands_free are also final-only by
            # default (ASA STT accuracy/latency recovery plan, RC-05) - flush() blocks on any
            # in-flight partial before the final decode can even start (`_settle_decoder` in
            # stt_stream_scheduler.py), and a slow model can turn that into 7-12s+ of pure wait on
            # top of the final decode itself. Not starting the decoder loop at all skips that decode
            # - and the wait - entirely. Live partial captions are opt-in per mode via
            # STT_DICTATION_PARTIALS_ENABLED/STT_HANDSFREE_PARTIALS_ENABLED for deployments willing
            # to pay the cost (widen STT_STREAM_INTERVAL_MS/STT_STREAM_MAX_PARTIAL_INTERVAL_MS too).
            partials_enabled = {
                "command": False,
                "dictation": settings.stt_dictation_partials_enabled,
                "hands_free": settings.stt_handsfree_partials_enabled,
            }.get(options.mode, False)
            if partials_enabled:
                scheduler.start()
            session_mode = options.mode
            metric_session = (_metric_provider(options.provider), options.mode)
            voice_metrics.add_gauge(
                "asa_voice_stt_sessions_active",
                1,
                provider=metric_session[0],
                mode=metric_session[1],
            )
            ready_sent = await send_json(ready)
            # Reset here, not at ws.accept() - handshake latency (model warm-up, JWT/relay round
            # trips) must not count against the silence budget before the user has said anything.
            last_speech_time[0] = time.monotonic()
            last_audio_time[0] = time.monotonic()
            return ready_sent
        except SttStreamError as exc:
            if scheduler is not None:
                await scheduler.close()
            elif session is not None:
                await session.close()
            scheduler = None
            session = None
            await send_stream_error(exc)
            return False

    async def handle_decode_error(error: Exception) -> None:
        if closed:
            return
        if isinstance(error, SttStreamError):
            await send_stream_error(error)
            return
        if isinstance(error, OperationBusyError):
            voice_metrics.increment(
                "asa_voice_stt_backpressure_events_total", layer="sidecar"
            )
            await send_stream_error(
                SttStreamError(error.code, str(error), retryable=error.retryable)
            )
            return
        log.exception("stt decode error (non-fatal)", exc_info=error)

    async def flush() -> bool:
        if session is None or scheduler is None:
            await send_stream_error(
                SttStreamError("STT_SESSION_NOT_CONFIGURED", "STT session is not configured")
            )
            return False
        try:
            flush_started = time.monotonic()
            final = await scheduler.flush()
            metrics = scheduler.metrics
            log.debug(
                "stt timeline: flush->final took %.3fs", time.monotonic() - flush_started
            )
            log.info(
                "stt stream finalized provider=%s duration_ms=%d partial_decodes=%d "
                "partial_rtf=%.3f partial_interval_ms=%d coalesced_signals=%d",
                session.state.provider,
                session.state.duration_ms,
                metrics.decode_count,
                metrics.real_time_factor,
                metrics.partial_interval_ms,
                metrics.coalesced_signals,
            )
            return await send_json(final)
        except SttStreamError as exc:
            await send_stream_error(exc)
            return False
        except OperationBusyError as exc:
            await send_stream_error(
                SttStreamError(exc.code, str(exc), retryable=exc.retryable)
            )
            return False
        except Exception:  # noqa: BLE001
            if closed:
                return False
            log.exception("stt flush error")
            await send_stream_error(
                SttStreamError("STT_STREAM_FAILED", "Streaming STT finalization failed", retryable=True)
            )
            return False

    async def reader_loop() -> None:
        nonlocal closed
        while True:
            if session is None:
                receive_timeout = settings.stt_stream_handshake_timeout_seconds
                timeout_error = SttStreamError(
                    "STT_HANDSHAKE_TIMEOUT",
                    "STT start handshake timed out",
                    retryable=True,
                )
            else:
                elapsed = time.monotonic() - last_audio_time[0]
                receive_timeout = max(
                    0.001,
                    settings.stt_stream_no_audio_idle_timeout_seconds - elapsed,
                )
                timeout_error = SttStreamError(
                    "STT_NO_AUDIO_IDLE_TIMEOUT",
                    "STT stream received no audio before its idle deadline",
                    retryable=True,
                )
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=receive_timeout)
            except asyncio.TimeoutError:
                await send_stream_error(timeout_error)
                break
            if msg.get("type") == "websocket.disconnect":
                closed = True
                break
            data = msg.get("bytes")
            if data is not None:
                if session is None or scheduler is None:
                    await send_stream_error(
                        SttStreamError(
                            "STT_SESSION_NOT_CONFIGURED",
                            "Send a start control before PCM audio",
                        )
                    )
                    break
                try:
                    await scheduler.append_pcm(data)
                except SttStreamError as exc:
                    await send_stream_error(exc)
                    break
                last_audio_time[0] = time.monotonic()
                voice_metrics.increment(
                    "asa_voice_stt_audio_received_seconds_total",
                    len(data) / 32_000,
                    provider=_metric_provider(session.state.provider),
                )
                if session.last_frame_had_speech():
                    last_speech_time[0] = time.monotonic()
                if (
                    # Server-side VAD fallback is a hands-free-only safety net. Command/dictation
                    # are explicit_stop per the client's mode contract (mode-policy.ts) - silently
                    # auto-flushing them here truncated real utterances mid-sentence.
                    session_mode == "hands_free"
                    and settings.stt_stream_silence_flush_s > 0
                    and session.state.sequence > 0
                    and not session.last_frame_had_speech()
                    and time.monotonic() - last_speech_time[0] > settings.stt_stream_silence_flush_s
                ):
                    if await flush():
                        last_speech_time[0] = time.monotonic()
                        await scheduler.reset()
                continue

            text = msg.get("text")
            if not text:
                continue
            if text == "flush":
                log.debug(
                    "stt timeline: +%dms flush received (legacy)",
                    int((time.monotonic() - session_started) * 1000),
                )
                await flush()
                continue
            if len(text.encode("utf-8")) > STT_MAX_CONTROL_BYTES:
                await send_stream_error(
                    SttStreamError("STT_CONTROL_TOO_LARGE", "STT control exceeds 8192 bytes")
                )
                continue

            try:
                raw_control = json.loads(text)
            except json.JSONDecodeError:
                await send_stream_error(
                    SttStreamError("STT_INVALID_JSON", "STT control must be valid JSON")
                )
                continue
            if isinstance(raw_control, dict) and raw_control.get("type") == "config":
                try:
                    context_payload = {
                        field: value for field, value in raw_control.items() if field != "type"
                    }
                    context = SttDecodeContext.model_validate(context_payload)
                    await configure(_legacy_start_control(context))
                except ValueError:
                    await send_stream_error(
                        SttStreamError("STT_INVALID_MESSAGE", "Invalid legacy STT config")
                    )
                continue

            try:
                control = parse_stt_client_control(
                    text,
                    allow_provider_override=(
                        settings.stt_allow_provider_override
                        and provider_override_allowed(client_id)
                    ),
                )
            except SttProtocolError as exc:
                await send_json(
                    SttErrorEvent(
                        type="error",
                        code=exc.code,
                        message=str(exc),
                        retryable=False,
                    )
                )
                continue
            if control.type == "start":
                await configure(control)
                continue
            if session is None or scheduler is None:
                await send_stream_error(
                    SttStreamError("STT_SESSION_NOT_CONFIGURED", "STT session is not configured")
                )
                continue
            if control.type == "reset":
                await scheduler.reset()
                continue
            if control.type == "cancel":
                await scheduler.close()
                break
            if control.type == "flush":
                log.debug(
                    "stt timeline: +%dms flush received",
                    int((time.monotonic() - session_started) * 1000),
                )
                await flush()

    reader_task = asyncio.create_task(reader_loop(), name="asa-stt-websocket-reader")
    try:
        await reader_task
    except WebSocketDisconnect:
        closed = True
    except Exception:  # noqa: BLE001
        log.exception("stt stream error")
        await send_stream_error(
            SttStreamError("STT_STREAM_FAILED", "Streaming STT failed", retryable=True)
        )
    finally:
        if not reader_task.done():
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)
        if scheduler is not None:
            await scheduler.close()
        elif session is not None:
            await session.close()
        if metric_session is not None:
            voice_metrics.add_gauge(
                "asa_voice_stt_sessions_active",
                -1,
                provider=metric_session[0],
                mode=metric_session[1],
            )
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


def _legacy_start_control(context: SttDecodeContext) -> SttStartControl:
    return SttStartControl(
        type="start",
        protocolVersion=settings.stt_stream_protocol_version,
        sessionId=str(uuid.uuid4()),
        requestId=context.request_id,
        mode="hands_free",
        provider="auto",
        audio=SttAudioFormat(
            sampleRate=16_000,
            channels=1,
            sampleFormat="s16le",
            frameDurationMs=settings.stt_frame_duration_ms,
        ),
        language=context.language,
        prompt=context.prompt,
        hotwords=context.hotwords,
        maxDurationSeconds=settings.stt_handsfree_max_seconds,
    )


def _busy_http_error(error: OperationBusyError) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={"code": error.code, "message": str(error), "retryable": error.retryable},
    )


def _metric_provider(provider: str | None) -> str:
    if provider in LABEL_VALUES["provider"]:
        return provider
    return "unknown"
