from fastapi import APIRouter, Depends, HTTPException, Response

from app import runtime
from app.auth import OVERRIDE_ALLOWED_TIERS, get_client_tier, require_api_key
from app.schemas import CueInfo, CuesResponse
from app.services.cue_definitions import CUE_DEFINITIONS
from app.services.cue_service import CueUnavailableError
from app.services.voice_catalog import voice_ids

router = APIRouter()


@router.get("/cues", response_model=CuesResponse)
async def cues(_client_id: str = Depends(require_api_key)) -> CuesResponse:
    return CuesResponse(
        cues=[
            CueInfo(id=d.id, text=d.text, maxDurationMs=d.max_duration_ms)
            for d in CUE_DEFINITIONS.values()
        ],
        voiceIds=voice_ids(),
    )


@router.get("/cues/{voice_id}/{cue_id}")
async def cue_clip(
    voice_id: str, cue_id: str, _client_id: str = Depends(require_api_key)
) -> Response:
    try:
        clip = await runtime.cue_service.get_cue(voice_id, cue_id)
    except CueUnavailableError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return Response(
        content=clip.audio_bytes,
        media_type=clip.content_type,
        headers={
            "Cache-Control": "private, max-age=86400",
            "ETag": f'"{clip.etag}"',
            "X-ASA-Cue-Pack": clip.cue_pack_fingerprint,
            "X-ASA-TTS-Provider": clip.tts_provider,
            "X-ASA-TTS-Model": clip.tts_model,
        },
    )


@router.post("/internal/cues/regenerate")
async def regenerate_cue(
    voice_id: str, cue_id: str, _client_id: str = Depends(require_api_key)
) -> Response:
    """Development/admin only (plan §9.2) - force a live resynthesis of one cue, bypassing every
    cache tier. Never exposed to production-tier clients regardless of CUE_RUNTIME_REGENERATION,
    since this is an explicit operator action, not the passive fallback tier."""
    if get_client_tier(_client_id) not in OVERRIDE_ALLOWED_TIERS:
        raise HTTPException(status_code=403, detail="Cue regeneration is restricted to dev/admin clients")

    try:
        clip = await runtime.cue_service.regenerate(voice_id, cue_id)
    except CueUnavailableError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return Response(
        content=clip.audio_bytes,
        media_type=clip.content_type,
        headers={
            "ETag": f'"{clip.etag}"',
            "X-ASA-Cue-Pack": clip.cue_pack_fingerprint,
            "X-ASA-TTS-Provider": clip.tts_provider,
            "X-ASA-TTS-Model": clip.tts_model,
        },
    )
