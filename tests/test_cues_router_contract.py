"""Unit tests for the /cues, /cues/{voice}/{cue}, and /internal/cues/regenerate endpoints
(setara-nx07.3, plan §9.2)."""
import asyncio
import json

import pytest
from fastapi import HTTPException

from app import runtime
from app.config import settings
from app.providers.base import TtsResult
from app.routers import cues as cues_router
from app.services.cue_service import CueService


class _FakeTtsRouter:
    def __init__(self, audio_bytes=b"RIFFgenerated"):
        self._audio_bytes = audio_bytes

    async def synthesize(self, text, options):
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".wav")
        with open(fd, "wb") as f:
            f.write(self._audio_bytes)
        return TtsResult(
            provider="pocket_tts", model="pocket-low", audio_path=path,
            content_type="audio/wav", latency_ms=5,
        )


@pytest.fixture(autouse=True)
def isolated_cue_service(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cue_embedded_pack_dir", str(tmp_path / "embedded"))
    monkeypatch.setattr(settings, "cue_runtime_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cue_runtime_regeneration", False)
    original_service = runtime.cue_service
    original_router = runtime.tts_router
    runtime.cue_service = CueService()
    yield
    runtime.cue_service = original_service
    runtime.tts_router = original_router


def test_cues_lists_all_definitions_and_voice_ids() -> None:
    response = asyncio.run(cues_router.cues(_client_id="test"))

    cue_ids = {cue.id for cue in response.cues}
    assert cue_ids == {"listening", "processing", "ok", "sorry"}
    assert set(response.voiceIds) == {"asa_default", "asa_bright", "asa_calm"}


def test_cue_clip_returns_audio_with_headers(tmp_path) -> None:
    cache_dir = tmp_path / "cache" / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFcached")
    (cache_dir / "listening.json").write_text(
        json.dumps({"provider": settings.tts_provider, "model": settings.tts_default_model})
    )

    response = asyncio.run(
        cues_router.cue_clip("asa_default", "listening", _client_id="test")
    )

    assert response.body == b"RIFFcached"
    assert response.media_type == "audio/wav"
    assert response.headers["Cache-Control"] == "private, max-age=86400"
    assert response.headers["X-ASA-TTS-Provider"] == settings.tts_provider
    assert "ETag" in response.headers
    assert "X-ASA-Cue-Pack" in response.headers


def test_cue_clip_maps_unavailable_cue_to_http_exception() -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(cues_router.cue_clip("asa_default", "not_a_cue", _client_id="test"))
    assert exc.value.status_code == 404


def test_cue_clip_returns_503_when_nothing_available() -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(cues_router.cue_clip("asa_default", "listening", _client_id="test"))
    assert exc.value.status_code == 503


def test_regenerate_rejects_production_tier_clients(monkeypatch) -> None:
    monkeypatch.setattr("app.routers.cues.get_client_tier", lambda client_id: "production")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            cues_router.regenerate_cue("asa_default", "listening", _client_id="prod-client")
        )
    assert exc.value.status_code == 403


def test_regenerate_allows_development_tier_clients(monkeypatch) -> None:
    monkeypatch.setattr("app.routers.cues.get_client_tier", lambda client_id: "development")
    runtime.tts_router = _FakeTtsRouter(audio_bytes=b"RIFFregenerated")

    response = asyncio.run(
        cues_router.regenerate_cue("asa_default", "listening", _client_id="dev-client")
    )

    assert response.body == b"RIFFregenerated"
