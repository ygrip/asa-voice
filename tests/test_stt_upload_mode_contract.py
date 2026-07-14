"""setara-w50k.3: the batch-upload /stt route (revived for command-mode MediaRecorder capture,
replacing the rolling-PCM WS stream for that mode) must use the mode-aware final-decode profile
(profile_for_mode in stt_service.py), not the old fixed stt_final_* settings — otherwise a batch
command upload gets dictation's slower/more-cautious profile or vice versa."""
import pytest


def test_transcribe_uses_profile_for_mode_not_fixed_final_settings() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "services" / "stt_service.py").read_text()
    body = source.split("def transcribe(", 1)[1].split("def decode_words", 1)[0]
    assert "profile_for_mode(mode)" in body
    assert "beam_size=profile.beam_size" in body
    assert "temperature=profile.temperatures" in body
    assert "settings.stt_final_beam_size" not in body


def test_stt_route_forwards_mode_field_to_options() -> None:
    """Behavioral: POST /stt with mode=dictation must reach SttOptions.mode, so the router (and
    ultimately profile_for_mode) sees the caller's mode instead of silently defaulting to command."""
    pytest.importorskip("faster_whisper")
    pytest.importorskip("numpy")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import runtime
    from app.auth import require_api_key
    from app.providers.base import SttResult
    from app.routers import stt
    from app.services import audio_service

    seen_modes = []

    class FakeSttRouter:
        async def transcribe(self, path, options, provider_override=None):
            seen_modes.append(options.mode)
            return SttResult(
                provider="faster_whisper", model="test", text="ok", language="en",
                duration_ms=1000, latency_ms=1, segments=[],
            )

    orig_router = runtime.stt_router
    orig_probe = audio_service.probe_duration_seconds
    runtime.stt_router = FakeSttRouter()
    audio_service.probe_duration_seconds = lambda path: 2.0
    app = FastAPI()
    app.include_router(stt.router)
    app.dependency_overrides[require_api_key] = lambda: "test"
    try:
        client = TestClient(app)

        resp = client.post("/stt", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        assert resp.status_code == 200, resp.text

        resp_dictation = client.post(
            "/stt", files={"file": ("u.wav", b"RIFFfake", "audio/wav")}, data={"mode": "dictation"}
        )
        assert resp_dictation.status_code == 200, resp_dictation.text

        assert seen_modes == ["command", "dictation"]
    finally:
        runtime.stt_router = orig_router
        audio_service.probe_duration_seconds = orig_probe
