"""setara-gwfj: long PTT dictation must transcribe fully.

The /stt upload path relies on faster-whisper to chunk long audio internally, so no per-utterance
length cap is needed there — our code must only avoid (a) a duration limit below a usable length
or (b) truncating the collected segments. Verified empirically with a 63.5s WAV (see bd setara-gwfj
notes: durationSeconds=63.5, 22 segments 0.1s->60.3s, full text). These tests guard that path.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_upload_duration_cap_allows_long_dictation() -> None:
    # A single PTT utterance can run to a minute+; the upload cap must stay well above that.
    source = (ROOT / "app" / "config.py").read_text()
    m = re.search(r"max_upload_seconds:\s*int\s*=\s*(\d+)", source)
    assert m, "max_upload_seconds default not found in config.py"
    assert int(m.group(1)) >= 60, "max_upload_seconds too low for long dictation"


def test_upload_route_bounds_duration_via_configurable_guard() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()
    stt_route = source.split('@router.post("/stt"', 1)[1].split("@router.websocket", 1)[0]
    # Duration is bounded by the configurable guard (max_upload_seconds), never a literal short cap.
    assert "audio_service.enforce_duration(path)" in stt_route


def test_transcribe_collects_all_segments_without_truncation() -> None:
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    body = source.split("def transcribe(", 1)[1].split("def decode_words", 1)[0]
    assert "for segment in segments:" in body
    assert "full_text.append(text)" in body
    # No slice/limit that would silently drop the later segments of a long transcript.
    assert "segments[:" not in body
    assert "break" not in body


def test_upload_route_returns_full_multisegment_text() -> None:
    """Behavioral: a mocked >30s decode with many segments returns the full concatenated text
    through the real route + response model — proving our layer never truncates. Skipped where the
    router's heavy deps (faster-whisper/numpy) aren't installed; runs in the container/CI."""
    pytest.importorskip("faster_whisper")
    pytest.importorskip("numpy")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import runtime
    from app.auth import require_api_key
    from app.routers import stt
    from app.services import audio_service

    segments = [
        {"start": float(i * 3), "end": float(i * 3 + 3), "text": f"segment number {i}"}
        for i in range(15)  # ~45s of speech across 15 segments
    ]
    full_text = " ".join(s["text"] for s in segments)

    class FakeStt:
        def transcribe(self, path, language=None, vad=None, context=None):  # noqa: D401
            return {
                "text": full_text, "segments": segments, "language": "en",
                "durationSeconds": 45.0, "engine": "faster-whisper", "model": "test",
            }

    orig_service = runtime.stt_service
    orig_probe = audio_service.probe_duration_seconds
    runtime.stt_service = FakeStt()
    audio_service.probe_duration_seconds = lambda path: 45.0  # under the cap; skip ffprobe
    app = FastAPI()
    app.include_router(stt.router)
    app.dependency_overrides[require_api_key] = lambda: "test"
    try:
        client = TestClient(app)
        resp = client.post("/stt", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        assert resp.status_code == 200, resp.text
        assert resp.json()["text"] == full_text  # every segment survived, in order
    finally:
        runtime.stt_service = orig_service
        audio_service.probe_duration_seconds = orig_probe
