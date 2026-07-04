from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tts_endpoints_validate_their_distinct_formats() -> None:
    source = (ROOT / "app" / "routers" / "tts.py").read_text()

    assert 'req.format.lower() != "wav"' in source
    assert 'req.format.lower() not in {"pcm", "l16"}' in source
    assert "Unsupported format; expected wav" in source
    assert "Unsupported format; expected pcm or l16" in source


def test_tts_stream_declares_pcm_wire_contract() -> None:
    source = (ROOT / "app" / "routers" / "tts.py").read_text()

    assert 'media_type="audio/l16"' in source
    assert '"X-Sample-Format": "s16le"' in source
    assert '"Cache-Control": "no-store, no-transform"' in source
    assert '"X-Content-Type-Options": "nosniff"' in source
