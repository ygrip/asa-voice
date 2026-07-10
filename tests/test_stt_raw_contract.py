from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_raw_stt_validates_pcm_contract_and_bounds() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert '@router.post("/stt/raw", response_model=SttResponse)' in source
    assert '!= "audio/l16"' in source
    assert "x_sample_rate != 16000" in source
    assert 'x_sample_format.lower() != "s16le"' in source
    assert "len(body) % 2 != 0" in source
    assert "settings.max_audio_seconds * 16000" in source
    assert "stt_router.transcribe_array(audio, options)" in source


def test_raw_stt_context_is_base64url_and_bounded() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert "base64.urlsafe_b64decode" in source
    assert "len(decoded) > 4096" in source
    assert "Invalid X-Stt-Context" in source


def test_health_exposes_runtime_model_metadata() -> None:
    source = (ROOT / "app" / "routers" / "health.py").read_text()

    assert "artifactReady=artifact_ready" in source
    assert "computeType=settings.stt_compute_type" in source
    assert "sampleRate=settings.tts_sample_rate" in source
    assert 'model_path / ".asa_model_ready"' in source
