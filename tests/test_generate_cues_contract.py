"""Unit tests for scripts/generate_cues.py (setara-nx07.4, plan §10)."""
import asyncio
import importlib.util
import struct
import sys
import wave
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("generate_cues", SCRIPTS_DIR / "generate_cues.py")
generate_cues = importlib.util.module_from_spec(_spec)
sys.modules["generate_cues"] = generate_cues
_spec.loader.exec_module(generate_cues)


def _make_wav_bytes(duration_ms: int = 500, sample_rate: int = 24_000, channels: int = 1) -> bytes:
    import io

    frame_count = int(sample_rate * duration_ms / 1000)
    samples = struct.pack(f"<{frame_count * channels}h", *([1000] * frame_count * channels))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return buf.getvalue()


class _FakeAdapter:
    def __init__(self, audio_bytes: bytes | None = None):
        self._audio_bytes = audio_bytes or _make_wav_bytes()
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text, options):
        import tempfile

        self.calls.append((options.voice_id, text))
        fd, path = tempfile.mkstemp(suffix=".wav")
        with open(fd, "wb") as f:
            f.write(self._audio_bytes)
        return _fake_result(path)


def _fake_result(path: str):
    from app.providers.base import TtsResult

    return TtsResult(provider="openai", model="tts-1", audio_path=path, content_type="audio/wav", latency_ms=1)


# --- fingerprint --------------------------------------------------------------------------


def test_fingerprint_is_deterministic_for_identical_inputs() -> None:
    kwargs = dict(
        provider="openai", model="tts-1", voice_profile="default", speed=1.0,
        instructions=None, output_format="wav", voice_refs={"asa_default": "alloy"},
    )
    assert generate_cues.compute_fingerprint(**kwargs) == generate_cues.compute_fingerprint(**kwargs)


@pytest.mark.parametrize("changed_key,changed_value", [
    ("model", "tts-1-hd"),
    ("speed", 1.2),
    ("instructions", "speak warmly"),
    ("output_format", "pcm"),
])
def test_fingerprint_changes_when_any_input_changes(changed_key, changed_value) -> None:
    base = dict(
        provider="openai", model="tts-1", voice_profile="default", speed=1.0,
        instructions=None, output_format="wav", voice_refs={"asa_default": "alloy"},
    )
    changed = dict(base, **{changed_key: changed_value})
    assert generate_cues.compute_fingerprint(**base) != generate_cues.compute_fingerprint(**changed)


def test_fingerprint_changes_when_voice_refs_change() -> None:
    base = dict(
        provider="openai", model="tts-1", voice_profile="default", speed=1.0,
        instructions=None, output_format="wav", voice_refs={"asa_default": "alloy"},
    )
    changed = dict(base, voice_refs={"asa_default": "shimmer"})
    assert generate_cues.compute_fingerprint(**base) != generate_cues.compute_fingerprint(**changed)


# --- WAV validation ------------------------------------------------------------------------


def test_validate_wav_file_accepts_a_well_formed_clip(tmp_path) -> None:
    path = tmp_path / "ok.wav"
    path.write_bytes(_make_wav_bytes(duration_ms=500))
    assert generate_cues.validate_wav_file(path, max_duration_ms=1800) == []


def test_validate_wav_file_rejects_missing_file(tmp_path) -> None:
    errors = generate_cues.validate_wav_file(tmp_path / "missing.wav", max_duration_ms=1800)
    assert errors and "missing or empty" in errors[0]


def test_validate_wav_file_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    errors = generate_cues.validate_wav_file(path, max_duration_ms=1800)
    assert errors and "missing or empty" in errors[0]


def test_validate_wav_file_rejects_garbage_bytes(tmp_path) -> None:
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"not a wav file at all")
    errors = generate_cues.validate_wav_file(path, max_duration_ms=1800)
    assert errors and "not a valid WAV" in errors[0]


def test_validate_wav_file_rejects_stereo(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    path.write_bytes(_make_wav_bytes(channels=2))
    errors = generate_cues.validate_wav_file(path, max_duration_ms=1800)
    assert any("mono" in e for e in errors)


def test_validate_wav_file_rejects_unexpected_sample_rate(tmp_path) -> None:
    path = tmp_path / "weird_rate.wav"
    path.write_bytes(_make_wav_bytes(sample_rate=12_345))
    errors = generate_cues.validate_wav_file(path, max_duration_ms=1800)
    assert any("sample rate" in e for e in errors)


def test_validate_wav_file_rejects_clip_exceeding_max_duration(tmp_path) -> None:
    path = tmp_path / "too_long.wav"
    path.write_bytes(_make_wav_bytes(duration_ms=5000))
    errors = generate_cues.validate_wav_file(path, max_duration_ms=1800)
    assert any("exceeds max" in e for e in errors)


# --- end-to-end generate() + validate_pack() -----------------------------------------------


def test_generate_writes_a_clip_per_voice_and_cue(tmp_path, monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(generate_cues, "build_adapter", lambda provider: fake)

    manifest = asyncio.run(
        generate_cues.generate(
            provider="openai", model="tts-1", voice_profile="default", speed=1.0,
            instructions=None, output_format="wav", out_dir=tmp_path,
        )
    )

    assert manifest["provider"] == "openai"
    assert manifest["model"] == "tts-1"
    assert set(manifest["files"].keys()) == {
        f"{voice}/{cue}.wav"
        for voice in ("asa_default", "asa_bright", "asa_calm")
        for cue in ("listening", "processing", "ok", "sorry")
    }
    assert (tmp_path / "cue-pack.json").is_file()
    assert (tmp_path / "asa_default" / "listening.wav").is_file()
    assert len(fake.calls) == 12  # 3 voices x 4 cues


def test_generate_records_correct_provider_voice_ref_per_file(tmp_path, monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(generate_cues, "build_adapter", lambda provider: fake)

    manifest = asyncio.run(
        generate_cues.generate(
            provider="openai", model="tts-1", voice_profile="default", speed=1.0,
            instructions=None, output_format="wav", out_dir=tmp_path,
        )
    )

    assert manifest["files"]["asa_bright/listening.wav"]["providerVoice"] == "shimmer"
    assert manifest["files"]["asa_default/listening.wav"]["providerVoice"] == "alloy"


def test_validate_pack_passes_for_a_freshly_generated_pack(tmp_path, monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(generate_cues, "build_adapter", lambda provider: fake)

    manifest = asyncio.run(
        generate_cues.generate(
            provider="openai", model="tts-1", voice_profile="default", speed=1.0,
            instructions=None, output_format="wav", out_dir=tmp_path,
        )
    )

    assert generate_cues.validate_pack(tmp_path, manifest) == []


def test_validate_pack_flags_checksum_mismatch_on_a_tampered_file(tmp_path, monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(generate_cues, "build_adapter", lambda provider: fake)

    manifest = asyncio.run(
        generate_cues.generate(
            provider="openai", model="tts-1", voice_profile="default", speed=1.0,
            instructions=None, output_format="wav", out_dir=tmp_path,
        )
    )
    (tmp_path / "asa_default" / "listening.wav").write_bytes(_make_wav_bytes(duration_ms=100))

    errors = generate_cues.validate_pack(tmp_path, manifest)
    assert any("checksum mismatch" in e for e in errors)


def test_generate_raises_for_a_provider_with_no_catalog_mapping(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generate_cues.voice_catalog, "entries", lambda: [])

    with pytest.raises(ValueError, match="No catalog voice"):
        asyncio.run(
            generate_cues.generate(
                provider="openai", model="tts-1", voice_profile="default", speed=1.0,
                instructions=None, output_format="wav", out_dir=tmp_path,
            )
        )
