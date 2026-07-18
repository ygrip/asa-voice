"""Unit tests for CueService (setara-nx07.3, plan §9.2 resolution order + §10.6 mismatch policy)."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import runtime
from app.config import settings
from app.providers.base import TtsResult
from app.services.cue_service import CueService, CueUnavailableError


class _FakeTtsRouter:
    def __init__(self, audio_bytes=b"RIFFfake", provider="pocket_tts", model="pocket-low"):
        self._audio_bytes = audio_bytes
        self._provider = provider
        self._model = model
        self.calls = 0

    async def synthesize(self, text, options):
        self.calls += 1
        path = _write_temp_wav(self._audio_bytes)
        return TtsResult(
            provider=self._provider, model=self._model, audio_path=str(path),
            content_type="audio/wav", latency_ms=5,
        )


def _write_temp_wav(audio_bytes: bytes) -> Path:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    with open(fd, "wb") as f:
        f.write(audio_bytes)
    return Path(path)


@pytest.fixture(autouse=True)
def isolated_cue_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cue_embedded_pack_dir", str(tmp_path / "embedded"))
    monkeypatch.setattr(settings, "cue_runtime_cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cue_runtime_regeneration", False)
    monkeypatch.setattr(settings, "cue_pack_strict_match", True)
    monkeypatch.setattr(settings, "cue_pack_mismatch_policy", "fail")
    original_router = runtime.tts_router
    yield
    runtime.tts_router = original_router


def test_get_cue_raises_404_for_unknown_cue_id() -> None:
    service = CueService()
    with pytest.raises(CueUnavailableError) as exc:
        asyncio.run(service.get_cue("asa_default", "not_a_cue"))
    assert exc.value.status_code == 404


def test_get_cue_raises_404_for_unknown_voice_id() -> None:
    service = CueService()
    with pytest.raises(CueUnavailableError) as exc:
        asyncio.run(service.get_cue("not_a_voice", "listening"))
    assert exc.value.status_code == 404


def test_get_cue_raises_503_when_nothing_available_and_regeneration_disabled() -> None:
    service = CueService()
    with pytest.raises(CueUnavailableError) as exc:
        asyncio.run(service.get_cue("asa_default", "listening"))
    assert exc.value.status_code == 503


def _write_cache_manifest(cache_dir: Path, cue_id: str, provider=None, model=None) -> None:
    manifest = {"provider": provider or settings.tts_provider, "model": model or settings.tts_default_model}
    (cache_dir / f"{cue_id}.json").write_text(json.dumps(manifest))


def test_get_cue_serves_from_disk_cache() -> None:
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFcached")
    _write_cache_manifest(cache_dir, "listening")

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFcached"
    assert clip.content_type == "audio/wav"
    assert clip.cue_pack_fingerprint == "runtime-cache"


def test_get_cue_caches_result_in_memory_after_first_lookup() -> None:
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFcached")
    _write_cache_manifest(cache_dir, "listening")

    service = CueService()
    first = asyncio.run(service.get_cue("asa_default", "listening"))

    # Remove the disk file - a memory-cache hit must not need to re-read it.
    (cache_dir / "listening.wav").unlink()
    second = asyncio.run(service.get_cue("asa_default", "listening"))

    assert first is second


def test_get_cue_regenerates_when_enabled_and_writes_disk_cache(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_runtime_regeneration", True)
    fake_router = _FakeTtsRouter(audio_bytes=b"RIFFgenerated")
    runtime.tts_router = fake_router

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFgenerated"
    assert clip.cue_pack_fingerprint == "runtime-regenerated"
    assert fake_router.calls == 1
    cached_path = Path(settings.cue_runtime_cache_dir) / "asa_default" / "listening.wav"
    assert cached_path.read_bytes() == b"RIFFgenerated"


def test_disk_cache_writes_a_manifest_that_matches_the_generating_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_runtime_regeneration", True)
    runtime.tts_router = _FakeTtsRouter(audio_bytes=b"RIFFgenerated", provider="openai", model="tts-1")

    service = CueService()
    asyncio.run(service.get_cue("asa_default", "listening"))

    manifest_path = Path(settings.cue_runtime_cache_dir) / "asa_default" / "listening.json"
    assert json.loads(manifest_path.read_text()) == {"provider": "openai", "model": "tts-1"}


def test_disk_cache_is_rejected_as_stale_when_the_provider_has_since_changed() -> None:
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFstale")
    _write_cache_manifest(cache_dir, "listening", provider="openai", model="tts-1")
    # settings.tts_provider defaults to pocket_tts/pocket-low - the manifest above claims openai.

    service = CueService()
    with pytest.raises(CueUnavailableError) as exc:
        asyncio.run(service.get_cue("asa_default", "listening"))
    assert exc.value.status_code == 503


def test_disk_cache_without_a_manifest_is_treated_as_unverifiable() -> None:
    """A clip cached before this manifest existed has no way to prove its provider - must not be
    trusted forever just because it predates the check (setara-e93g)."""
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFlegacy")

    service = CueService()
    with pytest.raises(CueUnavailableError):
        asyncio.run(service.get_cue("asa_default", "listening"))


def test_disk_cache_mismatch_served_anyway_under_ignore_policy(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_pack_mismatch_policy", "ignore")
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFstale")
    _write_cache_manifest(cache_dir, "listening", provider="openai", model="tts-1")

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFstale"


def test_disk_cache_mismatch_ignored_when_strict_match_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_pack_strict_match", False)
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFlegacy")
    # No manifest at all - strict_match=False should trust it without one.

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFlegacy"


def test_get_cue_raises_503_when_regeneration_enabled_but_no_tts_router(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_runtime_regeneration", True)
    runtime.tts_router = None

    service = CueService()
    with pytest.raises(CueUnavailableError) as exc:
        asyncio.run(service.get_cue("asa_default", "listening"))
    assert exc.value.status_code == 503


def test_embedded_pack_is_served_when_it_matches_active_config() -> None:
    pack_dir = Path(settings.cue_embedded_pack_dir)
    (pack_dir / "asa_default").mkdir(parents=True)
    (pack_dir / "asa_default" / "listening.wav").write_bytes(b"RIFFembedded")
    manifest = {"fingerprint": "sha256:abc", "provider": settings.tts_provider, "model": settings.tts_default_model}
    (pack_dir / "cue-pack.json").write_text(json.dumps(manifest))

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFembedded"
    assert clip.cue_pack_fingerprint == "sha256:abc"


def test_embedded_pack_mismatch_falls_through_to_disk_cache() -> None:
    pack_dir = Path(settings.cue_embedded_pack_dir)
    (pack_dir / "asa_default").mkdir(parents=True)
    (pack_dir / "asa_default" / "listening.wav").write_bytes(b"RIFFembedded")
    manifest = {"fingerprint": "sha256:abc", "provider": "openai", "model": "tts-1"}
    (pack_dir / "cue-pack.json").write_text(json.dumps(manifest))

    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFcached")
    _write_cache_manifest(cache_dir, "listening")

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFcached"


def test_embedded_pack_mismatch_served_anyway_under_ignore_policy(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cue_pack_mismatch_policy", "ignore")
    pack_dir = Path(settings.cue_embedded_pack_dir)
    (pack_dir / "asa_default").mkdir(parents=True)
    (pack_dir / "asa_default" / "listening.wav").write_bytes(b"RIFFembedded")
    manifest = {"fingerprint": "sha256:abc", "provider": "openai", "model": "tts-1"}
    (pack_dir / "cue-pack.json").write_text(json.dumps(manifest))

    service = CueService()
    clip = asyncio.run(service.get_cue("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFembedded"


def test_regenerate_bypasses_all_cache_tiers() -> None:
    cache_dir = Path(settings.cue_runtime_cache_dir) / "asa_default"
    cache_dir.mkdir(parents=True)
    (cache_dir / "listening.wav").write_bytes(b"RIFFstale")
    fake_router = _FakeTtsRouter(audio_bytes=b"RIFFfresh")
    runtime.tts_router = fake_router

    service = CueService()
    clip = asyncio.run(service.regenerate("asa_default", "listening"))

    assert clip.audio_bytes == b"RIFFfresh"
    assert fake_router.calls == 1
