"""Cue service (setara-nx07.3): asa-voice owns cue audio end-to-end - definitions, generation
fallback, caching, and validation. setara-core only proxies these endpoints (plan §11); it no
longer bundles cue WAVs or runs its own synthesis-on-miss.

Resolution order (plan §9.2):
  in-memory bytes -> embedded build-generated cue pack -> runtime disk cache -> runtime
  regeneration (CUE_RUNTIME_REGENERATION=true, development only).

The embedded pack tier is populated by the build-time generator (setara-nx07.4) and is simply
absent until that PR lands - every lookup there misses cleanly and falls through to the next tier.
"""
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.providers.base import TtsOptions
from app.services.cue_definitions import CUE_DEFINITIONS
from app.services.voice_catalog import voice_ids

log = logging.getLogger("asa.cues")


class CueUnavailableError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class CueClip:
    audio_bytes: bytes
    content_type: str
    etag: str
    cue_pack_fingerprint: str
    tts_provider: str
    tts_model: str


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CueService:
    """One instance is shared for the process lifetime (app.runtime.cue_service) so the in-memory
    tier and disk cache actually persist across requests."""

    def __init__(self):
        self._memory_cache: dict[tuple[str, str], CueClip] = {}

    def list_cue_ids(self) -> list[str]:
        return list(CUE_DEFINITIONS.keys())

    async def get_cue(self, voice_id: str, cue_id: str) -> CueClip:
        if cue_id not in CUE_DEFINITIONS:
            raise CueUnavailableError(404, f"Unknown cue '{cue_id}'")
        if voice_id not in voice_ids():
            raise CueUnavailableError(404, f"Unknown voice '{voice_id}'")

        key = (voice_id, cue_id)
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached

        clip = self._load_embedded(voice_id, cue_id) or self._load_disk_cache(voice_id, cue_id)
        if clip is None:
            clip = await self._maybe_regenerate(voice_id, cue_id)
        if clip is None:
            raise CueUnavailableError(
                503, f"Cue '{cue_id}' for voice '{voice_id}' is not available"
            )

        self._memory_cache[key] = clip
        return clip

    def _load_embedded(self, voice_id: str, cue_id: str) -> CueClip | None:
        pack_dir = Path(settings.cue_embedded_pack_dir)
        manifest_path = pack_dir / "cue-pack.json"
        clip_path = pack_dir / voice_id / f"{cue_id}.wav"
        if not manifest_path.is_file() or not clip_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            log.warning("Embedded cue-pack.json is unreadable; ignoring embedded pack")
            return None

        matches = _pack_matches_active_config(manifest)
        if not matches and settings.cue_pack_mismatch_policy != "ignore":
            log.warning(
                "Embedded cue pack fingerprint does not match active TTS config "
                "(policy=%s) - falling through to runtime tiers",
                settings.cue_pack_mismatch_policy,
            )
            return None

        audio_bytes = clip_path.read_bytes()
        return CueClip(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            etag=_sha256_hex(audio_bytes),
            cue_pack_fingerprint=str(manifest.get("fingerprint", "")),
            tts_provider=str(manifest.get("provider", settings.tts_provider)),
            tts_model=str(manifest.get("model", "")),
        )

    def _load_disk_cache(self, voice_id: str, cue_id: str) -> CueClip | None:
        cache_dir = Path(settings.cue_runtime_cache_dir) / voice_id
        clip_path = cache_dir / f"{cue_id}.wav"
        if not clip_path.is_file():
            return None

        # A clip cached under a since-changed TTS_PROVIDER/model must not be served forever just
        # because the volume persisted across a config change (setara-e93g) - validate it the same
        # way the embedded pack tier does, falling through to regeneration/unavailable on mismatch
        # instead of silently serving audio from the wrong provider. A missing manifest (clip
        # cached before this check existed) counts as unverifiable, not a pass.
        manifest = _read_json_or_none(cache_dir / f"{cue_id}.json")
        if settings.cue_pack_strict_match:
            matches = manifest is not None and _pack_matches_active_config(manifest)
            if not matches and settings.cue_pack_mismatch_policy != "ignore":
                log.warning(
                    "Runtime cue cache for %s/%s does not match active TTS config (policy=%s) - "
                    "treating as stale",
                    voice_id, cue_id, settings.cue_pack_mismatch_policy,
                )
                return None

        audio_bytes = clip_path.read_bytes()
        manifest = manifest or {}
        return CueClip(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            etag=_sha256_hex(audio_bytes),
            cue_pack_fingerprint=str(manifest.get("fingerprint", "runtime-cache")),
            tts_provider=str(manifest.get("provider", settings.tts_provider)),
            tts_model=str(manifest.get("model", _active_tts_model())),
        )

    async def _maybe_regenerate(self, voice_id: str, cue_id: str) -> CueClip | None:
        if not settings.cue_runtime_regeneration:
            return None
        return await self.regenerate(voice_id, cue_id)

    async def regenerate(self, voice_id: str, cue_id: str) -> CueClip:
        """Synthesize a cue clip live via the active TTS provider and cache it to disk. Called by
        the runtime-regeneration fallback tier, and directly by the admin-only
        POST /internal/cues/regenerate endpoint."""
        from app import runtime

        definition = CUE_DEFINITIONS.get(cue_id)
        if definition is None:
            raise CueUnavailableError(404, f"Unknown cue '{cue_id}'")
        if runtime.tts_router is None:
            raise CueUnavailableError(503, "No TTS provider is available to generate cues")

        options = TtsOptions(voice_id=voice_id, format="wav", purpose="cue")
        result = await runtime.tts_router.synthesize(definition.text, options)
        try:
            with open(result.audio_path, "rb") as f:
                audio_bytes = f.read()
        finally:
            try:
                os.remove(result.audio_path)
            except OSError:
                pass

        self._write_disk_cache(voice_id, cue_id, audio_bytes, provider=result.provider, model=result.model)
        clip = CueClip(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            etag=_sha256_hex(audio_bytes),
            cue_pack_fingerprint="runtime-regenerated",
            tts_provider=result.provider,
            tts_model=result.model,
        )
        self._memory_cache[(voice_id, cue_id)] = clip
        return clip

    def _write_disk_cache(self, voice_id: str, cue_id: str, audio_bytes: bytes, *, provider: str, model: str) -> None:
        cache_dir = Path(settings.cue_runtime_cache_dir) / voice_id
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{cue_id}.wav").write_bytes(audio_bytes)
            # Sidecar manifest so _load_disk_cache() can detect a stale clip after the active TTS
            # provider/model changes (setara-e93g) instead of serving it forever.
            (cache_dir / f"{cue_id}.json").write_text(json.dumps({"provider": provider, "model": model}))
        except OSError:
            log.warning("Could not write runtime cue cache for %s/%s", voice_id, cue_id)


def _read_json_or_none(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _pack_matches_active_config(manifest: dict) -> bool:
    if not settings.cue_pack_strict_match:
        return True
    return (
        manifest.get("provider") == settings.tts_provider
        and manifest.get("model") == _active_tts_model()
    )


def _active_tts_model() -> str:
    if settings.tts_provider == "openai":
        return settings.openai_tts_model
    return settings.tts_default_model
