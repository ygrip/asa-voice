"""Stable ASA voice catalog (setara-nx07.3): the single source of truth mapping a stable voice ID
(e.g. "asa_default") to each provider's own voice reference. Every TTS adapter and the future
build-time cue generator (setara-nx07.4) resolve through this module instead of keeping their own
copy, so the UI only ever needs to persist a stable ID (plan §5).

Plan reference: asa-hosted-tts-and-cue-migration-plan.md §5 (Stable ASA voice catalog).
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "assets" / "voice-catalog.yaml"


@dataclass(frozen=True)
class VoiceEntry:
    id: str
    label: str
    language: str
    providers: dict[str, str]  # provider name -> that provider's voice reference


class UnknownVoiceError(Exception):
    pass


def _load() -> dict[str, VoiceEntry]:
    raw = yaml.safe_load(_CATALOG_PATH.read_text())
    entries: dict[str, VoiceEntry] = {}
    for voice in raw.get("voices", []):
        providers = {
            provider_name: provider_conf["voice"]
            for provider_name, provider_conf in voice.get("providers", {}).items()
        }
        entries[voice["id"]] = VoiceEntry(
            id=voice["id"], label=voice["label"], language=voice.get("language", "en"),
            providers=providers,
        )
    return entries


# Loaded once at import time - the catalog is a static asset shipped with the image, not runtime
# configuration, so there is nothing to hot-reload.
_CATALOG: dict[str, VoiceEntry] = _load()


def voice_ids() -> list[str]:
    return list(_CATALOG.keys())


def entries() -> list[VoiceEntry]:
    return list(_CATALOG.values())


def entries_for_provider(provider: str) -> list[VoiceEntry]:
    return [entry for entry in _CATALOG.values() if provider in entry.providers]


def resolve_voice_ref(voice_id: str | None, provider: str, default_voice_id: str) -> str:
    """Resolve a stable ASA voice ID to the given provider's own voice reference. Falls back to
    `default_voice_id` when `voice_id` is missing/unknown, and to that provider's own raw voice
    string when a resolved entry doesn't define a mapping for this provider (lets adapters keep a
    provider-native default without every catalog entry needing every provider listed)."""
    entry = _CATALOG.get(voice_id or "") or _CATALOG.get(default_voice_id)
    if entry is None:
        raise UnknownVoiceError(f"No voice catalog entry for {default_voice_id!r}")
    ref = entry.providers.get(provider)
    if ref is None:
        raise UnknownVoiceError(f"Voice {entry.id!r} has no {provider!r} mapping")
    return ref


def list_voices_for_provider(provider: str, model: str) -> list[dict]:
    return [
        {"id": entry.id, "label": entry.label, "model": model, "language": entry.language}
        for entry in entries_for_provider(provider)
    ]
