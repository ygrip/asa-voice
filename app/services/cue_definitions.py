"""Cue manifest (setara-nx07.3): the only source of cue IDs and their phrases. Both the build-time
cue generator (setara-nx07.4) and the runtime cue service resolve cue text from here.

Plan reference: asa-hosted-tts-and-cue-migration-plan.md §9.1 (Cue manifest).
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "assets" / "cues.yaml"


@dataclass(frozen=True)
class CueDefinition:
    id: str
    text: str
    max_duration_ms: int


def _load() -> dict[str, CueDefinition]:
    raw = yaml.safe_load(_MANIFEST_PATH.read_text())
    return {
        cue_id: CueDefinition(id=cue_id, text=cue["text"], max_duration_ms=cue["maxDurationMs"])
        for cue_id, cue in raw.get("cues", {}).items()
    }


# Loaded once at import time - a static asset shipped with the image, not runtime configuration.
CUE_DEFINITIONS: dict[str, CueDefinition] = _load()


def cue_ids() -> list[str]:
    return list(CUE_DEFINITIONS.keys())
