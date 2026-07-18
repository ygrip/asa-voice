"""Unit tests for the shared voice catalog resolver (setara-nx07.3, plan §5)."""
import pytest

from app.services import voice_catalog


def test_catalog_loads_the_three_stable_voice_ids() -> None:
    assert set(voice_catalog.voice_ids()) == {"asa_default", "asa_bright", "asa_calm"}


def test_resolve_voice_ref_maps_stable_id_to_provider_voice() -> None:
    assert voice_catalog.resolve_voice_ref("asa_bright", "pocket_tts", "asa_default") == "eve"
    assert voice_catalog.resolve_voice_ref("asa_bright", "openai", "asa_default") == "shimmer"


def test_resolve_voice_ref_falls_back_to_default_voice_when_unknown() -> None:
    assert voice_catalog.resolve_voice_ref("not_a_real_voice", "openai", "asa_calm") == "sage"


def test_resolve_voice_ref_falls_back_to_default_when_none() -> None:
    assert voice_catalog.resolve_voice_ref(None, "pocket_tts", "asa_default") == "anna"


def test_resolve_voice_ref_raises_for_a_provider_with_no_mapping() -> None:
    with pytest.raises(voice_catalog.UnknownVoiceError):
        voice_catalog.resolve_voice_ref("asa_default", "elevenlabs", "asa_default")


def test_entries_for_provider_only_returns_voices_that_provider_supports() -> None:
    openai_voices = {entry.id for entry in voice_catalog.entries_for_provider("openai")}
    assert openai_voices == {"asa_default", "asa_bright", "asa_calm"}


def test_list_voices_for_provider_shape() -> None:
    voices = voice_catalog.list_voices_for_provider("openai", "tts-1")
    assert {"id": "asa_default", "label": "ASA Default", "model": "tts-1", "language": "en"} in voices
