import pytest

from app.config import settings
from app.services.stt_service import profile_for_mode


def test_command_profile_keeps_one_fallback_step_and_stays_independent() -> None:
    # Not a single temperature: live testing showed temp=0.0 alone failing the quality gate on real
    # audio, with no fallback, faster-whisper returns empty text ("Could not understand audio").
    profile = profile_for_mode("command")
    assert profile.temperatures == (0.0, 0.2)
    assert profile.condition_on_previous is False


def test_dictation_profile_keeps_a_fallback_ladder_and_context() -> None:
    profile = profile_for_mode("dictation")
    assert profile.temperatures == (0.0, 0.2)
    assert profile.condition_on_previous is True


def test_hands_free_profile_mirrors_command() -> None:
    command = profile_for_mode("command")
    hands_free = profile_for_mode("hands_free")
    assert hands_free.temperatures == command.temperatures
    assert hands_free.condition_on_previous == command.condition_on_previous


def test_unrecognized_mode_falls_back_to_the_command_profile() -> None:
    assert profile_for_mode("some_future_mode") == profile_for_mode("command")


def test_profiles_read_from_their_own_mode_specific_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guards against a copy-paste field-mapping bug (e.g. dictation silently reading command's beam
    # size) - each mode's profile must move independently when its own setting changes.
    monkeypatch.setattr(settings, "stt_dictation_beam_size", 7)
    assert profile_for_mode("dictation").beam_size == 7
    assert profile_for_mode("command").beam_size != 7
    assert profile_for_mode("hands_free").beam_size != 7
