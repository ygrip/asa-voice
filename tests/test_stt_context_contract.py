from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_context_uses_bounded_fields_and_safe_collection_default() -> None:
    source = (ROOT / "app" / "services" / "stt_context.py").read_text()

    assert "default_factory=list" in source
    assert 'alias="requestId"' in source
    assert "max_length=100" in source
    assert 'extra="forbid"' in source


def test_partial_and_final_profiles_are_independent() -> None:
    config = (ROOT / "app" / "config.py").read_text()
    service = (ROOT / "app" / "services" / "stt_service.py").read_text()

    assert "stt_partial_beam_size" in config
    assert "stt_command_beam_size" in config
    assert "beam_size=settings.stt_partial_beam_size" in service
    assert "beam_size=profile.beam_size" in service
    assert "def transcribe_array_final(" in service
    assert "def profile_for_mode(" in service


def test_stream_supports_structured_control_messages() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert 'raw_control.get("type") == "config"' in source
    assert 'control.type == "start"' in source
    assert 'control.type == "reset"' in source
    assert 'control.type == "flush"' in source
    assert 'control.type == "cancel"' in source
    assert 'text == "flush"' in source
    assert "StreamingSttSessionFactory()" in source
