"""Boot-time provider config validation (setara-s94o.4): the sidecar must default to a mode that
boots clean with zero OPENAI_API_KEY set, and must fail fast (not silently no-op) when an
unimplemented provider is configured.
"""
import pytest

from app import runtime
from app.config import Settings


def test_default_settings_boot_clean_in_local_mode() -> None:
    settings = Settings()
    assert settings.asa_voice_mode == "local"
    assert settings.stt_provider == "faster_whisper"
    assert settings.tts_provider == "pocket_tts"
    assert settings.stt_fallback_provider == "none"
    assert settings.tts_fallback_provider == "none"


def test_validate_provider_config_passes_for_default_settings() -> None:
    runtime.validate_provider_config()


@pytest.mark.parametrize(
    "env_var,value",
    [
        ("stt_provider", "azure_speech"),
        ("stt_fallback_provider", "azure_speech"),
        ("tts_provider", "elevenlabs"),
        ("tts_fallback_provider", "elevenlabs"),
    ],
)
def test_validate_provider_config_fails_fast_on_unimplemented_provider(env_var, value) -> None:
    from app.config import settings

    original = getattr(settings, env_var)
    setattr(settings, env_var, value)
    try:
        with pytest.raises(runtime.UnsupportedProviderError):
            runtime.validate_provider_config()
    finally:
        setattr(settings, env_var, original)


def test_build_stt_adapter_returns_none_for_none_provider() -> None:
    assert runtime.build_stt_adapter("none", service=object()) is None


def test_build_stt_adapter_builds_openai_adapter_without_a_local_service() -> None:
    # openai is a hosted provider - it must not require the local faster-whisper SttService.
    adapter = runtime.build_stt_adapter("openai", service=None)
    assert adapter.provider_name == "openai"


def test_build_stt_adapter_raises_for_unimplemented_provider() -> None:
    with pytest.raises(runtime.UnsupportedProviderError):
        runtime.build_stt_adapter("azure_speech", service=object())


def test_build_tts_adapter_returns_none_for_none_provider() -> None:
    assert runtime.build_tts_adapter("none", service=object()) is None


def test_build_tts_adapter_returns_none_for_pocket_tts_without_a_service() -> None:
    assert runtime.build_tts_adapter("pocket_tts", service=None) is None


def test_build_tts_adapter_builds_openai_adapter_without_a_local_service() -> None:
    # openai is a hosted provider - it must not require the local Pocket TTS TtsService.
    adapter = runtime.build_tts_adapter("openai", service=None)
    assert adapter.provider_name == "openai"


def test_build_tts_adapter_raises_for_unimplemented_provider() -> None:
    with pytest.raises(runtime.UnsupportedProviderError):
        runtime.build_tts_adapter("elevenlabs", service=object())
