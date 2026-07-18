import asyncio
import os
import subprocess
import sys

import pytest
from fastapi import FastAPI, Response, status

from app import main, runtime
from app.config import settings
from app.providers.router import SttProviderRouter, TtsProviderRouter
from app.routers import health


class _SttAdapter:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name


class _TtsAdapter:
    def __init__(self, provider_name: str = "pocket_tts") -> None:
        self.provider_name = provider_name

    def list_voices(self) -> list[dict]:
        return []


class _TtsService:
    def list_voices(self) -> list[dict]:
        return []


@pytest.fixture(autouse=True)
def reset_runtime_components():
    runtime.reset_components()
    yield
    runtime.reset_components()


@pytest.mark.parametrize(
    ("primary", "fallback", "expected"),
    [
        ("faster_whisper", "none", True),
        ("openai", "faster_whisper", True),
        ("openai", "none", False),
    ],
)
def test_local_stt_requirement_follows_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    fallback: str,
    expected: bool,
) -> None:
    _set_provider_settings(monkeypatch, "hybrid", primary, fallback)
    assert runtime.needs_local_stt() is expected


def test_local_health_requires_the_local_stt_component(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(monkeypatch, "local", "faster_whisper", "none")
    unavailable_response = Response()
    unavailable = health.health(unavailable_response)
    assert unavailable_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert unavailable.stt.ready is False
    assert unavailable.stt.localReady is False

    runtime.stt_service = object()
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("faster_whisper"))
    ready_response = Response()
    ready = health.health(ready_response)
    assert ready_response.status_code == status.HTTP_200_OK
    assert ready.stt.ready is True
    assert ready.stt.localReady is True
    assert ready.stt.hostedReady is False
    local_models = health.models()
    assert local_models.stt.activeProvider == "faster_whisper"
    assert local_models.stt.activeModel == settings.stt_model
    assert local_models.stt.activeLoaded is True
    assert local_models.stt.localLoaded is True
    assert local_models.stt.availableProviders == ["faster_whisper"]


def test_hosted_health_is_ready_without_local_stt_or_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))

    response = Response()
    payload = health.health(response)

    assert response.status_code == status.HTTP_200_OK
    assert payload.status == "degraded"
    assert payload.sttLoaded is True
    assert payload.ttsLoaded is False
    assert payload.stt.ready is True
    assert payload.stt.localReady is False
    assert payload.stt.hostedReady is True
    assert payload.tts.ready is False


def test_hosted_health_rejects_incomplete_openai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none")
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))

    response = Response()
    payload = health.health(response)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert payload.stt.ready is False
    assert payload.stt.hostedReady is False
    assert payload.stt.warning == "OpenAI STT configuration is incomplete"


def test_hybrid_health_reports_ready_primary_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(
        monkeypatch,
        "hybrid",
        "openai",
        "faster_whisper",
        api_key="sk-hybrid",
    )
    runtime.stt_service = object()
    runtime.stt_router = SttProviderRouter(
        primary=_SttAdapter("openai"),
        fallback=_SttAdapter("faster_whisper"),
    )
    _enable_tts()

    response = Response()
    payload = health.health(response)

    assert response.status_code == status.HTTP_200_OK
    assert payload.status == "ok"
    assert payload.stt.ready is True
    assert payload.stt.fallbackReady is True
    assert payload.stt.warning is None
    hybrid_models = health.models()
    assert hybrid_models.stt.activeProvider == "openai"
    assert hybrid_models.stt.fallbackProvider == "faster_whisper"
    assert hybrid_models.stt.fallbackModel == settings.stt_model
    assert hybrid_models.stt.fallbackLoaded is True
    assert hybrid_models.stt.localLoaded is True
    assert hybrid_models.stt.availableProviders == ["faster_whisper", "openai"]


def test_degraded_hybrid_stays_ready_with_fallback_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_provider_settings(
        monkeypatch,
        "hybrid",
        "openai",
        "faster_whisper",
        api_key="sk-hybrid",
    )
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))
    _enable_tts()

    response = Response()
    payload = health.health(response)

    assert response.status_code == status.HTTP_200_OK
    assert payload.status == "degraded"
    assert payload.stt.ready is True
    assert payload.stt.fallbackReady is False
    assert payload.stt.warning == "Configured STT fallback faster_whisper is unavailable"
    degraded_models = health.models()
    assert degraded_models.stt.activeLoaded is True
    assert degraded_models.stt.fallbackLoaded is False
    assert degraded_models.stt.localLoaded is False
    assert degraded_models.stt.availableProviders == ["openai"]


def test_models_report_actual_hosted_provider_model_and_loaded_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    monkeypatch.setattr(settings, "openai_stt_model", "gpt-4o-transcribe")
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))

    payload = health.models()

    assert payload.stt.engine == "openai"
    assert payload.stt.model == "gpt-4o-transcribe"
    assert payload.stt.activeProvider == "openai"
    assert payload.stt.activeModel == "gpt-4o-transcribe"
    assert payload.stt.activeLoaded is True
    assert payload.stt.localLoaded is False
    assert payload.stt.hostedConfigured is True
    assert payload.stt.device is None
    assert payload.stt.computeType is None
    assert payload.stt.availableProviders == ["openai"]
    assert payload.tts.loaded is False
    assert payload.tts.availableProviders == []


def test_hosted_lifespan_never_constructs_local_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    local_constructions = 0

    def fail_if_local_stt_is_constructed():
        nonlocal local_constructions
        local_constructions += 1
        raise AssertionError("hosted-only startup must not construct faster-whisper")

    def build_fake_routers() -> None:
        runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))
        runtime.tts_router = TtsProviderRouter(primary=_TtsAdapter())

    monkeypatch.setattr(runtime, "load_local_stt_service", fail_if_local_stt_is_constructed)
    monkeypatch.setattr(runtime, "load_local_tts_service", _TtsService)
    monkeypatch.setattr(runtime, "build_routers", build_fake_routers)

    async def exercise_lifespan() -> None:
        async with main.lifespan(FastAPI()):
            assert local_constructions == 0
            assert runtime.stt_service is None
            assert health.component_readiness().stt_primary_ready is True

    asyncio.run(exercise_lifespan())
    assert local_constructions == 0
    assert runtime.stt_service is None


@pytest.mark.parametrize(
    ("primary", "fallback", "expected"),
    [
        ("pocket_tts", "none", True),
        ("openai", "pocket_tts", True),
        ("openai", "none", False),
    ],
)
def test_local_tts_requirement_follows_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    fallback: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(settings, "tts_provider", primary)
    monkeypatch.setattr(settings, "tts_fallback_provider", fallback)
    assert runtime.needs_local_tts() is expected


def test_hosted_tts_health_is_ready_without_pocket_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    monkeypatch.setattr(settings, "tts_provider", "openai")
    monkeypatch.setattr(settings, "tts_fallback_provider", "none")
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))
    runtime.tts_router = TtsProviderRouter(primary=_TtsAdapter("openai"))

    response = Response()
    payload = health.health(response)

    assert response.status_code == status.HTTP_200_OK
    assert payload.ttsLoaded is True
    assert payload.tts.ready is True
    assert payload.tts.localReady is False
    assert payload.tts.hostedReady is True
    assert runtime.tts_service is None

    hosted_models = health.models()
    assert hosted_models.tts.activeProvider == "openai"
    assert hosted_models.tts.hostedConfigured is True
    assert hosted_models.tts.localLoaded is False
    assert hosted_models.tts.availableProviders == ["openai"]


def test_hosted_tts_health_rejects_incomplete_openai_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    monkeypatch.setattr(settings, "tts_provider", "openai")
    monkeypatch.setattr(settings, "tts_fallback_provider", "none")
    monkeypatch.setattr(settings, "openai_tts_model", "")
    runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))
    runtime.tts_router = TtsProviderRouter(primary=_TtsAdapter("openai"))

    response = Response()
    payload = health.health(response)

    assert payload.tts.ready is False
    assert payload.tts.hostedReady is False
    assert payload.tts.warning == "OpenAI TTS configuration is incomplete"


def test_hosted_tts_lifespan_never_constructs_pocket_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider_settings(monkeypatch, "hosted", "openai", "none", api_key="sk-hosted")
    monkeypatch.setattr(settings, "tts_provider", "openai")
    monkeypatch.setattr(settings, "tts_fallback_provider", "none")
    local_constructions = 0

    def fail_if_pocket_tts_is_constructed():
        nonlocal local_constructions
        local_constructions += 1
        raise AssertionError("hosted-only startup must not construct Pocket TTS")

    def build_fake_routers() -> None:
        runtime.stt_router = SttProviderRouter(primary=_SttAdapter("openai"))
        runtime.tts_router = TtsProviderRouter(primary=_TtsAdapter("openai"))

    monkeypatch.setattr(runtime, "load_local_stt_service", lambda: object())
    monkeypatch.setattr(runtime, "load_local_tts_service", fail_if_pocket_tts_is_constructed)
    monkeypatch.setattr(runtime, "build_routers", build_fake_routers)

    async def exercise_lifespan() -> None:
        async with main.lifespan(FastAPI()):
            assert local_constructions == 0
            assert runtime.tts_service is None
            assert health.component_readiness().tts_ready is True

    asyncio.run(exercise_lifespan())
    assert local_constructions == 0
    assert runtime.tts_service is None


def test_hosted_main_import_does_not_load_faster_whisper_modules() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "ASA_VOICE_MODE": "hosted",
            "STT_PROVIDER": "openai",
            "STT_FALLBACK_PROVIDER": "none",
            "OPENAI_API_KEY": "sk-hosted",
        }
    )
    script = """
import sys
import app.main
assert "app.services.stt_service" not in sys.modules
assert "app.providers.faster_whisper" not in sys.modules
assert "faster_whisper" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _set_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    primary: str,
    fallback: str,
    *,
    api_key: str = "",
) -> None:
    monkeypatch.setattr(settings, "asa_voice_mode", mode)
    monkeypatch.setattr(settings, "stt_provider", primary)
    monkeypatch.setattr(settings, "stt_fallback_provider", fallback)
    monkeypatch.setattr(settings, "openai_api_key", api_key)


def _enable_tts() -> None:
    runtime.tts_service = _TtsService()
    runtime.tts_router = TtsProviderRouter(primary=_TtsAdapter())
