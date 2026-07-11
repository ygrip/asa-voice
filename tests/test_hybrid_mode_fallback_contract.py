"""Integration test: hybrid mode with a forced OpenAI failure falls back to faster-whisper
(setara-s94o.8 acceptance criterion)."""
import asyncio

from app.providers.base import SttOptions, SttResult
from app.providers.errors import SttFallbackEligibleError
from app.providers.policy import RequestValidationPolicy
from app.providers.router import SttProviderRouter


class _FailingOpenAi:
    provider_name = "openai"

    async def transcribe(self, audio_path, options):
        raise SttFallbackEligibleError("OpenAI STT timed out")


class _WorkingFasterWhisper:
    provider_name = "faster_whisper"

    async def transcribe(self, audio_path, options):
        return SttResult(
            provider="faster_whisper", model="base.en", text="create a release plan",
            language="en", duration_ms=1200, latency_ms=300, segments=[],
        )


def test_hybrid_mode_falls_back_to_faster_whisper_on_openai_failure(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    router = SttProviderRouter(
        primary=_FailingOpenAi(),
        fallback=_WorkingFasterWhisper(),
        policy=RequestValidationPolicy(),
    )

    result = asyncio.run(router.transcribe(str(audio_path), SttOptions(client_id="test-client")))

    assert result.provider == "faster_whisper"
    assert result.fallback_used is True
    assert result.text == "create a release plan"
