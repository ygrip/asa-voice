"""Provider-agnostic STT/TTS adapter interfaces and shared dataclasses.

Every current and future provider (faster-whisper, OpenAI, whisper.cpp, voirs, ...) conforms to
these shapes so routers/policy/observability never need to know which provider is active.

Plan reference: asa-local-openai-hosted-mode-plan.md §5 (Provider Abstraction).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

# Sentinel audio_path passed to SttPolicy.validate_audio() by the file-free /stt/raw + streaming
# array-based path, where there is no real file on disk to inspect.
IN_MEMORY_AUDIO_MARKER = "<in-memory>"


@dataclass
class SttOptions:
    language: Optional[str] = None
    prompt: Optional[str] = None
    hotwords: Optional[List[str]] = None
    request_id: Optional[str] = None
    client_id: Optional[str] = None


@dataclass
class SttSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class SttResult:
    provider: str
    model: str
    text: str
    language: Optional[str]
    duration_ms: int
    latency_ms: int
    segments: List[SttSegment] = field(default_factory=list)
    fallback_used: bool = False


class SttAdapter(Protocol):
    async def transcribe(self, audio_path: str, options: SttOptions) -> SttResult: ...


@dataclass
class TtsOptions:
    voice_id: Optional[str] = None
    format: str = "wav"
    request_id: Optional[str] = None
    client_id: Optional[str] = None


@dataclass
class TtsResult:
    provider: str
    model: str
    audio_path: str
    content_type: str
    latency_ms: int


class TtsAdapter(Protocol):
    async def synthesize(self, text: str, options: TtsOptions) -> TtsResult: ...
