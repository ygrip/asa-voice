from pydantic import BaseModel, Field

from app.providers.base import SttResult


class SttSegment(BaseModel):
    start: float
    end: float
    text: str


class SttResponse(BaseModel):
    """Provider-agnostic STT result (plan §4.3). Field set is fixed across every STT provider —
    only `provider`/`model`/`fallbackUsed` differ once Phase 2 adds OpenAI."""
    requestId: str
    provider: str
    model: str
    language: str | None = None
    durationMs: int
    latencyMs: int
    text: str
    segments: list[SttSegment]
    fallbackUsed: bool = False


def stt_response(result: SttResult, request_id: str) -> SttResponse:
    """Build the wire response from a provider-agnostic SttResult + the request's echoed id."""
    return SttResponse(
        requestId=request_id,
        provider=result.provider,
        model=result.model,
        language=result.language,
        durationMs=result.duration_ms,
        latencyMs=result.latency_ms,
        text=result.text,
        segments=[SttSegment(start=s.start, end=s.end, text=s.text) for s in result.segments],
        fallbackUsed=result.fallback_used,
    )


class TtsRequest(BaseModel):
    text: str
    voiceId: str | None = None
    format: str = "wav"


class VoiceInfo(BaseModel):
    id: str
    label: str
    model: str
    language: str


class HealthSttInfo(BaseModel):
    model: str
    device: str | None = None
    computeType: str | None = None
    artifactReady: bool | None = None
    provider: str
    fallbackProvider: str | None = None
    fallbackModel: str | None = None
    localReady: bool
    hostedReady: bool
    fallbackReady: bool | None = None
    ready: bool
    warning: str | None = None


class HealthTtsInfo(BaseModel):
    engine: str
    model: str
    sampleRate: int
    provider: str
    ready: bool
    warning: str | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    sttLoaded: bool
    ttsLoaded: bool
    stt: HealthSttInfo
    tts: HealthTtsInfo


class ModelLimits(BaseModel):
    cpu: int
    memoryMb: int
    maxAudioSeconds: int
    maxUploadMb: int


class SttInfo(BaseModel):
    engine: str
    model: str
    device: str | None = None
    computeType: str | None = None
    activeProvider: str
    activeModel: str
    activeLoaded: bool
    fallbackProvider: str | None = None
    fallbackModel: str | None = None
    fallbackLoaded: bool | None = None
    localLoaded: bool
    hostedConfigured: bool
    availableProviders: list[str]
    supportedProviders: list[str]


class TtsInfo(BaseModel):
    engine: str
    activeModel: str
    loaded: bool
    defaultVoice: str
    voices: list[VoiceInfo]
    activeProvider: str
    availableProviders: list[str]
    supportedProviders: list[str]


class ModelsResponse(BaseModel):
    mode: str
    limits: ModelLimits
    stt: SttInfo
    tts: TtsInfo


class ErrorResponse(BaseModel):
    error: str
    message: str = Field(default="")
