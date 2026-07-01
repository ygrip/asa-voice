from pydantic import BaseModel, Field


class SttSegment(BaseModel):
    start: float
    end: float
    text: str


class SttResponse(BaseModel):
    text: str
    segments: list[SttSegment]
    language: str
    durationSeconds: float
    engine: str
    model: str


class TtsRequest(BaseModel):
    text: str
    voiceId: str | None = None
    format: str = "wav"


class VoiceInfo(BaseModel):
    id: str
    label: str
    model: str
    language: str


class HealthResponse(BaseModel):
    status: str
    sttLoaded: bool
    ttsLoaded: bool


class ModelLimits(BaseModel):
    cpu: int
    memoryMb: int
    maxAudioSeconds: int
    maxUploadMb: int


class SttInfo(BaseModel):
    engine: str
    model: str
    device: str
    computeType: str


class TtsInfo(BaseModel):
    engine: str
    defaultVoice: str
    voices: list[VoiceInfo]


class ModelsResponse(BaseModel):
    limits: ModelLimits
    stt: SttInfo
    tts: TtsInfo


class ErrorResponse(BaseModel):
    error: str
    message: str = Field(default="")
