import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator


STT_PROTOCOL_VERSION = "2"
STT_MAX_CONTROL_BYTES = 8_192

SttMode = Literal["command", "hands_free", "dictation"]
SttProviderPreference = Literal["auto", "faster_whisper", "openai"]
SttFlushReason = Literal[
    "user_stop",
    "vad_silence",
    "max_duration",
    "navigation",
    "barge_in",
    "client_shutdown",
]
SttFinality = Literal[
    "provider_final",
    "local_recovered_final",
    "partial_timeout",
    "connection_lost_partial",
    "cancelled",
]


class SttProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", strict=True)

    @field_validator("provider", "model", "code", "message", mode="before", check_fields=False)
    @classmethod
    def normalize_bounded_identity_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class SttAudioFormat(_WireModel):
    sample_rate: Literal[16000] = Field(alias="sampleRate")
    channels: Literal[1]
    sample_format: Literal["s16le"] = Field(alias="sampleFormat")
    frame_duration_ms: Literal[20] = Field(alias="frameDurationMs")


class SttStartControl(_WireModel):
    type: Literal["start"]
    protocol_version: Literal["2"] = Field(alias="protocolVersion")
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    request_id: str | None = Field(default=None, alias="requestId", min_length=1, max_length=128)
    mode: SttMode
    provider: SttProviderPreference
    audio: SttAudioFormat
    language: str | None = Field(default=None, min_length=1, max_length=16)
    prompt: str | None = Field(default=None, max_length=1_000)
    hotwords: list[str] = Field(default_factory=list, max_length=100)
    max_duration_seconds: int = Field(alias="maxDurationSeconds", ge=1, le=300)

    @field_validator("session_id", "request_id", "language")
    @classmethod
    def strip_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("hotwords")
    @classmethod
    def validate_hotwords(cls, words: list[str]) -> list[str]:
        cleaned = [word.strip() for word in words]
        if any(not word or len(word) > 100 for word in cleaned):
            raise ValueError("hotwords must contain non-blank values of at most 100 characters")
        return cleaned


class SttFlushControl(_WireModel):
    type: Literal["flush"]
    reason: SttFlushReason


class SttResetControl(_WireModel):
    type: Literal["reset"]


class SttCancelControl(_WireModel):
    type: Literal["cancel"]


SttClientControl = Annotated[
    SttStartControl | SttFlushControl | SttResetControl | SttCancelControl,
    Field(discriminator="type"),
]


class SttReadyEvent(_WireModel):
    type: Literal["ready"]
    protocol_version: Literal["2"] = Field(alias="protocolVersion")
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    supports_partials: bool = Field(alias="supportsPartials")
    max_duration_seconds: int = Field(alias="maxDurationSeconds", ge=1, le=300)


class SttPartialEvent(_WireModel):
    type: Literal["partial"]
    sequence: int = Field(ge=0)
    text: str
    committed_text: str = Field(alias="committedText")
    unstable_text: str = Field(alias="unstableText")
    audio_received_ms: int = Field(alias="audioReceivedMs", ge=0)


class SttFinalEvent(_WireModel):
    type: Literal["final"]
    text: str
    finality: SttFinality
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    duration_ms: int = Field(alias="durationMs", ge=0)
    latency_ms: int = Field(alias="latencyMs", ge=0)
    fallback_used: bool = Field(alias="fallbackUsed")
    audio_dropped_ms: int = Field(alias="audioDroppedMs", ge=0)


class SttErrorEvent(_WireModel):
    type: Literal["error"]
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(max_length=1_000)
    retryable: bool


SttServerEvent = Annotated[
    SttReadyEvent | SttPartialEvent | SttFinalEvent | SttErrorEvent,
    Field(discriminator="type"),
]


class SttCorrelationMetadata(_WireModel):
    """Safe metric/log context. Transcript and audio content are intentionally unrepresentable."""

    request_id: str | None = Field(default=None, alias="requestId", max_length=128)
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    client_id: str | None = Field(default=None, alias="clientId", max_length=128)

    @classmethod
    def from_start(cls, start: SttStartControl, client_id: str | None) -> "SttCorrelationMetadata":
        return cls(requestId=start.request_id, sessionId=start.session_id, clientId=client_id)

    def as_log_fields(self) -> dict[str, str]:
        return self.model_dump(by_alias=True, exclude_none=True)


_CLIENT_CONTROL_ADAPTER = TypeAdapter(SttClientControl)
_SERVER_EVENT_ADAPTER = TypeAdapter(SttServerEvent)


def parse_stt_client_control(
    value: str | bytes | dict,
    *,
    allow_provider_override: bool = False,
) -> SttClientControl:
    payload = _decode_control(value)
    try:
        control = _CLIENT_CONTROL_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SttProtocolError("STT_INVALID_MESSAGE", _first_error(exc)) from exc
    if (
        isinstance(control, SttStartControl)
        and control.provider != "auto"
        and not allow_provider_override
    ):
        raise SttProtocolError(
            "STT_PROVIDER_NOT_ALLOWED",
            "Explicit STT provider selection is not allowed for this client",
        )
    return control


def parse_stt_server_event(value: str | bytes | dict) -> SttServerEvent:
    payload = _decode_control(value)
    try:
        return _SERVER_EVENT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SttProtocolError("STT_INVALID_MESSAGE", _first_error(exc)) from exc


def is_authoritative_stt_finality(finality: SttFinality) -> bool:
    return finality in {"provider_final", "local_recovered_final"}


def _decode_control(value: str | bytes | dict) -> dict:
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        try:
            encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SttProtocolError("STT_INVALID_JSON", "STT control must be valid JSON") from exc
    if len(encoded) > STT_MAX_CONTROL_BYTES:
        raise SttProtocolError("STT_CONTROL_TOO_LARGE", "STT control exceeds 8192 bytes")
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SttProtocolError("STT_INVALID_JSON", "STT control must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise SttProtocolError("STT_INVALID_JSON", "STT control must be a JSON object")
    return payload


def _first_error(error: ValidationError) -> str:
    errors = error.errors()
    if not errors:
        return "Invalid STT message"
    return str(errors[0].get("msg", "Invalid STT message"))
