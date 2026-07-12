import json

import pytest

from app.services.stt_stream_protocol import (
    STT_MAX_CONTROL_BYTES,
    SttCancelControl,
    SttCorrelationMetadata,
    SttErrorEvent,
    SttFinalEvent,
    SttFlushControl,
    SttPartialEvent,
    SttProtocolError,
    SttReadyEvent,
    SttResetControl,
    SttStartControl,
    is_authoritative_stt_finality,
    parse_stt_client_control,
    parse_stt_server_event,
)


START = {
    "type": "start",
    "protocolVersion": "2",
    "sessionId": "uuid",
    "requestId": "request-42",
    "mode": "command",
    "provider": "auto",
    "audio": {
        "sampleRate": 16000,
        "channels": 1,
        "sampleFormat": "s16le",
        "frameDurationMs": 20,
    },
    "language": "en",
    "prompt": "Voice command for ASA inside Setara.",
    "hotwords": ["ASA", "Setara", "Raksara"],
    "maxDurationSeconds": 30,
}


def test_plan_client_control_examples_parse() -> None:
    start = parse_stt_client_control(json.dumps(START))
    assert isinstance(start, SttStartControl)
    assert start.protocol_version == "2"
    assert start.audio.sample_rate == 16000
    assert start.audio.frame_duration_ms == 20
    assert isinstance(
        parse_stt_client_control({"type": "flush", "reason": "user_stop"}),
        SttFlushControl,
    )
    assert isinstance(parse_stt_client_control({"type": "reset"}), SttResetControl)
    assert isinstance(parse_stt_client_control({"type": "cancel"}), SttCancelControl)


def test_plan_server_event_examples_parse() -> None:
    assert isinstance(
        parse_stt_server_event(
            {
                "type": "ready",
                "protocolVersion": "2",
                "provider": "faster_whisper",
                "model": "distil-small.en",
                "supportsPartials": True,
                "maxDurationSeconds": 300,
            }
        ),
        SttReadyEvent,
    )
    assert isinstance(
        parse_stt_server_event(
            {
                "type": "partial",
                "sequence": 18,
                "text": "create a release plan",
                "committedText": "create a release",
                "unstableText": "plan",
                "audioReceivedMs": 4600,
            }
        ),
        SttPartialEvent,
    )
    assert isinstance(
        parse_stt_server_event(
            {
                "type": "final",
                "text": "create a release plan for Raksara",
                "finality": "provider_final",
                "provider": "faster_whisper",
                "model": "distil-small.en",
                "durationMs": 6120,
                "latencyMs": 840,
                "fallbackUsed": False,
                "audioDroppedMs": 0,
            }
        ),
        SttFinalEvent,
    )
    assert isinstance(
        parse_stt_server_event(
            {
                "type": "error",
                "code": "STT_BACKPRESSURE",
                "message": "Audio could not be processed in real time.",
                "retryable": True,
            }
        ),
        SttErrorEvent,
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocolVersion",), "1"),
        (("mode",), "meeting"),
        (("audio", "sampleRate"), 48000),
        (("audio", "channels"), 2),
        (("audio", "sampleFormat"), "f32le"),
        (("audio", "frameDurationMs"), 10),
        (("maxDurationSeconds",), 301),
    ],
)
def test_invalid_start_contract_fails_deterministically(path: tuple[str, ...], value: object) -> None:
    payload = json.loads(json.dumps(START))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(SttProtocolError) as raised:
        parse_stt_client_control(payload)
    assert raised.value.code == "STT_INVALID_MESSAGE"


def test_provider_policy_and_control_size_are_enforced() -> None:
    explicit = {**START, "provider": "openai"}
    with pytest.raises(SttProtocolError) as raised:
        parse_stt_client_control(explicit)
    assert raised.value.code == "STT_PROVIDER_NOT_ALLOWED"
    allowed = parse_stt_client_control(explicit, allow_provider_override=True)
    assert isinstance(allowed, SttStartControl)
    assert allowed.provider == "openai"

    oversized = {**START, "prompt": "x" * STT_MAX_CONTROL_BYTES}
    with pytest.raises(SttProtocolError) as oversized_error:
        parse_stt_client_control(oversized)
    assert oversized_error.value.code == "STT_CONTROL_TOO_LARGE"


def test_invalid_finality_fails_and_only_authoritative_values_are_accepted_for_submission() -> None:
    with pytest.raises(SttProtocolError) as raised:
        parse_stt_server_event(
            {
                "type": "final",
                "text": "unsafe",
                "finality": "guessed_final",
                "provider": "faster_whisper",
                "model": "distil-small.en",
                "durationMs": 1,
                "latencyMs": 1,
                "fallbackUsed": False,
                "audioDroppedMs": 0,
            }
        )
    assert raised.value.code == "STT_INVALID_MESSAGE"
    assert is_authoritative_stt_finality("provider_final")
    assert is_authoritative_stt_finality("local_recovered_final")
    assert not is_authoritative_stt_finality("connection_lost_partial")


def test_correlation_metadata_is_id_only() -> None:
    start = parse_stt_client_control(START)
    assert isinstance(start, SttStartControl)
    metadata = SttCorrelationMetadata.from_start(start, "core-relay").as_log_fields()
    assert metadata == {
        "requestId": "request-42",
        "sessionId": "uuid",
        "clientId": "core-relay",
    }
    assert "text" not in metadata
    assert "audio" not in metadata


def test_bounded_identity_fields_are_normalized_and_whitespace_only_is_rejected() -> None:
    start = parse_stt_client_control({**START, "provider": " auto "})
    assert isinstance(start, SttStartControl)
    assert start.provider == "auto"

    ready = parse_stt_server_event(
        {
            "type": "ready",
            "protocolVersion": "2",
            "provider": " faster_whisper ",
            "model": " distil-small.en ",
            "supportsPartials": True,
            "maxDurationSeconds": 300,
        }
    )
    assert isinstance(ready, SttReadyEvent)
    assert ready.provider == "faster_whisper"
    assert ready.model == "distil-small.en"

    error = parse_stt_server_event(
        {
            "type": "error",
            "code": " STT_BACKPRESSURE ",
            "message": " Audio could not be processed. ",
            "retryable": True,
        }
    )
    assert isinstance(error, SttErrorEvent)
    assert error.code == "STT_BACKPRESSURE"
    assert error.message == "Audio could not be processed."

    for field in ("provider", "model"):
        payload = {
            "type": "ready",
            "protocolVersion": "2",
            "provider": "faster_whisper",
            "model": "distil-small.en",
            "supportsPartials": True,
            "maxDurationSeconds": 300,
            field: "   ",
        }
        with pytest.raises(SttProtocolError) as raised:
            parse_stt_server_event(payload)
        assert raised.value.code == "STT_INVALID_MESSAGE"

    for field in ("code", "message"):
        payload = {
            "type": "error",
            "code": "STT_FAILURE",
            "message": "stream failed",
            "retryable": True,
            field: "   ",
        }
        with pytest.raises(SttProtocolError) as raised:
            parse_stt_server_event(payload)
        assert raised.value.code == "STT_INVALID_MESSAGE"
