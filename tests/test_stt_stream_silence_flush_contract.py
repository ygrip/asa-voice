from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_silence_auto_flush_is_hands_free_only_and_timer_starts_at_ready() -> None:
    """Regression for a garbage-transcript incident: command/dictation sessions were being
    auto-flushed by the hands-free-only silence fallback, truncating real utterances to a
    fragment (e.g. "I-K"), and the timer's clock started before the handshake even completed."""
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert "session_mode: str | None = None" in source
    assert "session_mode = options.mode" in source

    # Timer resets only after `ready` is actually sent, not at ws.accept().
    configure_body = source.split("async def configure(start: SttStartControl)", 1)[1].split(
        "async def handle_decode_error", 1
    )[0]
    assert "ready_sent = await send_json(ready)" in configure_body
    assert "last_speech_time[0] = time.monotonic()" in configure_body
    assert "last_audio_time[0] = time.monotonic()" in configure_body
    assert "return ready_sent" in configure_body

    # The auto-flush condition itself must gate on hands_free.
    reader_body = source.split("async def reader_loop()", 1)[1]
    flush_condition = reader_body.split("if (", 1)[1].split("):", 1)[0]
    assert 'session_mode == "hands_free"' in flush_condition
    assert "stt_stream_silence_flush_s > 0" in flush_condition
