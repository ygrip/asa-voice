from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_flush_decodes_buffer_even_before_partial_interval() -> None:
    router = (ROOT / "app" / "routers" / "stt.py").read_text()
    session = (ROOT / "app" / "providers" / "streaming" / "faster_whisper_session.py").read_text()

    assert "def has_buffered_audio(self) -> bool:" in session
    assert "return self._total_samples > 0" in session
    assert "text = await run_in_threadpool(rolling.final_text)" in session
    assert "final = await scheduler.flush()" in router
    assert "await self._settle_decoder()" in (
        ROOT / "app" / "services" / "stt_stream_scheduler.py"
    ).read_text()
