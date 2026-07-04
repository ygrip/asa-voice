from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_flush_decodes_buffer_even_before_partial_interval() -> None:
    router = (ROOT / "app" / "routers" / "stt.py").read_text()
    service = (ROOT / "app" / "services" / "stt_service.py").read_text()

    assert "def has_buffered_audio(self) -> bool:" in service
    assert "return self._total_samples > 0" in service
    assert "if session.has_buffered_audio():" in router
    assert "await run_in_threadpool(session.decode)" in router
    assert "if session.should_decode():\n                            await run_in_threadpool(session.decode)" not in router
