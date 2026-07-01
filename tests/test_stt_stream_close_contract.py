from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stream_send_is_guarded_after_client_close() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert "closed = False" in source
    assert "async def send_json(message: dict) -> bool:" in source
    assert "except (WebSocketDisconnect, RuntimeError):" in source
    assert 'await send_json({"type": "partial", "text": text})' in source
    assert 'await send_json({"type": "final", "text": final["final"]})' in source
    assert "if closed:\n                return" in source
