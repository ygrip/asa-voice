from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stream_send_is_guarded_after_client_close() -> None:
    source = (ROOT / "app" / "routers" / "stt.py").read_text()

    assert "closed = False" in source
    assert "async def send_json(message:" in source
    assert "except (WebSocketDisconnect, RuntimeError):" in source
    assert "emit_partial=lambda partial: send_json(partial)" in source
    assert "return await send_json(final)" in source
    assert "reader_task = asyncio.create_task(reader_loop()" in source
    reader = source.split("async def reader_loop()", 1)[1].split("reader_task =", 1)[0]
    assert "decode_partial" not in reader
    assert "stt_semaphore" not in reader
    assert "if closed:\n            return" in source
