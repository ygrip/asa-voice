from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pcm_append_uses_deques_without_whole_buffer_copy() -> None:
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    add_pcm = source.split("def add_pcm", 1)[1].split("def should_decode", 1)[0]

    assert "deque[np.ndarray]" in source
    assert "self._chunks.append(samples)" in add_pcm
    assert "self._utterance_chunks.append(samples)" in add_pcm
    assert "np.concatenate" not in add_pcm


def test_flush_uses_full_utterance_final_profile() -> None:
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    flush = source.split("def flush", 1)[1].split("_agreed_len", 1)[0]

    assert "self._utterance_audio()" in flush
    assert "self._svc.transcribe_array_final(audio, self._context)" in flush
    assert "self.reset()" in flush
