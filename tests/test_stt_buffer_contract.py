from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pcm_append_uses_deques_without_whole_buffer_copy() -> None:
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    add_pcm = source.split("def add_pcm", 1)[1].split("def should_decode", 1)[0]

    assert "deque[np.ndarray]" in source
    assert "self._chunks.append(samples)" in add_pcm
    assert "self._utterance.append(samples)" in add_pcm
    assert "np.concatenate" not in add_pcm


def test_flush_decodes_full_utterance_when_it_fits() -> None:
    # Common case: the utterance fits the window -> decode it whole with the accurate profile
    # (full Whisper context, VAD-trimmed) so the final matches the batch /stt endpoint.
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    flush = source.split("def flush", 1)[1].split("_agreed_len", 1)[0]

    assert "if not self._utterance_capped:" in flush
    assert "self._utterance_audio()" in flush
    assert "self._svc.transcribe_array_final(audio, self._context)" in flush
    assert "self.reset()" in flush


def test_flush_falls_back_to_committed_plus_tail_when_capped() -> None:
    # Over-long case: the utterance outgrew the window, so a full re-decode would be truncated.
    # Fall back to the LocalAgreement-committed words + an accurate decode of the un-committed tail
    # -> complete transcript for arbitrarily long dictation (no garbled/truncated final).
    source = (ROOT / "app" / "services" / "stt_service.py").read_text()
    flush = source.split("def flush", 1)[1].split("_agreed_len", 1)[0]

    assert "self.committed_text" in flush
    assert "self._rolling_audio()" in flush
