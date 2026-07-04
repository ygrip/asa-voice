import os
from collections import deque

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings
from app.services.stt_context import SttDecodeContext, build_hotwords, build_prompt, resolve_language

STREAM_SAMPLE_RATE = 16000  # faster-whisper's native rate; the WS client sends PCM16 mono @ 16 kHz


class SttService:
    """Faster-Whisper STT. Model is loaded once at startup; transcribe() is blocking — callers
    run it in a threadpool so the event loop stays responsive."""

    def __init__(self):
        self.model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            cpu_threads=settings.stt_cpu_threads,
            num_workers=settings.stt_num_workers,
        )

    def transcribe(
        self,
        path: str,
        language: str | None = None,
        vad: bool | None = None,
        context: SttDecodeContext | None = None,
    ) -> dict:
        try:
            segments, info = self.model.transcribe(
                path,
                language=resolve_language(context, language),
                vad_filter=settings.stt_final_vad_filter if vad is None else vad,
                vad_parameters={"min_silence_duration_ms": settings.stt_vad_min_silence_ms},
                beam_size=settings.stt_final_beam_size,
                best_of=settings.stt_final_best_of,
                temperature=settings.stt_temperatures,
                repetition_penalty=settings.stt_repetition_penalty,
                no_repeat_ngram_size=settings.stt_no_repeat_ngram_size,
                compression_ratio_threshold=settings.stt_compression_ratio_threshold,
                log_prob_threshold=settings.stt_log_prob_threshold,
                condition_on_previous_text=settings.stt_final_condition_on_previous,
                no_speech_threshold=settings.stt_no_speech_threshold,
                initial_prompt=build_prompt(context),
                hotwords=build_hotwords(context),
            )

            collected = []
            full_text = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    full_text.append(text)
                    collected.append({"start": segment.start, "end": segment.end, "text": text})

            return {
                "text": collapse_repeats(" ".join(full_text).strip()),
                "segments": collected,
                "language": info.language,
                "durationSeconds": info.duration,
                "engine": "faster-whisper",
                "model": settings.stt_model,
            }
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def decode_words(self, audio: np.ndarray, context: SttDecodeContext | None = None) -> list[dict]:
        """Transcribe a float32 mono @16kHz array, returning word-level timing. Used by the rolling
        streaming session — no file IO, no VAD trimming (the session manages the buffer)."""
        segments, _ = self.model.transcribe(
            audio,
            language=resolve_language(context),
            vad_filter=settings.stt_partial_vad_filter,
            beam_size=settings.stt_partial_beam_size,
            best_of=settings.stt_partial_best_of,
            temperature=settings.stt_temperatures,
            repetition_penalty=settings.stt_repetition_penalty,
            no_repeat_ngram_size=settings.stt_no_repeat_ngram_size,
            compression_ratio_threshold=settings.stt_compression_ratio_threshold,
            log_prob_threshold=settings.stt_log_prob_threshold,
            condition_on_previous_text=False,
            no_speech_threshold=settings.stt_no_speech_threshold,
            initial_prompt=build_prompt(context),
            hotwords=build_hotwords(context),
            word_timestamps=settings.stt_partial_word_timestamps,
        )
        words: list[dict] = []
        for seg in segments:
            for w in (seg.words or []):
                t = w.word.strip()
                if t:
                    words.append({"w": t, "start": w.start, "end": w.end})
        return words

    def transcribe_array_final(
        self, audio: np.ndarray, context: SttDecodeContext | None = None
    ) -> dict:
        """Decode a complete utterance with the accuracy profile used for command execution."""
        segments, info = self.model.transcribe(
            audio,
            language=resolve_language(context),
            vad_filter=settings.stt_final_vad_filter,
            vad_parameters={"min_silence_duration_ms": settings.stt_vad_min_silence_ms},
            beam_size=settings.stt_final_beam_size,
            best_of=settings.stt_final_best_of,
            temperature=settings.stt_temperatures,
            repetition_penalty=settings.stt_repetition_penalty,
            no_repeat_ngram_size=settings.stt_no_repeat_ngram_size,
            compression_ratio_threshold=settings.stt_compression_ratio_threshold,
            log_prob_threshold=settings.stt_log_prob_threshold,
            condition_on_previous_text=settings.stt_final_condition_on_previous,
            no_speech_threshold=settings.stt_no_speech_threshold,
            initial_prompt=build_prompt(context),
            hotwords=build_hotwords(context),
            word_timestamps=settings.stt_final_word_timestamps,
        )
        collected = []
        full_text = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                full_text.append(text)
                collected.append({"start": segment.start, "end": segment.end, "text": text})
        return {
            "text": collapse_repeats(" ".join(full_text).strip()),
            "segments": collected,
            "language": info.language,
            "durationSeconds": info.duration,
            "engine": "faster-whisper",
            "model": settings.stt_model,
        }


class StreamingSttSession:
    """Rolling-window streaming transcription (the whisper_streaming technique). The client pushes
    PCM16 frames continuously; every stt_stream_interval_ms we re-decode the buffered audio and use
    LocalAgreement-2 — words that agree across the two most recent decodes get COMMITTED (final),
    the unstable tail stays tentative (partial). On flush (VAD silence / stop) the tail commits too.

    NOT thread-safe; one session per connection, decodes serialized by the caller's STT slot."""

    def __init__(self, service: "SttService"):
        self._svc = service
        self._context: SttDecodeContext | None = None
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._utterance_chunks: deque[np.ndarray] = deque()
        self._utterance_samples = 0
        self._buf_offset_s = 0.0  # audio (seconds) already trimmed off the front of _buf
        self._committed: list[str] = []  # committed word texts (the final transcript so far)
        self._prev: list[dict] = []      # previous decode's words (absolute times)
        self._samples_since_decode = 0

    @property
    def committed_text(self) -> str:
        return " ".join(self._committed).strip()

    def configure(self, context: SttDecodeContext) -> None:
        self._context = context

    def reset(self) -> None:
        self._chunks.clear()
        self._total_samples = 0
        self._utterance_chunks.clear()
        self._utterance_samples = 0
        self._buf_offset_s = 0.0
        self._committed = []
        self._prev = []
        self._agreed_len = 0
        self._samples_since_decode = 0

    def add_pcm(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        self._chunks.append(samples)
        self._total_samples += samples.size
        self._utterance_chunks.append(samples)
        self._utterance_samples += samples.size
        self._samples_since_decode += samples.size
        # Hard cap the buffer to MAX_AUDIO_SECONDS so a long talker can't grow it without bound.
        max_samples = settings.max_audio_seconds * STREAM_SAMPLE_RATE
        if self._total_samples > max_samples:
            drop = self._total_samples - max_samples
            self._drop_samples(self._chunks, drop, rolling=True)
            self._buf_offset_s += drop / STREAM_SAMPLE_RATE
        if self._utterance_samples > max_samples:
            drop = self._utterance_samples - max_samples
            self._drop_samples(self._utterance_chunks, drop, rolling=False)

    def should_decode(self) -> bool:
        interval = settings.stt_stream_interval_ms / 1000.0 * STREAM_SAMPLE_RATE
        if self._samples_since_decode < interval or self._total_samples == 0:
            return False
        if settings.stt_stream_energy_threshold > 0:
            audio = self._rolling_audio()
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < settings.stt_stream_energy_threshold:
                return False
        return True

    def has_buffered_audio(self) -> bool:
        return self._total_samples > 0

    def is_silent(self) -> bool:
        """True if the current buffer RMS is below the energy threshold."""
        if settings.stt_stream_energy_threshold <= 0 or self._total_samples == 0:
            return False
        audio = self._rolling_audio()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms < settings.stt_stream_energy_threshold

    def decode(self) -> dict:
        """Re-decode the buffer and apply LocalAgreement-2. Returns {committed, partial}."""
        self._samples_since_decode = 0
        words = self._svc.decode_words(self._rolling_audio(), self._context)
        for w in words:  # shift to absolute time so trimming survives across decodes
            w["start"] += self._buf_offset_s
            w["end"] += self._buf_offset_s

        newly = self._commit_agreement(words)
        partial = " ".join(w["w"] for w in words[self._agreed_len:])
        self._prev = words
        self._trim()
        return {"committed": newly, "partial": partial.strip()}

    def flush(self) -> dict:
        """End of utterance: commit the current tentative tail and reset for the next utterance."""
        audio = self._utterance_audio()
        if audio.size:
            text = self._svc.transcribe_array_final(audio, self._context)["text"]
        else:
            text = ""
        self.reset()
        return {"final": text}

    _agreed_len = 0  # how many words of the current decode are confirmed committed

    def _commit_agreement(self, cur: list[dict]) -> str:
        # Longest common prefix (by word text) of the two most recent decodes = the agreed region.
        n = 0
        while n < len(cur) and n < len(self._prev) and _norm(cur[n]["w"]) == _norm(self._prev[n]["w"]):
            n += 1
        newly = [w["w"] for w in cur[self._agreed_len:n]]
        self._committed.extend(newly)
        self._agreed_len = n
        return " ".join(newly).strip()

    def _trim(self) -> None:
        # Drop audio up to the end of the last committed word so re-decode cost stays bounded.
        if self._agreed_len == 0:
            return
        cut_s = self._prev[self._agreed_len - 1]["end"] - self._buf_offset_s
        cut_samples = int(max(0.0, cut_s) * STREAM_SAMPLE_RATE)
        if cut_samples > 0 and cut_samples < self._total_samples:
            self._drop_samples(self._chunks, cut_samples, rolling=True)
            self._buf_offset_s += cut_samples / STREAM_SAMPLE_RATE
            self._prev = self._prev[self._agreed_len:]
            self._agreed_len = 0

    def _rolling_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(tuple(self._chunks))

    def _utterance_audio(self) -> np.ndarray:
        if not self._utterance_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(tuple(self._utterance_chunks))

    def _drop_samples(
        self, chunks: deque[np.ndarray], count: int, *, rolling: bool
    ) -> None:
        remaining = count
        while remaining > 0 and chunks:
            chunk = chunks[0]
            if chunk.size <= remaining:
                chunks.popleft()
                removed = chunk.size
            else:
                chunks[0] = chunk[remaining:]
                removed = remaining
            remaining -= removed
            if rolling:
                self._total_samples -= removed
            else:
                self._utterance_samples -= removed


def _norm(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum())


def collapse_repeats(text: str, max_reps: int = 2) -> str:
    """Safety net for Whisper's repetition loops ("can you can you can you …"). Collapses any n-gram
    (n=1..4) repeated more than max_reps times in a row down to a single copy. Legit doublets
    ("very very") survive (reps<=2); only runaway loops are trimmed."""
    if not text:
        return text
    words = text.split()
    if len(words) < 4:
        return text
    for n in (1, 2, 3, 4):
        out: list[str] = []
        i = 0
        while i < len(words):
            gram = words[i:i + n]
            if len(gram) < n:
                out.extend(words[i:])
                break
            reps = 1
            j = i + n
            while j + n <= len(words) and words[j:j + n] == gram:
                reps += 1
                j += n
            if reps > max_reps:
                out.extend(gram)  # keep a single copy of the repeated phrase
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)
