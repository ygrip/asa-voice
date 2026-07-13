import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings
from app.services.stt_context import SttDecodeContext, build_hotwords, resolve_language

log = logging.getLogger("asa.stt.service")


@dataclass(frozen=True)
class SttDecodeProfile:
    beam_size: int
    best_of: int
    temperatures: tuple[float, ...]
    vad_filter: bool
    condition_on_previous: bool


def profile_for_mode(mode: str) -> SttDecodeProfile:
    """Mode-specific final-decode profile (ASA STT accuracy/latency recovery plan, RC-06). Any
    mode other than dictation/hands_free - including an unrecognized/future one - falls back to
    the command profile: the fastest, most deterministic option, not the slowest."""
    if mode == "dictation":
        return SttDecodeProfile(
            beam_size=settings.stt_dictation_beam_size,
            best_of=settings.stt_dictation_best_of,
            temperatures=settings.stt_dictation_temperatures,
            vad_filter=settings.stt_dictation_vad_filter,
            condition_on_previous=settings.stt_dictation_condition_on_previous,
        )
    if mode == "hands_free":
        return SttDecodeProfile(
            beam_size=settings.stt_handsfree_beam_size,
            best_of=settings.stt_handsfree_best_of,
            temperatures=settings.stt_handsfree_temperatures,
            vad_filter=settings.stt_handsfree_vad_filter,
            condition_on_previous=settings.stt_handsfree_condition_on_previous,
        )
    return SttDecodeProfile(
        beam_size=settings.stt_command_beam_size,
        best_of=settings.stt_command_best_of,
        temperatures=settings.stt_command_temperatures,
        vad_filter=settings.stt_command_vad_filter,
        condition_on_previous=settings.stt_command_condition_on_previous,
    )


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
                # No initial_prompt: on gapped/multi-region (VAD-collected) audio it makes Whisper
                # terminate after the first segment, truncating long utterances. hotwords give the
                # same domain biasing (ASA/Setara/entity names) without the truncation.
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
            # Single greedy pass, no temperature-fallback ladder: a short/incomplete rolling-buffer
            # snippet often decodes to garbage (high compression_ratio) at temp=0, and the full
            # accuracy ladder (settings.stt_temperatures) used by the final profile then re-decodes
            # the WHOLE snippet at each successive temperature - a live capture showed this taking
            # 13s for 1s of audio on a partial. flush() blocks on this in-flight decode before the
            # final can even start (_settle_decoder in stt_stream_scheduler.py), so a slow partial
            # was adding 10+ seconds of pure dead time to every command's response (setara-s94o STT
            # latency incident). Partial text is a live-preview/last-resort-fallback value only
            # (see stt-session.ts finishDegraded()), never the authoritative transcript, so it isn't
            # worth the accuracy ladder's cost.
            temperature=0.0,
            repetition_penalty=settings.stt_repetition_penalty,
            no_repeat_ngram_size=settings.stt_no_repeat_ngram_size,
            compression_ratio_threshold=settings.stt_compression_ratio_threshold,
            log_prob_threshold=settings.stt_log_prob_threshold,
            condition_on_previous_text=False,
            no_speech_threshold=settings.stt_no_speech_threshold,
            # No initial_prompt (see transcribe): avoids first-segment early-termination.
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
        self, audio: np.ndarray, context: SttDecodeContext | None = None, mode: str = "command"
    ) -> dict:
        """Decode a complete utterance with the mode-specific accuracy profile (RC-06) used for
        command execution / dictation / hands-free."""
        hotwords = build_hotwords(context)
        audio_seconds = audio.size / 16000
        profile = profile_for_mode(mode)
        started = time.monotonic()
        segments, info = self.model.transcribe(
            audio,
            language=resolve_language(context),
            vad_filter=profile.vad_filter,
            vad_parameters={"min_silence_duration_ms": settings.stt_vad_min_silence_ms},
            beam_size=profile.beam_size,
            best_of=profile.best_of,
            temperature=profile.temperatures,
            repetition_penalty=settings.stt_repetition_penalty,
            no_repeat_ngram_size=settings.stt_no_repeat_ngram_size,
            compression_ratio_threshold=settings.stt_compression_ratio_threshold,
            log_prob_threshold=settings.stt_log_prob_threshold,
            condition_on_previous_text=profile.condition_on_previous,
            no_speech_threshold=settings.stt_no_speech_threshold,
            # No initial_prompt (see transcribe): avoids first-segment early-termination.
            hotwords=hotwords,
            word_timestamps=settings.stt_final_word_timestamps,
        )
        collected = []
        full_text = []
        segment_count = 0
        # faster-whisper's `segments` is a lazy generator - the actual beam-search/temperature-
        # fallback decode work happens HERE, during iteration, not in the model.transcribe() call
        # above. `temperature` > 0.0 on a segment means the first (greedy) pass failed the
        # compression_ratio/log_prob/no_speech quality gates and got re-decoded from scratch at a
        # higher temperature - each retry is a full extra decode pass, which is the usual cause of
        # a final decode taking noticeably longer than the audio itself (RTF > 1).
        for segment in segments:
            segment_count += 1
            log.debug(
                "stt final segment temp=%.2f avg_logprob=%.3f compression_ratio=%.3f "
                "no_speech_prob=%.3f text=%r",
                segment.temperature,
                segment.avg_logprob,
                segment.compression_ratio,
                segment.no_speech_prob,
                segment.text,
            )
            text = segment.text.strip()
            if text:
                full_text.append(text)
                collected.append({"start": segment.start, "end": segment.end, "text": text})
        elapsed = time.monotonic() - started
        log.info(
            "stt final decode audio_s=%.2f elapsed_s=%.2f rtf=%.2f segments=%d hotwords=%d",
            audio_seconds,
            elapsed,
            elapsed / audio_seconds if audio_seconds > 0 else 0.0,
            segment_count,
            len(hotwords.split()) if hotwords else 0,
        )
        return {
            "text": collapse_repeats(" ".join(full_text).strip()),
            "segments": collected,
            "language": info.language,
            "durationSeconds": info.duration,
            "engine": "faster-whisper",
            "model": settings.stt_model,
        }


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
