import io
import logging

import numpy as np
import scipy.io.wavfile

from app.config import settings

log = logging.getLogger("asa.tts")

# Maps our stable voiceId → a Kyutai pocket-tts pre-made voice name. The /tts contract stays
# engine-agnostic; this is the only pocket-tts-aware surface (swap here for another engine).
# Full voice list: https://huggingface.co/kyutai/tts-voices
VOICE_CATALOG = [
    {"id": "asa_default", "label": "ASA Default (Anna)", "voiceRef": "anna"},
    {"id": "asa_bright", "label": "ASA Bright (Eve)", "voiceRef": "eve"},
    {"id": "asa_calm", "label": "ASA Calm (George)", "voiceRef": "george"},
]


class TtsSynthesisError(Exception):
    pass


class TtsService:
    """pocket-tts (Kyutai) via its in-process Python API. The 100M model loads ONCE at startup;
    voice states are built lazily and cached (both are slow ops). generate_audio() is blocking —
    callers run it in a threadpool. Output is converted to 16-bit PCM wav for universal browser
    playback. First startup downloads the model from HuggingFace (cached in /root/.cache)."""

    def __init__(self):
        from pocket_tts import TTSModel  # imported here so module import stays cheap/testable
        self.model = TTSModel.load_model(lsd_decode_steps=settings.tts_lsd_decode_steps)
        self._refs = {v["id"]: v["voiceRef"] for v in VOICE_CATALOG}
        self._states: dict[str, object] = {}
        # Compile once at startup so the first real request is hot (compilation itself is lazy — it
        # fires on the first forward pass, which the warmup synth below triggers).
        if settings.tts_compile:
            try:
                self.model.compile(mode="reduce-overhead")
                log.info("tts model compiled (reduce-overhead)")
            except Exception as exc:  # noqa: BLE001 — compile is best-effort; fall back to eager
                log.warning("tts compile failed, running eager: %s", exc)
        # Warm the default voice (builds the voice state AND triggers compilation).
        try:
            state = self._state_for(self._resolve_ref(settings.tts_default_voice))
            self.model.generate_audio(state, "Ready.")
            log.info("tts warmup complete")
        except Exception:  # noqa: BLE001 — best-effort warmup; real errors surface on synthesize
            pass

    def list_voices(self) -> list[dict]:
        return [
            {"id": v["id"], "label": v["label"], "model": settings.tts_default_model, "language": "en"}
            for v in VOICE_CATALOG
        ]

    def _resolve_ref(self, voice_id: str | None) -> str:
        return self._refs.get(voice_id or settings.tts_default_voice, VOICE_CATALOG[0]["voiceRef"])

    def _state_for(self, voice_ref: str):
        state = self._states.get(voice_ref)
        if state is None:
            state = self.model.get_state_for_audio_prompt(voice_ref)
            self._states[voice_ref] = state
        return state

    @staticmethod
    def _to_pcm16(audio) -> bytes:
        """1D torch float tensor [-1, 1] → little-endian PCM16 bytes."""
        samples = audio.detach().cpu().numpy()
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def synthesize(self, text: str, voice_id: str | None = None) -> bytes:
        if len(text) > settings.tts_max_text_chars:
            text = text[: settings.tts_max_text_chars].rstrip()
        try:
            state = self._state_for(self._resolve_ref(voice_id))
            audio = self.model.generate_audio(state, text)  # 1D torch float tensor, [-1, 1]
            pcm16 = np.frombuffer(self._to_pcm16(audio), dtype="<i2")
            buf = io.BytesIO()
            scipy.io.wavfile.write(buf, int(self.model.sample_rate), pcm16)
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001 — wrap any engine error into our typed failure
            raise TtsSynthesisError(str(exc)) from exc

    def synthesize_stream(self, text: str, voice_id: str | None = None):
        """Yield raw PCM16 mono bytes (@ model.sample_rate) as pocket-tts decodes them, so the
        client can start playback on the first chunk. Coalesces tts_stream_coalesce chunks per yield.
        NOT thread-safe per model instance — callers must hold the single TTS slot."""
        if len(text) > settings.tts_max_text_chars:
            text = text[: settings.tts_max_text_chars].rstrip()
        try:
            state = self._state_for(self._resolve_ref(voice_id))
            pending: list[bytes] = []
            for chunk in self.model.generate_audio_stream(state, text):  # yields [samples] tensors
                pending.append(self._to_pcm16(chunk))
                if len(pending) >= settings.tts_stream_coalesce:
                    yield b"".join(pending)
                    pending = []
            if pending:
                yield b"".join(pending)
        except Exception as exc:  # noqa: BLE001 — wrap any engine error into our typed failure
            raise TtsSynthesisError(str(exc)) from exc
