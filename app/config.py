from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Sidecar configuration. Every field maps to an UPPER_SNAKE env var of the same name."""

    app_name: str = "ASA Voice Sidecar"

    # Provider mode (plan §3 / §7.1): local (faster_whisper + pocket_tts only) | hosted (OpenAI
    # primary) | hybrid (OpenAI primary + faster_whisper fallback). Only "faster_whisper"/"pocket_tts"
    # adapters exist until Phase 2, so the default MUST boot clean with zero OPENAI_API_KEY set —
    # see runtime.py, which fails fast at startup if an unimplemented provider is configured instead
    # of silently no-op'ing.
    asa_voice_mode: str = "local"
    stt_provider: str = "faster_whisper"
    stt_fallback_provider: str = "none"
    stt_allow_provider_override: bool = False
    tts_provider: str = "pocket_tts"
    tts_fallback_provider: str = "none"

    # STT (faster-whisper). distil-small.en is the sweet spot for a 3 GB / 4 CPU box: distilled so it
    # runs faster than small.en, yet noticeably more accurate than base.en. Alternatives via STT_MODEL:
    # base.en (lightest), small.en (most accurate, slower). Avoid distil-large-v3 — too heavy for ASA.
    stt_model: str = "distil-small.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "en"
    # Use all cores for CTranslate2 decode — the single biggest CPU latency win.
    stt_cpu_threads: int = 4
    stt_num_workers: int = 1
    # Partial decode is display-only and optimized for first-text latency.
    stt_partial_beam_size: int = 1
    stt_partial_best_of: int = 1
    stt_partial_word_timestamps: bool = True
    stt_partial_vad_filter: bool = False
    # Final decode is the command-execution transcript and favors accuracy.
    stt_final_beam_size: int = 3
    stt_final_best_of: int = 3
    stt_final_word_timestamps: bool = False
    stt_final_vad_filter: bool = True
    stt_final_condition_on_previous: bool = False
    stt_vad_filter: bool = True
    # Decode-quality knobs: drop silence/hallucinations, don't carry context between clips (faster,
    # avoids the model "completing" a previous sentence into the next short command).
    stt_no_speech_threshold: float = 0.6
    stt_condition_on_previous: bool = False
    stt_vad_min_silence_ms: int = 300
    # Anti-repetition: Whisper sometimes loops a phrase ("can you can you can you…"). repetition_penalty
    # + no_repeat_ngram_size discourage it during decode; compression_ratio_threshold flags a degenerate
    # segment so the temperature fallback (the tuple below) re-decodes it. Forcing a single temperature=0
    # disables that fallback — keep a small ladder so a looped segment gets a second chance.
    stt_repetition_penalty: float = 1.15
    stt_no_repeat_ngram_size: int = 3
    stt_compression_ratio_threshold: float = 2.4
    stt_log_prob_threshold: float = -1.0
    stt_temperatures: tuple[float, ...] = (0.0, 0.2, 0.4)
    # Bias decoding toward the assistant name and domain vocabulary so "ASA" isn't heard as "Elsa",
    # "Setara" as "set are", etc. initial_prompt seeds context; hotwords boosts these tokens.
    stt_prompt: str = (
        "This is a voice command for ASA, the AI assistant inside Setara, a test management "
        "platform with projects, plans, builds, scenarios, squads, and tribes."
    )
    stt_hotwords: str = "ASA Setara"
    # Rolling-window streaming (WS /stt/stream): re-decode the buffer at most this often. Lower =
    # snappier partials but more CPU; ~600ms balances latency vs decode cost on a 4-core box.
    stt_stream_interval_ms: int = 600
    # RMS energy threshold below which a buffer is considered silence and skipped. Float32 range is
    # [-1,1]; mic noise floor is ~0.005–0.02, speech is typically >0.03. Set 0 to disable.
    stt_stream_energy_threshold: float = 0.02
    # Silence duration (seconds) after which the stream auto-flushes (server-side VAD fallback).
    # Useful when the browser's VAD is energy-only and misses a pause. Set 0 to disable.
    stt_stream_silence_flush_s: float = 1.5

    # TTS (Pocket TTS — behind TtsService adapter)
    tts_engine: str = "pocket-tts"
    tts_default_voice: str = "asa_default"
    tts_default_model: str = "pocket-low"
    tts_format: str = "wav"
    tts_sample_rate: int = 24000
    tts_max_text_chars: int = 600
    # torch.compile the model once at startup (reduce-overhead) — first synth pays the compile cost,
    # later calls are faster. Disable if compile is unstable on a given platform.
    tts_compile: bool = True
    # pocket-tts diffusion (latent-space-decode) steps. pocket-tts 2.1.0 already defaults to 1 (the
    # minimum); the "10" figure is the full Kyutai DSM, not this packaged model. Cannot be lowered —
    # raise (>1) only to trade speed for quality. Kept at 1 for fastest inference.
    tts_lsd_decode_steps: int = 1
    # Stream chunk pacing: number of generate_audio_stream chunks to coalesce before flushing to the
    # client. 1 = lowest latency, more frames; higher = fewer/bigger frames.
    tts_stream_coalesce: int = 1

    # Auth: comma-separated "client_id:secret" pairs (same format as asa-rust-voice).
    # Empty = open / dev mode (no key required).
    allowed_clients: str = ""

    # Resource guards (protect the 4 CPU / 3 GB capped container)
    # max_audio_seconds: rolling-window buffer cap per utterance in streaming STT
    # max_upload_seconds: limit for file-upload STT (longer recordings are fine)
    max_audio_seconds: int = 20
    max_upload_seconds: int = 300
    max_upload_mb: int = 15
    max_concurrent_stt: int = 1
    max_concurrent_tts: int = 1

    tmp_dir: str = "/tmp/asa-voice"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
