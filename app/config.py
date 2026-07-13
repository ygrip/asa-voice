from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Sidecar configuration. Every field maps to an UPPER_SNAKE env var of the same name."""

    app_name: str = "ASA Voice Sidecar"

    # Provider mode is descriptive health metadata. STT_PROVIDER and STT_FALLBACK_PROVIDER are the
    # authoritative construction policy: local (faster_whisper), hosted (OpenAI), or hybrid
    # (OpenAI with faster_whisper fallback). The default boots without OPENAI_API_KEY; hosted
    # readiness requires a non-blank key, model, and positive timeout.
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
    # False to match partial (stt_partial_vad_filter) - Silero VAD ran pre-decode on final only,
    # with its threshold/speech_pad_ms never tuned for live 16kHz mic PCM (only validated against
    # clean uploaded WAVs), and could trim a real, partial-confirmed utterance to near-nothing
    # before Whisper ever saw it (setara-s94o STT quality incident: partials showed real text, the
    # final came back empty). no_speech_threshold below already gates non-speech segments
    # post-decode, uniformly for both partial and final - keep silence handling to that one place.
    stt_final_vad_filter: bool = False
    stt_final_condition_on_previous: bool = False
    # Decode-quality knobs: drop silence/hallucinations, don't carry context between clips (faster,
    # avoids the model "completing" a previous sentence into the next short command).
    stt_no_speech_threshold: float = 0.6
    stt_vad_min_silence_ms: int = 300
    # Anti-repetition: Whisper sometimes loops a phrase ("can you can you can you…"). repetition_penalty
    # (soft, discourages during decode) plus the post-hoc collapse_repeats() safety net (stt_service.py)
    # are enough; compression_ratio_threshold flags a degenerate segment so the temperature fallback
    # (the tuple below) re-decodes it. Forcing a single temperature=0 disables that fallback — keep a
    # small ladder so a looped segment gets a second chance. no_repeat_ngram_size is left at 0
    # (disabled, the library default) - it's a hard block on ANY repeated n-gram, which would corrupt
    # legitimate repeated content a command/dictation assistant actually sees ("test test one two
    # three", spelled-out serials); the soft penalty + post-hoc collapse already cover runaway loops
    # without that risk.
    stt_repetition_penalty: float = 1.15
    stt_no_repeat_ngram_size: int = 0
    stt_compression_ratio_threshold: float = 2.4
    stt_log_prob_threshold: float = -1.0
    stt_temperatures: tuple[float, ...] = (0.0, 0.2, 0.4)
    # Bias decoding toward the assistant name and domain vocabulary so "ASA" isn't heard as "Elsa",
    # "Setara" as "set are", etc. initial_prompt seeds context; hotwords boosts these tokens.
    stt_hotwords: str = "ASA Setara"
    # Rolling-window streaming (WS /stt/stream): re-decode the buffer at most this often. Lower =
    # snappier partials but more CPU; ~600ms balances latency vs decode cost on a 4-core box.
    stt_stream_protocol_version: str = "2"
    stt_command_max_seconds: int = 15
    stt_handsfree_max_seconds: int = 30
    stt_dictation_max_seconds: int = 300
    stt_frame_duration_ms: int = 20
    stt_stream_max_frame_bytes: int = 4096
    stt_stream_max_session_bytes: int = 9_600_000
    stt_stream_queue_max_ms: int = 2000
    stt_stream_handshake_timeout_seconds: float = 10.0
    stt_stream_no_audio_idle_timeout_seconds: float = 45.0
    stt_provider_final_timeout_seconds: float = 300.0
    stt_stream_interval_ms: int = 600
    # When partial inference falls behind real time, reduce partial frequency only. Final decode
    # keeps the independent accuracy profile above.
    stt_stream_max_partial_interval_ms: int = 2400
    stt_stream_rtf_slow_threshold: float = 1.0
    # RMS energy threshold below which a buffer is considered silence and skipped. Float32 range is
    # [-1,1]; mic noise floor is ~0.005–0.02, speech is typically >0.03. Set 0 to disable.
    stt_stream_energy_threshold: float = 0.02
    # Silence duration (seconds) after which the stream auto-flushes (server-side VAD fallback).
    # Useful when the browser's VAD is energy-only and misses a pause. Set 0 to disable.
    stt_stream_silence_flush_s: float = 1.5

    # Per-mode partial-decode enablement (ASA STT accuracy/latency recovery plan, RC-05). Command
    # mode never runs a partial decoder (see routers/stt.py) - these two flags extend that to
    # dictation/hands_free: false by default because flush() blocks on any in-flight partial before
    # the final decode can start (_settle_decoder in stt_stream_scheduler.py), and a slow model can
    # make that wait 7-12s+. If re-enabled, widen STT_STREAM_INTERVAL_MS/
    # STT_STREAM_MAX_PARTIAL_INTERVAL_MS (e.g. 2500/5000) so partials can't outrun the final decode.
    stt_dictation_partials_enabled: bool = False
    stt_handsfree_partials_enabled: bool = False

    # Per-mode final-decode profiles (RC-06): the final profile previously always retried at
    # settings.stt_temperatures=[0.0,0.2,0.4] for every mode, multiplying latency whenever the
    # quality gates rejected the greedy pass - expensive for a short command. A SINGLE temperature
    # (no fallback) went too far the other way: this session's own captures showed temp=0.0 alone
    # failing the log_prob/compression_ratio gate on real (non-noise) audio again and again, with
    # temp=0.2 the one that actually succeeded - with no fallback, that failure has nowhere to go and
    # faster-whisper returns empty text, which the client reports as "Could not understand audio"
    # even though the user was heard fine. Keep one cheap fallback step (only paid when temp=0.0's
    # segment fails its own quality gate, not on every decode) instead of trusting the single-value
    # recommendation over what was actually observed.
    stt_command_beam_size: int = 2
    stt_command_best_of: int = 2
    stt_command_temperatures: tuple[float, ...] = (0.0, 0.2)
    stt_command_vad_filter: bool = False
    stt_command_condition_on_previous: bool = False

    stt_dictation_beam_size: int = 3
    stt_dictation_best_of: int = 3
    stt_dictation_temperatures: tuple[float, ...] = (0.0, 0.2)
    stt_dictation_vad_filter: bool = False
    stt_dictation_condition_on_previous: bool = True

    stt_handsfree_beam_size: int = 2
    stt_handsfree_best_of: int = 2
    stt_handsfree_temperatures: tuple[float, ...] = (0.0, 0.2)
    stt_handsfree_vad_filter: bool = False
    stt_handsfree_condition_on_previous: bool = False

    # OpenAI hosted STT (setara-s94o.6/.7/.8). gpt-4o-mini-transcribe/gpt-4o-transcribe only support
    # the json/text response formats (no segment timestamps, no reported audio duration).
    openai_api_key: str = ""
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_stt_timeout_seconds: int = 30
    openai_stt_buffer_directory: str = "/tmp/asa-voice/stt"
    openai_stt_max_temp_bytes: int = 15_728_640
    openai_stt_orphan_ttl_seconds: int = 3600
    # Short domain glossary bias (plan §8.2) — used as options.prompt's default when the caller
    # (stt_context.py) doesn't already supply a more specific one. Deliberately short: OpenAI's
    # transcription prompt is a priming glossary, not a place to dump the whole product manual.
    openai_stt_prompt: str = (
        "Common product terms: Setara, Raksara, scenario, test case, suite, execution, build, "
        "release plan, coverage, rerun failed, automation coverage, squad, tribe."
    )

    # Policy layer v1 (setara-s94o.9) — request validation before any provider runs, plus an
    # in-memory per-client daily quota. File-size validation reuses the existing max_upload_mb.
    # Plan §10.2 recommends 30s/request, but /stt already supports long PTT dictation up to
    # max_upload_seconds (see KNOWLEDGE.md, bd setara-gwfj) — default here matches that instead of
    # silently regressing it; tighten via env var for cost-sensitive hosted deployments.
    max_stt_seconds_per_request: int = 300
    max_stt_seconds_per_client_per_day: int = 600

    # TTS (Pocket TTS — behind TtsService adapter)
    tts_engine: str = "pocket-tts"
    tts_default_voice: str = "asa_default"
    tts_default_model: str = "pocket-low"
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
    # max_audio_seconds: rolling-window buffer ceiling in streaming STT, composed via min() with
    # the session's own negotiated max_duration_seconds (command=15s/hands_free=30s/
    # dictation=300s) - matches max_upload_seconds so it no longer clips dictation before its own
    # advertised limit (previously 20s, silently truncating every hands_free/dictation session -
    # setara-s94o STT quality incident). Raw PCM float32 at this ceiling is ~19MB/session, trivial
    # for the container; this exists as a safety net against a future misconfigured
    # max_duration_seconds, not a real memory constraint at 300s.
    # max_upload_seconds: limit for file-upload STT (longer recordings are fine)
    max_audio_seconds: int = 300
    max_upload_seconds: int = 300
    max_upload_mb: int = 15
    # Preferred provider-operation limits. ``None`` preserves deployments that still set the
    # legacy process-wide MAX_CONCURRENT_STT/MAX_CONCURRENT_TTS names below.
    local_stt_max_concurrent: int | None = None
    hosted_stt_max_concurrent: int = 4
    tts_max_concurrent: int | None = None
    max_concurrent_stt: int = 1
    max_concurrent_tts: int = 1

    tmp_dir: str = "/tmp/asa-voice"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def local_stt_concurrency_limit(self) -> int:
        return (
            self.local_stt_max_concurrent
            if self.local_stt_max_concurrent is not None
            else self.max_concurrent_stt
        )

    def tts_concurrency_limit(self) -> int:
        return (
            self.tts_max_concurrent
            if self.tts_max_concurrent is not None
            else self.max_concurrent_tts
        )


settings = Settings()
