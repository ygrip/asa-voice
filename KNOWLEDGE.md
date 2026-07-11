## STT Stream Pattern

- Treat normal browser WebSocket close as a quiet terminal state. Guard every `send_json` from streaming STT with a
  shared closed flag so decode, auto-flush, and explicit flush paths do not keep sending after the close frame.
- Streaming STT decode errors are non-fatal only while the connection is open. If the client has already gone away,
  stop work quietly instead of logging repeated stack traces.
- Explicit stream `flush` must decode any buffered audio before finalizing, even when the rolling partial interval or
  RMS gate did not allow a partial decode yet. Short valid utterances should not finalize to an empty transcript just
  because they ended before the partial interval.
- Keep batch and streaming TTS formats explicit and distinct. `/tts` accepts WAV only, while `/tts/stream` accepts
  `pcm` or `l16` and declares raw PCM16 little-endian through `audio/l16`, sample-rate/channel headers, and
  `X-Sample-Format: s16le`. Never feed the raw stream to a browser compressed-audio decoder.
- Treat streaming STT partials as display-only output. Use independently configured greedy decoding for rolling
  partials and a full-utterance accuracy profile for the final transcript that can trigger commands. Merge bounded
  request context with static prompt/hotwords and reject unknown context fields.
- Append microphone PCM as bounded deque chunks. Materialize NumPy arrays only at decode boundaries, retain a separate
  full-utterance deque for final decoding, and trim both stores to the configured duration cap.
- Keep core-to-sidecar final STT file-free through `/stt/raw`. Validate the exact PCM media type and sample metadata,
  reject partial samples and over-duration bodies before decoding, and bound base64url request context independently.
- Convert custom Whisper checkpoints once at an immutable source revision with separate build dependencies. Publish
  only atomically completed artifacts carrying source metadata and `.asa_model_ready`; runtime never converts by
  default. Model architecture, language coverage, license, tokenizer compatibility, RSS, and entity accuracy are
  independent deployment gates.

## Provider Adapter Pattern (setara-s94o Phase 1)

- STT/TTS providers are wrapped behind `app/providers/base.py`'s `SttAdapter`/`TtsAdapter` protocols and their shared
  `SttOptions`/`SttResult`/`SttSegment`/`TtsOptions`/`TtsResult` dataclasses. Every route goes through
  `runtime.stt_router`/`runtime.tts_router` (`app/providers/router.py`'s `SttProviderRouter`/`TtsProviderRouter`) —
  never call `runtime.stt_service`/`runtime.tts_service` (the raw engines) directly from a router except for the
  streaming WS session and `/tts/stream`, which are lower-level paths orthogonal to provider selection (only
  faster-whisper does rolling-window streaming; hosted realtime STT is a separate code path, see setara-s94o.18).
- `SttAdapter.transcribe(audio_path, options)` is the file-based entry point (used by `/stt`). The router also
  exposes `transcribe_array(audio, options)`, a file-free fast path used by `/stt/raw` and the streaming session's
  flush — this only local providers implement; do not force hosted providers (which require an uploaded file) to
  support it. Preserving this array-based path matters: `/stt/raw` is deliberately file-free (no temp-file round
  trip) for the core-to-sidecar final-decode leg.
- `TtsAdapter.synthesize()` returns a `TtsResult.audio_path`, not raw bytes — adapters write synthesized audio to a
  temp file via the existing `audio_service.write_temp()` helper (reused, not duplicated) so every provider
  (including the future OpenAI TTS adapter) hands back the same file-based shape; routers read the file and clean
  it up. This is a deliberate small perf trade (one extra disk round-trip per `/tts` call) for provider symmetry.
- New STT/TTS providers must be registered in `runtime.SUPPORTED_STT_PROVIDERS`/`SUPPORTED_TTS_PROVIDERS` (and the
  `*_FALLBACK_PROVIDERS` sets) and in `runtime.build_stt_adapter`/`build_tts_adapter`. Configuring an unregistered
  provider name must fail fast at boot (`runtime.validate_provider_config`, called from `main.py`'s lifespan before
  model loading) — never silently no-op into a broken mode. This is distinct from a model *load* failure, which
  still degrades gracefully (`/health` reports 503, process stays up).
- An OpenAI-compatible HTTP surface does not make an ASR engine a Faster Whisper model. External engines such as
  achetronic/parakeet belong behind their own named provider adapter and health boundary. Batch multipart compatibility,
  file-free PCM finalization, and bidirectional WebSocket partials are separate capabilities and must be gated
  independently; never promote a provider based only on `/v1/audio/transcriptions` compatibility.
- Treat STT alternatives by architecture, not branding. LiteASR is a custom Transformers/Triton or MLX Whisper runtime,
  not a CTranslate2 model; keep it in a batch GPU/MLX benchmark lane. Moonshine's incremental PCM API is a better match
  for raw finalization and live partials, but current permissive licensing is English-only and Indonesian is unsupported.
  Evaluate it server-side first so Core authorization, provenance, quotas, and UI contracts remain unchanged.
- Keep `.env.example` equal to the active `Settings` field set and keep every non-metadata setting referenced by runtime
  code. Do not retain compatibility variables that Pydantic silently ignores. Platform-prefixed variables must map to
  these exact sidecar names in `setara-platform/docker-compose.yml`.
- `app/schemas.py`'s wire-facing models were reconciled, not replaced: `SttResponse` was renamed to the
  provider-agnostic field set (`provider`/`durationMs`/`latencyMs`/`fallbackUsed`/...) since it had exactly one real
  external consumer (`setara-core`'s `AsaVoiceSessionService.resolve()`, updated alongside); `HealthResponse`/
  `ModelsResponse` were extended additively (new `mode`/`provider`/`ready`/`activeProvider` fields sit next to the
  existing `device`/`computeType`/`engine` fields) because those have no real external consumer yet and existing
  tests assert the old field names verbatim.
