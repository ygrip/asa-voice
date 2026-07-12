## STT Stream Pattern

- Keep STT WebSocket v2 wire schemas and deterministic parsers in
  `app/services/stt_stream_protocol.py`. Start controls require protocol version `2`, PCM16 mono
  16 kHz audio in fixed 20 ms frames, bounded durations and control size, known modes, and provider
  policy that defaults to `auto`. Explicit providers require a trusted override decision.
- Only `provider_final` and `local_recovered_final` are authoritative. Correlation metadata may expose
  request, session, and client IDs for diagnostics but must not contain transcripts or audio content.
- Create every streaming provider session through `app/providers/streaming/factory.py`; the router
  owns WebSocket/auth/control flow while the provider session owns configure, PCM append, partial
  decode, flush, reset, close, metrics, bounded audio, finality, and cleanup state. Keep the local
  LocalAgreement implementation in `faster_whisper_session.py`, not in the model service or router.
- Bound both rolling and full-utterance local PCM deques while preserving committed text outside the
  audio windows. Once the full utterance is capped, finalize from committed text plus an accurate
  tail decode so beginning, middle, and end survive. Frame, alignment, byte, mode-duration, and
  queued-speech limits fail with explicit protocol errors instead of dropping audio silently.
- Keep exactly one WebSocket reader task and one `SttStreamScheduler` decoder task per connection.
  The reader appends into the provider session's bounded audio state and services controls without
  awaiting inference. Decode requests use one level-triggered event, so frame bursts coalesce and
  never allocate a job per frame. Acquire the local decode operation slot in the decoder rather
  than gating WebSocket ingestion or skipping a signal while local inference is busy.
- Snapshot rolling audio on the event loop before partial inference and apply LocalAgreement state
  after inference returns. Frames may continue appending to the deques while the worker reads only
  its immutable NumPy snapshot. Subtract only speech represented by that snapshot from overload
  debt, including when a partial decode fails.
- Track partial decode real-time factor in the scheduler. Back off only the partial cadence, within
  `STT_STREAM_INTERVAL_MS` and `STT_STREAM_MAX_PARTIAL_INTERVAL_MS`, when RTF exceeds
  `STT_STREAM_RTF_SLOW_THRESHOLD`; never change the provider's final accuracy profile.
- Flush gates new ingestion, coalesces away redundant partial signals, safely settles an in-flight
  decode, emits at most one accurate final, and clears audio in `finally`. Explicit reset re-arms a
  fresh decoder/session generation. Close waits for non-cancellable local inference to settle before
  provider cleanup, then leaves no reader or decoder task. Never abandon a `run_in_threadpool`
  decode and reset its rolling buffers concurrently because cancelling the await does not stop the
  worker thread.
- Hosted OpenAI streams never accumulate PCM in Python collections. Create a random 0600 WAV in the
  dedicated 0700 buffer directory, reserve its header, append PCM frames directly, and patch the
  header in place only at flush. The provider session owns the path and deletes it in every terminal
  path, including timeout, provider failure, limit rejection, reset, cancel, and disconnect.
- Reuse `SttProviderRouter.transcribe()` for a hosted stream final so provider classification,
  policy, quota recording, fallback metadata, and the existing OpenAI adapter remain one path.
  Hosted buffered sessions advertise no partial support and emit exactly one `provider_final`.
- Keep concurrency provider-operation specific through `OperationLimiter`: local Faster Whisper
  decode defaults to one slot, hosted STT upload/request to four independent slots, and TTS
  synthesis to one slot. Acquisition is immediate and deterministic; N+1 returns a typed retryable
  busy response instead of waiting without a bound. Preferred environment names are
  `LOCAL_STT_MAX_CONCURRENT`, `HOSTED_STT_MAX_CONCURRENT`, and `TTS_MAX_CONCURRENT`; legacy
  `MAX_CONCURRENT_STT` and `MAX_CONCURRENT_TTS` remain fallbacks when preferred values are unset.
- Acquire each slot once at the expensive boundary. Local batch adapters and the local streaming
  scheduler share the local limiter. Hosted batch and buffered-stream finalization share the OpenAI
  adapter's hosted limiter. Batch TTS acquires in its adapter; raw streaming TTS acquires before
  response headers and holds the lease only while synthesis iterates. Never add a second router
  gate around these calls, and never treat a busy response as provider fallback eligibility.
- A streaming response that acquires before headers must own idempotent cleanup in both the body
  iterator's `finally` and a response background fallback, and release on response-construction
  failure. If cancellation arrives while a threadpool synthesis step is running, wait for that
  non-abandonable worker before releasing the operation slot so another synthesis cannot overlap it.
- Keep stream deadlines operation-specific: the sidecar owns start-handshake and no-audio idle
  deadlines, hosted provider finalization owns its provider ceiling, and mode duration remains a
  negotiated protocol limit. Do not cancel local threadpool inference and release its limiter while
  the worker is still running merely to satisfy a wall-clock deadline.
- Voice metrics accept only the fixed P0.10 names and bounded label values in `voice_metrics.py`.
  Correlation IDs, transcripts, PCM, temporary paths, credentials, and provider error strings are
  never labels. The registry is an instrumentation adapter, not a new metrics endpoint or backend.
- Orphan cleanup is restricted to regular files with the owned OpenAI STT prefix and suffix, never
  follows symlinks, and removes only entries older than the configured TTL. Run it best-effort only
  when OpenAI is selected as primary or fallback. Local-only startup must neither create nor inspect
  hosted storage; an active hosted session still fails explicitly if its private directory is unsafe.
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
  `runtime.stt_router`/`runtime.tts_router` (`app/providers/router.py`'s `SttProviderRouter`/`TtsProviderRouter`) -
  never call raw engines directly from a router. Streaming STT selects provider-owned sessions through its factory;
  `/tts/stream` remains a lower-level path because it exposes the local engine's raw PCM generator.
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
  model loading) - never silently no-op into a broken mode. A primary STT load/config failure reports 503; an
  unavailable fallback or TTS component keeps usable primary STT ready and reports explicit degraded state.
- Derive local STT construction from `STT_PROVIDER` and `STT_FALLBACK_PROVIDER`, not the descriptive mode label.
  Keep faster-whisper service and adapter imports behind the local loader and local streaming route so hosted-only
  startup neither imports nor constructs the local engine. OpenAI readiness requires a non-blank key/model and a
  positive timeout; adapter construction alone is not readiness.
- `/health` gates HTTP readiness on the configured primary STT while reporting local STT, hosted STT, fallback, and
  TTS components independently. A usable primary with an unavailable fallback remains HTTP 200 with a warning.
  `/models` reports provider-specific model metadata and actual loaded/available state rather than static local
  defaults.
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
