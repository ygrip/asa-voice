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
