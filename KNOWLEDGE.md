## STT Stream Pattern

- Treat normal browser WebSocket close as a quiet terminal state. Guard every `send_json` from streaming STT with a
  shared closed flag so decode, auto-flush, and explicit flush paths do not keep sending after the close frame.
- Streaming STT decode errors are non-fatal only while the connection is open. If the client has already gone away,
  stop work quietly instead of logging repeated stack traces.
- Explicit stream `flush` must decode any buffered audio before finalizing, even when the rolling partial interval or
  RMS gate did not allow a partial decode yet. Short valid utterances should not finalize to an empty transcript just
  because they ended before the partial interval.
