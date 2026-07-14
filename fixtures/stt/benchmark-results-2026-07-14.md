# Moonshine Stage 0 benchmark — 2026-07-14

Package: `moonshine-voice==0.0.68` (pinned). Scope: English-only pilot (no Indonesian
model exists in the catalog; see epic setara-ri76 scope decision). Corpus:
`fixtures/stt/manifest.json` (synthetic TTS clean speech only — no real-mic
noise conditions recorded yet, see caveat below).

Models tested: `tiny-streaming-en`, `small-streaming-en`, `medium-streaming-en`.
Full raw JSON: `/tmp/moonshine-bench-results/*.json` (not checked in, rerun via
`scripts/benchmark_moonshine.py` to regenerate).

## Pass gate (from asa-moonshine-migration-plan.md §6)

| Gate | Target | Result |
|---|---|---|
| Raksara/Setara entity accuracy | >= 95% clean | **0%** (0/6 across all 3 models) |
| First stable partial | < 700ms | inconsistent: 586–1419ms, only hit on 1 fixture on tiny |
| Final after stop | < 500ms | pass (1–8ms — genuinely fast, no re-decode) |
| WER vs baseline | materially better | 0.17–1.0, incl. two silent (empty-text) results on short utterances |

**Verdict: FAIL.** Every one of the 3 streaming model sizes mangles or drops
both product entities on every attempt:

- "Raksara" → "your" (tiny/small), "an" (tiny), "the release" (medium) — never correct
- "Setara" → dropped entirely, transcript truncates to "Show automation covers[,]" — never correct

This isn't a borderline miss — it's 0/6 across 3 model sizes on synthetic
clean speech, the easiest case. There is no hotwords/initial-prompt/vocabulary-bias
API in `moonshine-voice` (grepped `transcriber.py`, `moonshine_api.py` —
nothing) to steer it toward domain vocabulary the way
`STT_HOTWORDS`/`initial_prompt` already do for faster-whisper. Nothing to tune.

Two short acknowledgement-style utterances ("Yes?", "Done.") also came back
as **empty transcript** on some models — the same "Could not understand
audio" failure mode this session already fixed once for faster-whisper's
temperature ladder, but here there's no equivalent fallback knob.

## Caveat

Corpus is synthetic TTS (macOS `say` + reused Pocket-TTS asa_default cue
audio), not real recorded mic audio — the plan's fan-noise/keyboard-noise/
far-mic/Android/Bluetooth conditions were not run. That would only make these
numbers worse, not better, so it doesn't change the recommendation.

## Recommendation

Do not proceed to PR2–7. Moonshine fails Stage 0's own pass gate on the two
requirements that most directly threaten Setara's actual usage (product
entity accuracy, consistent sub-700ms partials), with no tuning path
available in the current package to close the gap. Keep faster-whisper.
