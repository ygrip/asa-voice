"""Stage 0 proof-of-concept benchmark for the Moonshine STT migration
(asa-moonshine-migration-plan, setara-ri76.6).

English-only scope: Moonshine (moonshine-voice, pinned ==0.0.68) has no
Indonesian model in its catalog and true streaming architectures
(tiny/small/medium-streaming) only exist for English — other languages are
non-commercially-licensed batch-only models. Indonesian/code-switch benchmarking
was dropped from this corpus for that reason (see fixtures/stt/manifest.json).

Usage:
    /tmp/asa-venv311/bin/python scripts/benchmark_moonshine.py
    /tmp/asa-venv311/bin/python scripts/benchmark_moonshine.py --model-arch tiny-streaming
    /tmp/asa-venv311/bin/python scripts/benchmark_moonshine.py --no-realtime --chunk-ms 100

Real-time pacing (default on) sleeps between add_audio() calls to match live
mic cadence, since first-partial/final latency is meaningless if a whole
utterance is fed in one wall-clock burst.
"""

import argparse
import json
import sys
import time
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "stt"

MODEL_ARCH_CHOICES = ["tiny-streaming", "small-streaming", "medium-streaming", "tiny", "base"]


def word_error_rate(expected: str, actual: str) -> float:
    """Word-level Levenshtein distance normalized by expected word count."""
    ref = expected.lower().split()
    hyp = actual.lower().strip(".").split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # standard DP edit distance over words
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(hyp)] / len(ref)


def entities_match(actual: str, entities: list[str]) -> bool:
    lowered = actual.lower()
    return all(entity.lower() in lowered for entity in entities)


def run_streaming(transcriber, audio, sample_rate, chunk_ms, realtime):
    from moonshine_voice import TranscriptEventListener

    events = []

    class Listener(TranscriptEventListener):
        def on_line_started(self, event):
            events.append(("started", time.monotonic(), event.line.text))

        def on_line_text_changed(self, event):
            events.append(("changed", time.monotonic(), event.line.text))

        def on_line_completed(self, event):
            events.append(("completed", time.monotonic(), event.line.text))

        def on_error(self, event):
            events.append(("error", time.monotonic(), str(event.error)))

    stream = transcriber.create_stream(update_interval=0.25)
    stream.add_listener(Listener())
    stream.start()

    chunk_n = max(1, int(sample_rate * chunk_ms / 1000))
    chunk_s = chunk_n / sample_rate
    t_start = time.monotonic()
    for i in range(0, len(audio), chunk_n):
        stream.add_audio(audio[i : i + chunk_n], sample_rate=sample_rate)
        if realtime:
            time.sleep(chunk_s)

    t_stop_requested = time.monotonic()
    final = stream.stop()
    t_final = time.monotonic()
    stream.close()

    first_partial_ms = None
    for _kind, ts, _text in events:
        first_partial_ms = round((ts - t_start) * 1000)
        break

    final_text = " ".join(line.text for line in final.lines).strip()
    return {
        "firstPartialMs": first_partial_ms,
        "finalMsAfterStop": round((t_final - t_stop_requested) * 1000, 1),
        "totalElapsedMs": round((t_final - t_start) * 1000, 1),
        "transcript": final_text,
        "eventCount": len(events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en")
    parser.add_argument("--model-arch", default="small-streaming", choices=MODEL_ARCH_CHOICES)
    parser.add_argument("--corpus", default=str(FIXTURES_DIR / "manifest.json"))
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.add_argument("--output", default=None, help="write JSON report here")
    args = parser.parse_args()

    from moonshine_voice import Transcriber, get_model_for_language, load_wav_file, ModelArch

    arch_by_name = {
        "tiny-streaming": ModelArch.TINY_STREAMING,
        "small-streaming": ModelArch.SMALL_STREAMING,
        "medium-streaming": ModelArch.MEDIUM_STREAMING,
        "tiny": ModelArch.TINY,
        "base": ModelArch.BASE,
    }
    wanted_arch = arch_by_name[args.model_arch]

    model_path, model_arch = get_model_for_language(args.language, wanted_arch)
    print(f"model: {model_path} arch={model_arch}", file=sys.stderr)
    transcriber = Transcriber(model_path=model_path, model_arch=model_arch)

    manifest = json.loads(Path(args.corpus).read_text())
    corpus_dir = Path(args.corpus).parent

    results = []
    for fixture in manifest["fixtures"]:
        wav_path = corpus_dir / fixture["file"]
        audio, sample_rate = load_wav_file(str(wav_path))
        audio_seconds = round(len(audio) / sample_rate, 3)

        streaming = run_streaming(transcriber, audio, sample_rate, args.chunk_ms, args.realtime)
        wer = word_error_rate(fixture["expected"], streaming["transcript"])
        entity_match = entities_match(streaming["transcript"], fixture.get("entities", []))

        result = {
            "file": fixture["file"],
            "condition": fixture.get("condition"),
            "model": f"moonshine-{args.model_arch}-{args.language}",
            "audioSeconds": audio_seconds,
            "firstPartialMs": streaming["firstPartialMs"],
            "finalMsAfterStop": streaming["finalMsAfterStop"],
            "totalElapsedMs": streaming["totalElapsedMs"],
            "transcript": streaming["transcript"],
            "expected": fixture["expected"],
            "wordErrorRate": round(wer, 3),
            "entityMatch": entity_match,
            "languageClass": "english",
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    report = {"modelArch": args.model_arch, "language": args.language, "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
