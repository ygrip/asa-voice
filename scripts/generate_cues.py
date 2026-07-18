#!/usr/bin/env python3
"""Build-time cue pack generator (setara-nx07.4, plan §10).

Generates one WAV clip per (voice, cue) pair for every voice in the catalog that supports the
given provider, using the SAME provider adapter and voice resolver the runtime sidecar uses - so
a generated pack can never drift from what the sidecar would produce on its own. Writes
cue-pack.json (fingerprint + per-file manifest) alongside the generated WAVs; never receives or
embeds a raw API key in that output.

Usage:
  python scripts/generate_cues.py --provider openai --model tts-1 --output build/cues
  python scripts/generate_cues.py --provider pocket_tts --model pocket-low --output build/cues

Exits non-zero (and prints one "error: ..." line per problem to stderr) if any generated clip
fails WAV validation - a release build must not ship a broken or truncated cue.
"""
import argparse
import asyncio
import hashlib
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python scripts/generate_cues.py` from any cwd

from app.providers.base import TtsOptions  # noqa: E402
from app.services import voice_catalog  # noqa: E402
from app.services.cue_definitions import CUE_DEFINITIONS  # noqa: E402

SCHEMA_VERSION = 1
# Sample rates any of our providers legitimately produce (pocket-tts @24kHz, OpenAI wav @24kHz).
# Reject anything else outright rather than silently accept a surprising provider default.
ALLOWED_SAMPLE_RATES = {16_000, 22_050, 24_000, 44_100, 48_000}


def build_adapter(provider: str):
    """Construct the real production adapter - never a separate, drifting generator-only copy
    of provider/voice logic (plan §10.2)."""
    if provider == "openai":
        from app.providers.openai_tts import OpenAiTtsAdapter

        return OpenAiTtsAdapter()
    if provider == "pocket_tts":
        from app.providers.pocket_tts import PocketTtsAdapter
        from app.services.tts_service import TtsService

        return PocketTtsAdapter(TtsService())
    raise ValueError(f"Unsupported provider: {provider!r}")


def compute_fingerprint(
    *, provider: str, model: str, voice_profile: str, speed: float,
    instructions: str | None, output_format: str, voice_refs: dict[str, str],
) -> str:
    """Hash every input that changes what audio comes out, per plan §10.3. Never include a secret
    - only the provider/model/voice/speed/instructions *identifiers*, not the API key."""
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "cues": {cue_id: definition.text for cue_id, definition in sorted(CUE_DEFINITIONS.items())},
        "voiceRefs": dict(sorted(voice_refs.items())),
        "provider": provider,
        "model": model,
        "voiceProfile": voice_profile,
        "speed": speed,
        "instructionsHash": hashlib.sha256((instructions or "").encode()).hexdigest(),
        "outputFormat": output_format,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# Some encoders (observed with OpenAI's TTS wav output) write a streaming placeholder - e.g.
# 0xFFFFFFFF - into the RIFF data-chunk size field instead of the true byte length. wave.getnframes()
# blindly trusts that field, which for mono 16-bit audio computes to exactly 2**31-1 frames (~24
# days). Recompute the real frame count from bytes actually readable in the file instead.
_MAX_SANE_FRAMES = 48_000 * 60 * 5  # 5 minutes at 48kHz - far beyond any legitimate cue clip


def _actual_frame_count(clip: wave.Wave_read, declared_frame_count: int, channels: int, sample_width: int) -> int:
    frame_size = channels * sample_width
    if frame_size == 0 or declared_frame_count == 0:
        return 0
    capped = min(declared_frame_count, _MAX_SANE_FRAMES)
    return len(clip.readframes(capped)) // frame_size


def validate_wav_file(path: Path, max_duration_ms: int) -> list[str]:
    """Plan §10.5: every required cue must exist, be non-empty, parse as WAV, be mono, use an
    allowed sample rate, use 16-bit PCM, and stay within its duration bound."""
    if not path.is_file() or path.stat().st_size == 0:
        return [f"{path}: missing or empty file"]
    try:
        with wave.open(str(path), "rb") as clip:
            channels = clip.getnchannels()
            rate = clip.getframerate()
            sample_width = clip.getsampwidth()
            frame_count = _actual_frame_count(clip, clip.getnframes(), channels, sample_width)
    except (wave.Error, EOFError, OSError) as exc:
        return [f"{path}: not a valid WAV file ({exc})"]

    errors: list[str] = []
    if channels != 1:
        errors.append(f"{path}: expected mono, got {channels} channel(s)")
    if rate not in ALLOWED_SAMPLE_RATES:
        errors.append(f"{path}: unexpected sample rate {rate}Hz")
    if sample_width != 2:
        errors.append(f"{path}: expected 16-bit PCM, got {sample_width * 8}-bit")
    if frame_count == 0:
        errors.append(f"{path}: zero audio frames")
    elif rate:
        duration_ms = (frame_count / rate) * 1000
        if duration_ms > max_duration_ms:
            errors.append(f"{path}: {duration_ms:.0f}ms exceeds max {max_duration_ms}ms")
    return errors


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def generate(
    *, provider: str, model: str, voice_profile: str, speed: float,
    instructions: str | None, output_format: str, out_dir: Path,
) -> dict:
    adapter = build_adapter(provider)
    voices = [entry for entry in voice_catalog.entries() if provider in entry.providers]
    if not voices:
        raise ValueError(f"No catalog voice defines a {provider!r} mapping")

    voice_refs = {entry.id: entry.providers[provider] for entry in voices}
    files: dict[str, dict] = {}

    for entry in voices:
        voice_dir = out_dir / entry.id
        voice_dir.mkdir(parents=True, exist_ok=True)
        for cue_id, definition in CUE_DEFINITIONS.items():
            options = TtsOptions(
                voice_id=entry.id, format=output_format, speed=speed,
                instructions=instructions, purpose="cue",
            )
            result = await adapter.synthesize(definition.text, options)
            try:
                with open(result.audio_path, "rb") as f:
                    audio_bytes = f.read()
            finally:
                Path(result.audio_path).unlink(missing_ok=True)

            clip_path = voice_dir / f"{cue_id}.{output_format}"
            clip_path.write_bytes(audio_bytes)

            with wave.open(str(clip_path), "rb") as clip:
                sample_rate = clip.getframerate()
                channels = clip.getnchannels()
                sample_width = clip.getsampwidth()
                frame_count = _actual_frame_count(clip, clip.getnframes(), channels, sample_width)
                duration_ms = (frame_count / sample_rate) * 1000 if sample_rate else 0

            files[f"{entry.id}/{cue_id}.{output_format}"] = {
                "providerVoice": voice_refs[entry.id],
                "sha256": _sha256_hex(audio_bytes),
                "sizeBytes": len(audio_bytes),
                "durationMs": round(duration_ms),
                "sampleRate": sample_rate,
                "channels": channels,
            }

    fingerprint = compute_fingerprint(
        provider=provider, model=model, voice_profile=voice_profile, speed=speed,
        instructions=instructions, output_format=output_format, voice_refs=voice_refs,
    )
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "provider": provider,
        "model": model,
        "voiceProfile": voice_profile,
        "speed": speed,
        "instructionsHash": hashlib.sha256((instructions or "").encode()).hexdigest(),
        "files": files,
    }
    (out_dir / "cue-pack.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_pack(out_dir: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    for relative_path, file_manifest in manifest["files"].items():
        cue_id = relative_path.rsplit("/", 1)[-1].split(".", 1)[0]
        definition = CUE_DEFINITIONS.get(cue_id)
        max_duration_ms = definition.max_duration_ms if definition else 10_000
        clip_path = out_dir / relative_path
        errors.extend(validate_wav_file(clip_path, max_duration_ms))
        if clip_path.is_file():
            actual_sha256 = _sha256_hex(clip_path.read_bytes())
            if actual_sha256 != file_manifest["sha256"]:
                errors.append(f"{clip_path}: checksum mismatch (manifest is stale)")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the ASA voice cue pack")
    parser.add_argument("--provider", required=True, choices=["openai", "pocket_tts"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice-profile", default="default")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--output-format", default="wav")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = asyncio.run(
        generate(
            provider=args.provider, model=args.model, voice_profile=args.voice_profile,
            speed=args.speed, instructions=args.instructions, output_format=args.output_format,
            out_dir=args.output,
        )
    )

    errors = validate_pack(args.output, manifest)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Generated cue pack {manifest['fingerprint']} ({len(manifest['files'])} clips) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
