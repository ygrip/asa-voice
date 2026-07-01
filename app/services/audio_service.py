import os
import subprocess
import tempfile

from app.config import settings


class AudioTooLarge(Exception):
    pass


class AudioTooLong(Exception):
    pass


def check_upload_size(num_bytes: int) -> None:
    limit = settings.max_upload_mb * 1024 * 1024
    if num_bytes > limit:
        raise AudioTooLarge(f"Upload exceeds {settings.max_upload_mb} MB")


def probe_duration_seconds(path: str) -> float | None:
    """Probe audio duration via ffprobe (ffmpeg ships in the image). Returns None if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        value = out.stdout.strip()
        return float(value) if value else None
    except (subprocess.SubprocessError, ValueError):
        return None


def write_temp(audio_bytes: bytes, suffix: str) -> str:
    """Persist an upload to the tmp dir; caller is responsible for removing it."""
    os.makedirs(settings.tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=settings.tmp_dir, suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        return f.name


def enforce_duration(path: str) -> None:
    duration = probe_duration_seconds(path)
    if duration is not None and duration > settings.max_audio_seconds:
        raise AudioTooLong(
            f"Audio is {duration:.1f}s; max is {settings.max_audio_seconds}s"
        )
