from __future__ import annotations

import os
import stat
import time
from pathlib import Path


OPENAI_STT_FILE_PREFIX = "asa-openai-stt-"
OPENAI_STT_FILE_SUFFIX = ".wav"


class UnsafeTempDirectoryError(RuntimeError):
    pass


def ensure_private_directory(directory: str | Path, *, create: bool = True) -> Path:
    path = Path(directory)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeTempDirectoryError("OpenAI STT buffer path must be a real directory")
    os.chmod(path, 0o700)
    return path


def cleanup_expired_openai_stt_files(
    directory: str | Path,
    orphan_ttl_seconds: int,
    *,
    now: float | None = None,
) -> int:
    if orphan_ttl_seconds <= 0:
        raise ValueError("OpenAI STT orphan TTL must be positive")
    try:
        path = ensure_private_directory(directory, create=False)
    except FileNotFoundError:
        return 0
    cutoff = (time.time() if now is None else now) - orphan_ttl_seconds
    removed = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.name.startswith(OPENAI_STT_FILE_PREFIX) or not entry.name.endswith(
                OPENAI_STT_FILE_SUFFIX
            ):
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_mtime > cutoff:
                    continue
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                continue
            except OSError:
                continue
    return removed
