from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO

from app.services.temp_audio_cleanup import (
    OPENAI_STT_FILE_PREFIX,
    OPENAI_STT_FILE_SUFFIX,
    ensure_private_directory,
)


WAV_HEADER_BYTES = 44
PCM16_SAMPLE_WIDTH_BYTES = 2


class TempAudioLimitExceeded(RuntimeError):
    pass


class IncrementalPcmWavWriter:
    def __init__(
        self,
        directory: str | Path,
        *,
        sample_rate: int,
        channels: int,
        max_file_bytes: int,
    ) -> None:
        if sample_rate <= 0 or channels <= 0 or max_file_bytes <= WAV_HEADER_BYTES:
            raise ValueError("Invalid incremental WAV writer configuration")
        private_directory = ensure_private_directory(directory)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=OPENAI_STT_FILE_PREFIX,
            suffix=OPENAI_STT_FILE_SUFFIX,
            dir=private_directory,
        )
        self.path = Path(raw_path)
        self._file: BinaryIO | None = None
        self._sample_rate = sample_rate
        self._channels = channels
        self._max_file_bytes = max_file_bytes
        self._pcm_bytes = 0
        descriptor_owned = True
        try:
            os.fchmod(descriptor, 0o600)
            self._file = os.fdopen(descriptor, "w+b", buffering=0)
            descriptor_owned = False
            self._file.write(b"\x00" * WAV_HEADER_BYTES)
        except BaseException:
            if self._file is not None:
                try:
                    self._file.close()
                except BaseException:
                    pass
                self._file = None
            elif descriptor_owned:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            try:
                self.path.unlink()
            except BaseException:
                pass
            raise

    @property
    def pcm_bytes(self) -> int:
        return self._pcm_bytes

    def append_pcm(self, frame: bytes) -> None:
        file = self._active_file()
        if WAV_HEADER_BYTES + self._pcm_bytes + len(frame) > self._max_file_bytes:
            raise TempAudioLimitExceeded(
                f"Temporary STT audio exceeds {self._max_file_bytes} bytes"
            )
        file.write(frame)
        self._pcm_bytes += len(frame)

    def finalize(self) -> Path:
        file = self._active_file()
        block_align = self._channels * PCM16_SAMPLE_WIDTH_BYTES
        byte_rate = self._sample_rate * block_align
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + self._pcm_bytes,
            b"WAVE",
            b"fmt ",
            16,
            1,
            self._channels,
            self._sample_rate,
            byte_rate,
            block_align,
            PCM16_SAMPLE_WIDTH_BYTES * 8,
            b"data",
            self._pcm_bytes,
        )
        file.seek(0)
        file.write(header)
        file.flush()
        os.fsync(file.fileno())
        file.close()
        self._file = None
        return self.path

    def close_and_delete(self) -> None:
        try:
            file = self._file
            self._file = None
            if file is not None:
                file.close()
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _active_file(self) -> BinaryIO:
        if self._file is None:
            raise RuntimeError("Temporary STT audio writer is closed")
        return self._file
