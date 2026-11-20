"""Deterministic artifact writers shared by rebuild pipeline stages."""
from __future__ import annotations

import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cytir.acquire.gzip_pinned import (  # noqa: F401  re-exports
    GZIP_OS_UNKNOWN, deterministic_gzip, deterministic_gzip_open)


@contextmanager
def deterministic_gzip_text(path: str | Path, *, compresslevel: int = 6):
    """Write UTF-8 gzip with a stable header (no filename or wall-clock mtime)."""
    raw = Path(path).open("wb")
    compressed = deterministic_gzip_open(
        raw, compresslevel=compresslevel)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    try:
        yield text
    finally:
        text.close()
        raw.close()


def _json(value: Any) -> bytes:
    from cytir.acquire.artifact_io import canonical_json

    return (canonical_json(value) + "\n").encode("utf-8")


class DeterministicGzipWriter:
    def __init__(self, final_path: Path):
        self.final_path = final_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        raw = tempfile.NamedTemporaryFile(
            prefix=f".{final_path.name}.", suffix=".tmp",
            dir=final_path.parent, delete=False)
        self._raw = raw
        self._temporary = Path(raw.name)
        self._gzip = deterministic_gzip_open(raw, compresslevel=9)

    def write(self, value: Any) -> None:
        self._gzip.write(_json(value))

    def close(self) -> None:
        from cytir.acquire.artifact_io import sha256_file

        self._gzip.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        if self.final_path.exists():
            if sha256_file(self.final_path) != sha256_file(self._temporary):
                raise ValueError(f"immutable output collision: {self.final_path}")
            self._temporary.unlink()
        else:
            os.replace(self._temporary, self.final_path)
