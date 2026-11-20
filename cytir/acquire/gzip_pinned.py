"""Stdlib-only gzip helpers that write the same bytes on every Python.

A leaf on purpose: writers of every layer import it, so it holds no file writes and no
`aw` imports (the drain's import closure and the -S start guard both depend on that).
"""
from __future__ import annotations

import gzip
import os

GZIP_OS_UNKNOWN = 0xFF


def deterministic_gzip(value: bytes) -> bytes:
    """Gzip member that is byte-identical on every Python version.

    gzip.compress writes the header OS byte (offset 9) per interpreter
    (0x03 on 3.12, 0xff on 3.10 and 3.14). Sealed digests were made with
    0xff, so pin it; mtime 0, level 6, no filename.
    """
    packed = bytearray(gzip.compress(value, compresslevel=6, mtime=0))
    packed[9] = GZIP_OS_UNKNOWN
    return bytes(packed)


class _PinnedHeaderGzipFile(gzip.GzipFile):
    """GzipFile whose header is the same on every Python: mtime 0, OS 0xff, no name."""

    def _write_gzip_header(self, compresslevel):
        xfl = 2 if compresslevel == 9 else 4 if compresslevel == 1 else 0
        self.fileobj.write(
            b"\x1f\x8b\x08\x00\x00\x00\x00\x00" + bytes([xfl, GZIP_OS_UNKNOWN]))


def deterministic_gzip_open(path_or_fileobj, mode: str = "wb",
                            compresslevel: int = 6) -> gzip.GzipFile:
    """Streaming twin of `deterministic_gzip`: same header, same bytes, any Python.

    Closing the returned file does not close a caller-supplied file object.
    """
    if mode not in ("wb", "xb"):
        raise ValueError(f"write modes only: {mode!r}")
    if isinstance(path_or_fileobj, (str, os.PathLike)):
        return _PinnedHeaderGzipFile(
            filename=os.fspath(path_or_fileobj), mode=mode, compresslevel=compresslevel)
    return _PinnedHeaderGzipFile(
        fileobj=path_or_fileobj, mode="wb", compresslevel=compresslevel)
