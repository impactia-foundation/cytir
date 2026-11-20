"""Shared helpers of the Route-C exact-record materializers.

One home for the artifact binding, the logical (decompressed) read of a
preserved artifact and the response content-type read; the pipeline Route-C
HTML materializers import these instead of carrying a copy.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from cytir.acquire import preserved_stage_io as PS
from cytir.acquire.artifact_io import sha256_file

HTML_TYPES = {"text/html", "application/xhtml+xml"}


def binding(path: Path, artifact_root: Path) -> dict:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def logical(artifact_root: Path, artifact: dict) -> bytes:
    PS.validate_artifact(artifact_root, artifact)
    return gzip.decompress(PS.artifact_path(artifact_root, artifact).read_bytes())


def content_type(artifact_root: Path, receipt: dict) -> str:
    artifact = (receipt.get("artifacts") or {}).get("response_headers")
    headers = json.loads(logical(artifact_root, artifact))
    return next((str(value).split(";", 1)[0].strip().casefold()
                 for name, value in headers if str(name).casefold() == "content-type"), "")
