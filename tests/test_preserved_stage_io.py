from pathlib import Path

import pytest

from cytir.acquire import preserved_stage_io as subject


def test_roundtrip_dedup_and_tamper(tmp_path: Path):
    first = subject.store_artifact(tmp_path, "markdown", "md", b"hello")
    second = subject.store_artifact(tmp_path, "markdown", "md", b"hello")
    assert first == second
    assert subject.load_validated_artifact(tmp_path, first) == b"hello"
    subject.validate_artifact(tmp_path, first)
    path = subject.artifact_path(tmp_path, first)
    path.write_bytes(b"bad")
    with pytest.raises(subject.PreservedArtifactError):
        subject.validate_artifact(tmp_path, first)


def test_unsafe_artifact_uri_fails(tmp_path: Path):
    with pytest.raises(subject.PreservedArtifactError):
        subject.artifact_path(tmp_path, {"uri": "artifact://../escape"})


def test_deterministic_gzip_is_frozen_across_python_versions():
    packed = subject.deterministic_gzip(b"x")
    assert packed[9] == 0xFF
    assert packed.hex() == "1f8b08000000000000ffab00008316dc8c01000000"


def test_deterministic_gzip_open_header_and_one_shot_identity(tmp_path):
    import gzip
    import io

    from cytir.acquire.preserved_stage_io import deterministic_gzip, deterministic_gzip_open

    body = b"<html>" + bytes(range(256)) * 200 + b"hello world " * 3000
    sink = io.BytesIO()
    with deterministic_gzip_open(sink) as stream:
        for start in range(0, len(body), 7001):
            stream.write(body[start:start + 7001])
    assert sink.getvalue()[:10] == bytes.fromhex("1f8b0800" "00000000" "00ff")
    assert sink.getvalue() == deterministic_gzip(body)
    path = tmp_path / "named.bin.gz"
    with deterministic_gzip_open(path) as stream:
        stream.write(body)
    assert path.read_bytes() == deterministic_gzip(body)
    assert gzip.decompress(path.read_bytes()) == body
