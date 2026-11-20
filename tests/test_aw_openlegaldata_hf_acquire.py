import json
import os
import stat
from pathlib import Path

import pytest

from cytir.acquire import openlegaldata_hf as subject


REV = "a" * 40


def test_validate_head_requires_302_linked_headers_and_pinned_revision(tmp_path: Path):
    row = {"relative_path": "dump-20260520/train-00000-of-00054.parquet",
           "url": "https://huggingface.co/x/resolve/" + REV + "/y", "revision": REV}
    headers = tmp_path / "h"
    headers.write_bytes(
        b"HTTP/2 302\r\nAccept-Ranges: bytes\r\nX-Linked-Size: 174966839\r\n"
        b"X-Linked-ETag: \"abc\"\r\nX-Repo-Commit: " + REV.encode() + b"\r\n\r\n")
    metrics = tmp_path / "m"
    metrics.write_text(json.dumps({"http_code": 302, "url": row["url"]}))
    value = subject._validate_head(headers, metrics, row)
    assert value["content_length"] == 174966839
    assert value["x_repo_commit"] == REV


def test_validate_head_rejects_mid_run_revision_change(tmp_path: Path):
    other = "b" * 40
    row = {"relative_path": "x", "url": "https://x/resolve/" + other + "/y",
           "revision": other}
    headers = tmp_path / "h"
    headers.write_bytes(
        b"HTTP/2 302\r\nAccept-Ranges: bytes\r\nX-Linked-Size: 10\r\n"
        b"X-Linked-ETag: \"abc\"\r\nX-Repo-Commit: " + REV.encode() + b"\r\n\r\n")
    metrics = tmp_path / "m"
    metrics.write_text(json.dumps({"http_code": 302, "url": row["url"]}))
    with pytest.raises(ValueError, match="repo commit changed mid-run"):
        subject._validate_head(headers, metrics, row)


def test_validate_head_rejects_non_302_or_missing_headers(tmp_path: Path):
    row = {"relative_path": "x", "url": "https://x/resolve/" + REV + "/y",
           "revision": REV}
    headers = tmp_path / "h"
    headers.write_bytes(b"HTTP/2 401\r\n\r\n")
    metrics = tmp_path / "m"
    metrics.write_text(json.dumps({"http_code": 401, "url": row["url"]}))
    with pytest.raises(ValueError, match="HEAD identity mismatch"):
        subject._validate_head(headers, metrics, row)


def test_interrupted_attempt_is_closed_without_discarding_partial(tmp_path: Path):
    attempts = tmp_path / "transport/x/attempts"; attempts.mkdir(parents=True)
    partial = tmp_path / "body.partial"; partial.write_bytes(b"abc")
    prefix = attempts / "attempt-0001"
    Path(str(prefix) + ".started").write_text("now\n")
    Path(str(prefix) + ".start_offset").write_text("0\n")
    subject._close_interrupted(attempts, partial)
    assert Path(str(prefix) + ".end_offset").read_text() == "3\n"
    assert Path(str(prefix) + ".rc").read_text() == "130\n"
    assert partial.read_bytes() == b"abc"


def test_discover_frontier_requires_exactly_54_contiguous_shards(tmp_path: Path, monkeypatch):
    info = {
        "id": subject.DATASET, "sha": REV, "gated": "auto",
        "siblings": [{"rfilename": f"dump-20260520/train-{i:05d}-of-00054.parquet"}
                     for i in range(54)],
        "cardData": {"dataset_info": [{
            "config_name": "dump-20260520",
            "splits": [{"num_examples": 423941}],
            "download_size": 10678094516, "dataset_size": 26686194854}]},
    }
    body = json.dumps(info).encode()

    def fake_curl_json(url, *, token_config, ua, extra=None):
        return info, {"http_code": 200, "url": url}, body

    monkeypatch.setattr(subject, "_curl_json", fake_curl_json)
    rows, revision, hint, artifacts = subject.discover_frontier(
        token_config=Path("unused"), ua="ua", root=tmp_path)
    assert revision == REV
    assert len(rows) == 54
    assert all(r["revision"] == REV for r in rows)
    assert hint["declared_num_examples"] == 423941
    assert [r.name for r in artifacts] == ["dataset_info.json", "dataset_info.curl-metrics.json"]
    assert (tmp_path / "original/manifest/dataset_info.json").read_bytes() == body


def test_discover_frontier_resumes_without_refetching(tmp_path: Path, monkeypatch):
    """A second call must NOT hit the network again -- curl's %{json} metrics
    embed live timing fields that differ every request even for an identical
    body, so re-fetching on resume would collide against the write-once
    metrics file (the exact bug hit live 2026-08-31)."""
    info = {
        "id": subject.DATASET, "sha": REV,
        "siblings": [{"rfilename": f"dump-20260520/train-{i:05d}-of-00054.parquet"}
                     for i in range(54)],
        "cardData": {"dataset_info": []},
    }
    body = json.dumps(info).encode()
    calls = []

    def fake_curl_json(url, *, token_config, ua, extra=None):
        calls.append(url)
        return info, {"http_code": 200, "url": url}, body

    monkeypatch.setattr(subject, "_curl_json", fake_curl_json)
    first = subject.discover_frontier(token_config=Path("unused"), ua="ua", root=tmp_path)
    assert len(calls) == 1

    def boom(*a, **k):
        raise AssertionError("must not re-fetch dataset_info on resume")

    monkeypatch.setattr(subject, "_curl_json", boom)
    second = subject.discover_frontier(token_config=Path("unused"), ua="ua", root=tmp_path)
    assert second[0] == first[0] and second[1] == first[1]


def test_discover_frontier_rejects_missing_shard(tmp_path: Path, monkeypatch):
    info = {
        "id": subject.DATASET, "sha": REV,
        "siblings": [{"rfilename": f"dump-20260520/train-{i:05d}-of-00054.parquet"}
                     for i in range(53)],  # one short
        "cardData": {"dataset_info": []},
    }
    monkeypatch.setattr(subject, "_curl_json",
                        lambda *a, **k: (info, {"http_code": 200}, b"{}"))
    with pytest.raises(ValueError, match="exactly shards"):
        subject.discover_frontier(token_config=Path("unused"), ua="ua", root=tmp_path)


def test_token_config_requires_0600_and_never_leaks_token_to_disk_readably(tmp_path: Path):
    token_file = tmp_path / "token"
    token_file.write_text("secret-token-value\n")
    os.chmod(token_file, 0o644)
    with pytest.raises(ValueError, match="0600"):
        subject.TokenConfig(token_file)
    os.chmod(token_file, 0o600)
    with subject.TokenConfig(token_file) as config_path:
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        contents = config_path.read_text()
        assert 'Authorization: Bearer secret-token-value' in contents
        saved_path = config_path
    assert not saved_path.exists()
