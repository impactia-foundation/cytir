import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from cytir.sources.justice_canada.snapshot import seal


COMMIT = "a" * 40
URL = f"https://codeload.github.com/justicecanada/laws-lois-xml/tar.gz/{COMMIT}"


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "source"
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    archive = input_dir / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        root_info = tarfile.TarInfo(f"laws-lois-xml-{COMMIT}")
        root_info.type = tarfile.DIRTYPE
        tar.addfile(root_info)
        for name in ("eng/acts/A.xml", "fra/lois/A.xml"):
            body = b"<Statute><Section id='1'>Text</Section></Statute>"
            info = tarfile.TarInfo(f"laws-lois-xml-{COMMIT}/{name}")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    headers = input_dir / "snapshot.headers"
    headers.write_bytes(b"HTTP/2 200\r\ncontent-type: application/x-gzip\r\n\r\n")
    metrics = input_dir / "snapshot.curl.json"
    metrics.write_text(json.dumps({"http_code": 200, "url_effective": URL}))
    transport = input_dir / "snapshot.transport.json"
    transport.write_text(json.dumps({
        "schema": "aw.http-fetch-metadata.v1",
        "requested_url": URL,
        "final_url": URL,
        "http_status": 200,
        "fetched_at": "2026-08-15T00:00:01Z",
        "body": {"bytes": archive.stat().st_size,
                 "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
        "response_headers": {"bytes": headers.stat().st_size,
                             "sha256": hashlib.sha256(headers.read_bytes()).hexdigest()},
    }))
    ls_remote = input_dir / "git-ls-remote.txt"
    ls_remote.write_text(f"{COMMIT}\tHEAD\n")
    ls_stderr = input_dir / "git-ls-remote.stderr"
    ls_stderr.write_bytes(b"")
    started = input_dir / "git-ls-remote.started"
    completed = input_dir / "git-ls-remote.completed"
    started.write_text("2026-08-15T00:00:00Z\n")
    completed.write_text("2026-08-15T00:00:01Z\n")
    raw = input_dir / "git-ls-remote.raw"
    raw.write_text(f"{COMMIT}\tHEAD\n")
    ls_rc = input_dir / "git-ls-remote.rc"
    ls_rc.write_text("0\n")
    fetch_started = input_dir / "fetch.started"
    fetch_completed = input_dir / "fetch.completed"
    fetch_started.write_text("2026-08-15T00:00:00Z\n")
    fetch_completed.write_text("2026-08-15T00:00:01Z\n")
    fetch_rc = input_dir / "fetch.rc"
    fetch_rc.write_text("0\n")
    curl_stderr = input_dir / "snapshot.curl.stderr"
    curl_stderr.write_bytes(b"")
    return dict(root=root, artifact_root=tmp_path, archive=archive,
                headers=headers, curl_metrics=metrics, transport=transport,
                ls_remote=ls_remote, ls_remote_stderr=ls_stderr,
                ls_remote_started=started, ls_remote_completed=completed,
                ls_remote_raw=raw, ls_remote_rc=ls_rc,
                fetch_started=fetch_started, fetch_completed=fetch_completed,
                fetch_rc=fetch_rc, curl_stderr=curl_stderr,
                commit=COMMIT, requested_url=URL)


def test_seals_pinned_bilingual_xml_snapshot_with_global_artifacts(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    run = seal(**kwargs)
    assert run["status"] == "source_acquired_pending_source_native_projection"
    assert run["counts"]["xml_members"] == 2
    assert run["observed_language_paths"] == ["en", "fr"]
    receipt = json.loads((kwargs["root"] / "stages/original/receipt.json").read_text())
    assert receipt["records"] == 1
    assert len(receipt["artifacts"]) == 14
    assert all(a["uri"].startswith("artifact://source/input/")
               for a in receipt["artifacts"])


def test_rejects_unpinned_or_wrong_head_snapshot(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    kwargs["requested_url"] = "https://codeload.github.com/justicecanada/laws-lois-xml/tar.gz/main"
    with pytest.raises(ValueError, match="not pinned"):
        seal(**kwargs)
