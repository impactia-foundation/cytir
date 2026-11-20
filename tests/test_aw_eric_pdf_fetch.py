"""ERIC PDF fetcher + frontier builder (task 2182), no network.

The first two tests are the recovered tests of the deleted fetcher (fetcher seam, resumability, migration,
one shared start gate). The rest drive the real consent-guarded client over `httpx.MockTransport`.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from cytir.consent.block import ConsentBlocked, ConsentRegistryError
from cytir.core import user_agent
from cytir.sources.eric import frontier as frontier_mod
from cytir.sources.eric import pdf_fetch as subject

PDF_HEADERS = b"HTTP/1.1 200\r\nContent-Type: application/pdf\r\n"


@pytest.fixture(autouse=True)
def _identity(monkeypatch, tmp_path):
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"hosts": {"declined.example": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(registry))
    monkeypatch.setattr(subject, "_validate_stage", lambda *a, **k: None)


def _frontier(tmp_path: Path, n: int) -> Path:
    frontier = tmp_path / "frontier"
    frontier.mkdir()
    rows = [{"eric_id": f"ED12345{i}", "pdf_url": f"https://files.eric.ed.gov/fulltext/ED12345{i}.pdf"}
            for i in range(n)]
    (frontier / "frontier.jsonl.gz").write_bytes(gzip.compress(
        b"".join((json.dumps(r) + "\n").encode() for r in rows), mtime=0))
    (frontier / "RUN.json").write_text(json.dumps({
        "status": "complete", "counts": {"official_pdf_urls": n}, "stages": {"frontier": {}}}))
    return frontier


def _ok(url, tmp, wait):
    wait()
    return b"%PDF-1.7\nbody", PDF_HEADERS, {"http_status": 200, "final_url": url, "fetched_at": "now"}


def test_pdf_tranche_is_resumable_and_counts_only_exact_pdf(tmp_path: Path):
    frontier = _frontier(tmp_path, 1)
    root = tmp_path / "pdf"
    result = subject.acquire(frontier_root=frontier, root=root, artifact_root=tmp_path,
                             offset=0, limit=1, rate=1, fetcher=_ok)
    assert result["counts"] == {"frontier": 1, "tranche": 1, "pdf_complete": 1,
                                "quarantine": 0, "migrated_terminal": 0}
    subject.acquire(frontier_root=frontier, root=root, artifact_root=tmp_path, offset=0, limit=1, rate=1,
                    fetcher=lambda *a: (_ for _ in ()).throw(AssertionError("refetched")))


def test_parallel_successor_migrates_terminals_without_refetch(tmp_path: Path):
    frontier = _frontier(tmp_path, 3)
    predecessor = tmp_path / "predecessor"
    subject.acquire(frontier_root=frontier, root=predecessor, artifact_root=tmp_path,
                    offset=0, limit=1, rate=1, workers=1, fetcher=_ok)
    starts, lock = [], threading.Lock()

    def fetch(url, tmp, wait):
        with lock:
            starts.append(url)
        return _ok(url, tmp, wait)

    successor = tmp_path / "successor"
    result = subject.acquire(frontier_root=frontier, root=successor, artifact_root=tmp_path,
                             offset=0, limit=3, rate=1, workers=3, predecessor_root=predecessor,
                             fetcher=fetch)
    assert result["counts"] == {"frontier": 3, "tranche": 3, "pdf_complete": 3,
                                "quarantine": 0, "migrated_terminal": 1}
    assert len(starts) == 2
    migration = [json.loads(x) for x in
                 (successor / "migration/predecessor-receipts.jsonl").read_text().splitlines()]
    assert migration[0]["source_receipt"]["uri"].startswith("artifact://predecessor/")


def _client_fetcher(handler):
    return subject._http_fetcher(subject._client(httpx.MockTransport(handler)))


def test_real_client_path_admits_a_pdf_and_quarantines_a_non_pdf(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["user-agent"] == "test-agent"
        if request.url.path.endswith("ED123450.pdf"):
            return httpx.Response(200, content=b"%PDF-1.7\nx", headers={"content-type": "application/pdf"})
        return httpx.Response(404, content=b"<html>no</html>", headers={"content-type": "text/html"})

    result = subject.acquire(frontier_root=_frontier(tmp_path, 2), root=tmp_path / "pdf",
                             artifact_root=tmp_path, offset=0, limit=2, rate=1,
                             fetcher=_client_fetcher(handler))
    assert result["counts"]["pdf_complete"] == 1 and result["counts"]["quarantine"] == 1
    receipt = json.loads((tmp_path / "pdf/receipts/ED123451.json").read_text())
    assert receipt["quarantine_reason"] == "non_200_or_non_pdf_response"


def test_two_bounded_attempts_then_network_failure_quarantine(tmp_path: Path):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("boom")

    result = subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf",
                             artifact_root=tmp_path, offset=0, limit=1, rate=1,
                             fetcher=_client_fetcher(handler))
    assert len(calls) == 2
    assert result["counts"]["quarantine"] == 1
    receipt = json.loads((tmp_path / "pdf/receipts/ED123450.json").read_text())
    assert receipt["quarantine_reason"] == "network_failure"


def _decline_pdf_host(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "declined.json"
    registry.write_text(json.dumps({"hosts": {"files.eric.ed.gov": {"state": "declined"}}}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(registry))


def test_declined_host_refuses_and_writes_no_receipt(tmp_path: Path, monkeypatch):
    _decline_pdf_host(tmp_path, monkeypatch)
    reached = []
    fetcher = _client_fetcher(lambda r: reached.append(r) or httpx.Response(200))
    with pytest.raises(ConsentBlocked):
        subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                        offset=0, limit=1, rate=1, fetcher=fetcher)
    assert reached == []
    assert not list((tmp_path / "pdf").glob("receipts/*"))
    assert not (tmp_path / "pdf/RUN.json").exists()


def test_declined_refusal_returns_promptly_without_draining_the_queue(tmp_path: Path, monkeypatch):
    """C1: six declined rows at 0.5/s used to idle ~10 s in the pool's exit; now the run ends at the first refusal."""
    _decline_pdf_host(tmp_path, monkeypatch)
    reached = []
    fetcher = _client_fetcher(lambda r: reached.append(r) or httpx.Response(200))
    started = time.monotonic()
    with pytest.raises(ConsentBlocked):
        subject.acquire(frontier_root=_frontier(tmp_path, 6), root=tmp_path / "pdf", artifact_root=tmp_path,
                        offset=0, limit=6, rate=.5, fetcher=fetcher)
    assert time.monotonic() - started < 2.5      # 6 rows at .5/s would take >= 10 s if the queue drained
    assert reached == []


def test_cross_host_redirect_is_quarantined_and_never_requested(tmp_path: Path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://api.ies.ed.gov/x.pdf"})

    result = subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                             offset=0, limit=1, rate=1, fetcher=_client_fetcher(handler))
    assert seen == ["files.eric.ed.gov"]
    assert result["counts"]["pdf_complete"] == 0 and result["counts"]["quarantine"] == 1
    receipt = json.loads((tmp_path / "pdf/receipts/ED123450.json").read_text())
    assert receipt["quarantine_reason"] == "cross_host_redirect"


def test_same_host_redirect_is_followed(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fulltext/ED123450.pdf":
            return httpx.Response(302, headers={"location": "/moved/ED123450.pdf"})
        return httpx.Response(200, content=b"%PDF-1.7\nx", headers={"content-type": "application/pdf"})

    result = subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                             offset=0, limit=1, rate=1, fetcher=_client_fetcher(handler))
    assert result["counts"]["pdf_complete"] == 1
    transport = json.loads((tmp_path / "pdf/transport/ED12/ED123450.json").read_text())
    assert transport["final_url"].endswith("/moved/ED123450.pdf")
    assert transport["http_metrics"]["num_redirects"] == 1


def test_frontier_row_off_the_pdf_host_is_quarantined_without_a_request(tmp_path: Path):
    frontier = _frontier(tmp_path, 1)
    row = {"eric_id": "ED123450", "pdf_url": "https://elsewhere.example/ED123450.pdf"}
    (frontier / "frontier.jsonl.gz").write_bytes(gzip.compress((json.dumps(row) + "\n").encode(), mtime=0))
    reached = []
    result = subject.acquire(frontier_root=frontier, root=tmp_path / "pdf", artifact_root=tmp_path,
                             offset=0, limit=1, rate=1,
                             fetcher=_client_fetcher(lambda r: reached.append(r) or httpx.Response(200)))
    assert reached == []
    receipt = json.loads((tmp_path / "pdf/receipts/ED123450.json").read_text())
    assert receipt["quarantine_reason"] == "frontier_host_mismatch"
    assert result["counts"]["quarantine"] == 1


def test_total_transfer_deadline_aborts_a_slow_body_as_network_failure(tmp_path: Path, monkeypatch):
    """N1: every read is quick, the whole transfer is not; the 600 s cap is wall-clock."""
    monkeypatch.setattr(subject, "TOTAL_SECONDS", 0.05)

    def slow():
        for _ in range(10):
            time.sleep(0.02)
            yield b"x" * 10

    result = subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                             offset=0, limit=1, rate=1,
                             fetcher=_client_fetcher(lambda r: httpx.Response(200, content=slow())))
    receipt = json.loads((tmp_path / "pdf/receipts/ED123450.json").read_text())
    assert receipt["quarantine_reason"] == "network_failure"
    assert result["counts"]["quarantine"] == 1


def test_disk_errors_stop_the_run_instead_of_quarantining(tmp_path: Path, monkeypatch):
    """N2: an OSError while publishing is not a network failure."""
    def boom(path, value):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(subject, "_publish", lambda path, value: boom(path, value) if "original" in str(path) else None)
    with pytest.raises(OSError):
        subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                        offset=0, limit=1, rate=1, fetcher=_ok)


def test_unreadable_registry_refuses(tmp_path: Path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(bad))
    with pytest.raises(ConsentRegistryError):
        subject.acquire(frontier_root=_frontier(tmp_path, 1), root=tmp_path / "pdf", artifact_root=tmp_path,
                        offset=0, limit=1, rate=1,
                        fetcher=_client_fetcher(lambda r: httpx.Response(200)))


def test_bounds_are_unchanged(tmp_path: Path):
    for kw in ({"limit": 10001}, {"rate": 1.5}, {"workers": 9}, {"offset": -1}):
        args = {"offset": 0, "limit": 1, "rate": .5, "workers": 1, **kw}
        with pytest.raises(ValueError):
            subject.acquire(frontier_root=tmp_path, root=tmp_path, artifact_root=tmp_path, **args)


def test_frontier_builder_is_deterministic_and_conserves_ids(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(frontier_mod, "_validate_stage", lambda *a, **k: None)
    inv = tmp_path / "inv"
    (inv / "pages").mkdir(parents=True)
    (inv / "stages/original").mkdir(parents=True)
    (inv / "stages/original/receipt.portable.v2.json").write_text("{}")
    (inv / "pages/000000000.body.json").write_text(json.dumps({"response": {"docs": [
        {"id": "ED000002", "e_fulltextauth": 1}, {"id": "ED000001", "e_fulltextauth": 1}]}}))
    (inv / "RUN.portable.v2.json").write_text(json.dumps({
        "schema": "aw.eric-api-inventory-run.v1", "status": "complete",
        "counts": {"unique_document_ids": 2}, "stages": {"original": {}}}))
    run = frontier_mod.build(inventory_root=inv, root=tmp_path / "fr", artifact_root=tmp_path)
    assert run["counts"]["official_pdf_urls"] == 2
    rows = [json.loads(x) for x in gzip.open(tmp_path / "fr/frontier.jsonl.gz", "rt")]
    assert [r["eric_id"] for r in rows] == ["ED000001", "ED000002"]
    assert rows[0]["pdf_url"].endswith("/fulltext/ED000001.pdf")
    with pytest.raises(ValueError):
        frontier_mod.build(inventory_root=inv, root=tmp_path / "fr", artifact_root=tmp_path)
