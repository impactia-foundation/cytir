import hashlib
import json
from pathlib import Path

import cytir.sources.technical_manufacturer.host_consent as subject


def test_one_current_receipt_per_unique_host_and_resume(tmp_path: Path):
    frontier = tmp_path / "frontier.jsonl"
    frontier.write_text("".join(json.dumps({"url": url, "domain": host}) + "\n"
        for host, url in [("a.test", "https://a.test/p1"),
                          ("a.test", "https://a.test/p2"),
                          ("b.test", "https://b.test/p")]))
    seal = tmp_path / "frontier.RECEIPT.json"
    seal.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {"urls": 3,
            "sha256": hashlib.sha256(frontier.read_bytes()).hexdigest()}}}))
    calls = []
    def resolve(host):
        calls.append(host)
        return {"host": host, "status": "ok",
                "search_ok": host == "a.test", "checked_at": "now"}
    root = tmp_path / "consent"
    run = subject.acquire(frontier=frontier, frontier_receipt=seal,
        root=root, artifact_root=tmp_path, rate=100, workers=2,
        resolver=resolve)
    assert run["counts"] == {"hosts": 2, "affirmative_search_consent": 1,
                              "not_affirmative": 1}
    assert sorted(calls) == ["a.test", "b.test"]
    subject.acquire(frontier=frontier, frontier_receipt=seal,
        root=root, artifact_root=tmp_path, rate=100, workers=2,
        resolver=lambda _host: (_ for _ in ()).throw(AssertionError("re-resolved")))


def test_consent_fails_closed_unless_status_ok_and_search_ok_true(tmp_path: Path):
    """Fail-closed (unchanged by the port): only status == "ok" with search_ok is True counts as
    affirmative; a declined host, a missing flag, a truthy non-True value and an unreadable host
    all do not."""
    frontier = tmp_path / "frontier.jsonl"
    hosts = ["declined.test", "missing.test", "truthy.test", "unreadable.test", "yes.test"]
    frontier.write_text("".join(json.dumps({"url": f"https://{h}/p", "domain": h}) + "\n"
                                for h in hosts))
    seal = tmp_path / "frontier.RECEIPT.json"
    seal.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {"urls": len(hosts),
                                "sha256": hashlib.sha256(frontier.read_bytes()).hexdigest()}}}))
    answers = {
        "declined.test": {"status": "ok", "search_ok": False},
        "missing.test": {"status": "ok"},
        "truthy.test": {"status": "ok", "search_ok": "yes"},
        "unreadable.test": {"status": "unreadable", "search_ok": True},
        "yes.test": {"status": "ok", "search_ok": True},
    }
    run = subject.acquire(frontier=frontier, frontier_receipt=seal, root=tmp_path / "consent",
                          artifact_root=tmp_path, rate=100, workers=2,
                          resolver=lambda host: {"host": host, **answers[host]})
    assert run["counts"] == {"hosts": 5, "affirmative_search_consent": 1, "not_affirmative": 4}
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "consent" / "receipts").glob("*.json")]
    assert [r["host"] for r in receipts if r["affirmative_search_consent"]] == ["yes.test"]


def test_tech_product_verbs_are_routed_through_aw_cli():
    import subprocess
    import sys
    app = Path(__file__).resolve().parents[1]
    for verb in ("frontier", "consent", "pages"):
        out = subprocess.run([sys.executable, "-m", "cytir.cli", "source", "tech-product", verb, "--help"],
                             cwd=app, check=True, text=True, capture_output=True).stdout
        assert f"cyt source tech-product {verb}" in out
    bad = subprocess.run([sys.executable, "-m", "cytir.cli", "source", "tech-product", "nope"],
                         cwd=app, text=True, capture_output=True)
    assert bad.returncode == 2
