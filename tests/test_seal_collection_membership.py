"""Task 452: a mirror's baseline membership seals as an exact, portable set."""
from __future__ import annotations

import urllib.request
import hashlib
import json
import uuid

import pytest

from cytir.index import seal_collection_membership as subject


def ids(count: int) -> list[str]:
    return sorted(str(uuid.UUID(int=index)) for index in range(count))


def pager(rows: list[str], *, pages: int = 3):
    """A scroll that hands `rows` back in `pages` slices, in the given order."""
    size = max(1, -(-len(rows) // pages))
    chunks = [rows[start:start + size] for start in range(0, len(rows), size)]

    def scroll(offset, limit):
        index = 0 if offset is None else int(offset)
        page = chunks[index]
        return page, (str(index + 1) if index + 1 < len(chunks) else None)
    return scroll


def expected_digest(rows: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{row}\n" for row in sorted(rows)).encode()).hexdigest()


def run(tmp_path, rows, *, role="twin", expect=None, observed=None):
    return subject.seal(
        artifact_root=tmp_path, role=role, url="http://127.0.0.1:6444",
        collection="aw_e3", expect_points=len(rows) if expect is None else expect,
        scroll=pager(rows), receipt_path=tmp_path / f"{role}.json",
        evidence_path=tmp_path / f"{role}.evidence.json", batch=4,
        pace_seconds=0, observed_points=(len(rows) if observed is None
                                         else observed))


def test_the_digest_is_exactly_sort_pipe_sha256sum(tmp_path):
    rows = ids(10)
    result = run(tmp_path, rows)
    assert result["logical_membership_sha256"] == expected_digest(rows)
    assert result["rows"] == result["points"] == 10
    receipt = json.loads((tmp_path / "twin.json").read_text())
    assert receipt["schema"] == subject.RECEIPT_SCHEMA
    assert receipt["status"] == "PASS" and receipt["role"] == "twin"
    assert receipt["collection"] == "aw_e3" and receipt["points"] == 10
    assert receipt["logical_membership_sha256"] == expected_digest(rows)
    evidence = json.loads((tmp_path / "twin.evidence.json").read_text())
    assert evidence["schema"] == subject.EVIDENCE_SCHEMA
    assert evidence["points"] == 10
    assert evidence["checkpoints"][-1]["cumulative_rows"] == 10
    assert evidence["checkpoints"][-1]["running_sha256"] == expected_digest(rows)


def test_two_mirrors_holding_the_same_set_agree_and_one_that_differs_does_not(
        tmp_path):
    rows = ids(10)
    twin = run(tmp_path, rows, role="twin")
    prod = run(tmp_path, list(rows), role="prod")
    assert twin["logical_membership_sha256"] == prod["logical_membership_sha256"]
    # One different point id — same count — is a different membership.
    drifted = sorted(rows[:-1] + [str(uuid.UUID(int=999))])
    other = subject.seal(
        artifact_root=tmp_path, role="prod", url="http://x", collection="aw_prod",
        expect_points=10, scroll=pager(drifted),
        receipt_path=tmp_path / "drift.json",
        evidence_path=tmp_path / "drift.evidence.json", batch=4,
        pace_seconds=0, observed_points=10)
    assert other["logical_membership_sha256"] != twin["logical_membership_sha256"]


def test_an_unordered_scroll_aborts_rather_than_digesting_the_wrong_order(
        tmp_path):
    rows = ids(10)
    with pytest.raises(subject.MembershipError, match="strictly ascending"):
        subject.seal(
            artifact_root=tmp_path, role="twin", url="http://x",
            collection="aw_e3", expect_points=10,
            scroll=pager(list(reversed(rows))),
            receipt_path=tmp_path / "t.json",
            evidence_path=tmp_path / "t.evidence.json", batch=4,
            pace_seconds=0, observed_points=10)
    assert not (tmp_path / "t.json").exists()


def test_a_duplicate_point_id_is_not_a_set(tmp_path):
    rows = ids(5)
    with pytest.raises(subject.MembershipError, match="strictly ascending"):
        subject.seal(
            artifact_root=tmp_path, role="twin", url="http://x",
            collection="aw_e3", expect_points=6,
            scroll=pager(sorted(rows + [rows[2]])),
            receipt_path=tmp_path / "t.json",
            evidence_path=tmp_path / "t.evidence.json", batch=4,
            pace_seconds=0, observed_points=6)


def test_a_short_enumeration_refuses_instead_of_sealing_a_partial_set(tmp_path):
    with pytest.raises(subject.MembershipError, match="membership is not exact"):
        run(tmp_path, ids(10), expect=11, observed=11)


def test_a_count_that_moved_under_the_seal_refuses(tmp_path):
    with pytest.raises(subject.MembershipError, match="expected"):
        run(tmp_path, ids(10), expect=10, observed=11)


def test_the_receipt_is_immutable(tmp_path):
    rows = ids(10)
    run(tmp_path, rows)
    run(tmp_path, rows)  # byte-identical re-seal is a no-op
    with pytest.raises(subject.MembershipError, match="immutable"):
        subject.seal(
            artifact_root=tmp_path, role="twin", url="http://other",
            collection="aw_e3", expect_points=10, scroll=pager(rows),
            receipt_path=tmp_path / "twin.json",
            evidence_path=tmp_path / "twin.evidence.json", batch=4,
            pace_seconds=0, observed_points=10)


def test_the_receipt_must_live_inside_the_portable_artifact_root(tmp_path):
    outside = tmp_path.parent / "outside.json"
    with pytest.raises(subject.MembershipError, match="outside the portable"):
        subject.seal(
            artifact_root=tmp_path / "root", role="twin", url="http://x",
            collection="aw_e3", expect_points=4, scroll=pager(ids(4)),
            receipt_path=outside, evidence_path=outside.with_suffix(".e.json"),
            batch=4, pace_seconds=0, observed_points=4)


def test_the_membership_seal_refuses_a_prod_scroll_without_a_reason(monkeypatch):
    from cytir.index import prod_guard as PG
    SM = subject
    PROD = "http://%s:%d" % ("127.0.0.1", PG.PROD_PORT)
    TWIN = "http://%s:%d" % ("127.0.0.1", PG.PROD_PORT + 123)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("a refused call must not reach the network"))
    with pytest.raises(SM.MembershipError, match="scroll on prod"):
        SM.rest_scroll(PROD, "aw_e9", 5)(None, 10)

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"status": "ok", "result": {"points": [{"id": "a"}]}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())
    PG.ALLOWED.clear()
    ids, _ = SM.rest_scroll(PROD, "aw_e9", 5, prod_scroll_reason="operator baseline, task 9")(None, 10)
    assert ids == ["a"] and PG.ALLOWED[-1]["reason"] == "operator baseline, task 9"
    ids, _ = SM.rest_scroll(TWIN, "aw_e8", 5)(None, 10)               # the twin needs no reason
    assert ids == ["a"]
