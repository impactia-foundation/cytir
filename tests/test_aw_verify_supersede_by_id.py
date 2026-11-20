"""The by-id supersede verifier (task 1050): a small fixture, a fake Qdrant that answers ONLY the
retrieve endpoint (a scroll or a search raises), through the real `cytir.index.point_delete` reader."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cytir.index import verify_supersede_by_id as V  # noqa: E402

WRITTEN = [f"00000000-0000-5000-8000-{i:012d}" for i in range(1, 7)]       # 6 rows, 2 with a window
REMOVED = [f"11111111-0000-5000-8000-{i:012d}" for i in range(1, 5)]       # 4 victims
TWIN, PROD = "http://127.0.0.1:6486", "http://127.0.0.1:6334"


def _row(i, *, window=None):
    p = {"url": f"https://www.fedlex.admin.ch/eli/cc/x/de#art_{i}", "license": "public-domain"}
    if window:
        p["window"] = window
    return {"payload": p, "vector": {"dense": [0.1 * i, 0.2, 0.3]}}


def _store():
    s = {}
    for n, pid in enumerate(WRITTEN):
        w = {"index": n % 2 + 1, "count": 2} if n < 2 else None
        s[pid] = _row(n, window=w)
    return s


class FakeQdrant:
    """{(url, collection): {id: row}}; answers `POST /collections/<c>/points` and nothing else."""

    def __init__(self, data):
        self.data, self.calls = data, []

    def __call__(self, url, path, body=None, method=None, **_kw):
        self.calls.append(path)
        parts = path.strip("/").split("/")
        assert parts[0] == "collections" and parts[2:] == ["points"], f"not a retrieve: {path}"
        coll = self.data[(url, parts[1])]
        out = []
        for i in body["ids"]:
            if i in coll:
                r = {"id": i}
                if body.get("with_payload"):
                    r["payload"] = coll[i]["payload"]
                wv = body.get("with_vector")
                if wv:
                    r["vector"] = {k: v for k, v in coll[i]["vector"].items() if k in wv}
                out.append(r)
        return out


def _receipt(tmp_path, **over):
    doc = {"schema": V.SCHEMA, "pack_id": "t", "consolidation": "20250707", "license": "public-domain",
           "window_rows": 2, "distinct_urls": 6, "window_urls": 2,
           "written": WRITTEN, "removed": REMOVED, **over}
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _run(monkeypatch, tmp_path, twin, prod, capsys, extra=(), receipt=None):
    fake = FakeQdrant({(TWIN, "aw_e10_twin"): twin, (PROD, "aw_prod"): prod})
    monkeypatch.setattr(V.PD, "_api", fake)
    rc = V.main(["--receipt", receipt or _receipt(tmp_path),
                 "--target", f"twin={TWIN},aw_e10_twin", "--target", f"prod={PROD},aw_prod", *extra])
    return rc, capsys.readouterr().out, fake


def test_a_correct_apply_on_both_targets_passes_with_one_line_each(monkeypatch, tmp_path, capsys):
    rc, out, fake = _run(monkeypatch, tmp_path, _store(), _store(), capsys)
    lines = out.strip().splitlines()
    assert rc == 0
    assert lines[0].startswith("PASS twin ") and "written=6/6" in lines[0] and "window=2/2" in lines[0]
    assert lines[1].startswith("PASS prod ")
    assert lines[2].startswith("PASS pair twin==prod") and lines[-1] == "RESULT PASS"
    assert all(p.endswith("/points") for p in fake.calls)       # retrieve only: never a scroll


def test_a_victim_that_is_still_served_fails_that_target(monkeypatch, tmp_path, capsys):
    prod = _store()
    prod[REMOVED[0]] = _row(9)
    rc, out, _ = _run(monkeypatch, tmp_path, _store(), prod, capsys)
    assert rc == 1
    assert "PASS twin" in out and "FAIL prod" in out and "1 removed id(s) still present" in out


def test_a_missing_written_id_and_a_wrong_window_count_fail(monkeypatch, tmp_path, capsys):
    twin = _store()
    del twin[WRITTEN[5]]
    rc, out, _ = _run(monkeypatch, tmp_path, twin, _store(), capsys)
    assert rc == 1 and "1 written id(s) absent" in out
    rc, out, _ = _run(monkeypatch, tmp_path, _store(), _store(), capsys,
                      receipt=_receipt(tmp_path, window_rows=28))
    assert rc == 1 and "window rows 2, expected 28" in out


def test_equal_ids_with_a_different_payload_or_vector_fail_the_pair(monkeypatch, tmp_path, capsys):
    prod = _store()
    prod[WRITTEN[3]]["payload"]["title"] = "stale"
    prod[WRITTEN[4]]["vector"]["dense"][0] = 9.9
    rc, out, _ = _run(monkeypatch, tmp_path, _store(), prod, capsys)
    assert rc == 1
    assert "PASS twin" in out and "PASS prod" in out        # each target is fine alone
    assert "FAIL pair twin==prod" in out and "1 payload digest(s) differ" in out
    assert "1 vector digest(s) differ" in out


def test_a_license_other_than_the_receipts_fails(monkeypatch, tmp_path, capsys):
    twin = _store()
    twin[WRITTEN[2]]["payload"]["license"] = "cc-by"
    rc, out, _ = _run(monkeypatch, tmp_path, twin, _store(), capsys)
    assert rc == 1 and "1 row(s) without license 'public-domain'" in out


def test_a_collection_that_is_not_a_declared_live_name_is_refused_unless_scratch_is_allowed(
        monkeypatch, tmp_path, capsys):
    fake = FakeQdrant({(TWIN, "aw_c1040"): _store()})
    monkeypatch.setattr(V.PD, "_api", fake)
    args = ["--receipt", _receipt(tmp_path), "--target", f"clone={TWIN},aw_c1040"]
    assert V.main(args) == 2
    out = capsys.readouterr().out
    assert out.startswith("REFUSED") and "aw_c1040" in out and fake.calls == []     # nothing was read
    assert V.main([*args, "--allow-scratch"]) == 0


def test_an_unreadable_target_fails_and_never_passes(monkeypatch, tmp_path, capsys):
    def down(*_a, **_k):
        raise OSError("connection refused")
    monkeypatch.setattr(V.PD, "_api", down)
    rc = V.main(["--receipt", _receipt(tmp_path), "--target", f"twin={TWIN},aw_e10_twin"])
    assert rc == 1 and "FAIL twin" in capsys.readouterr().out


def test_the_fake_server_refuses_a_scroll(monkeypatch):
    with pytest.raises(AssertionError):
        FakeQdrant({})(TWIN, "/collections/aw_e10_twin/points/scroll", {})


def test_pack_records_must_carry_exactly_the_receipts_ids_and_consolidation(tmp_path, capsys):
    rows = [{"point_id": i, "source_version": "20250707"} for i in WRITTEN]
    f = tmp_path / "records.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    args = ["--receipt", _receipt(tmp_path), "--pack-records", str(f)]
    assert V.main(args) == 0
    assert f"ids_sha256={V.ids_sha256(WRITTEN)}" in capsys.readouterr().out
    rows[0]["point_id"] = "99999999-0000-5000-8000-000000000001"
    rows[1]["source_version"] = "20260101"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert V.main(args) == 1
    out = capsys.readouterr().out
    assert "1 new" in out and "1 missing" in out and "consolidation" in out


def test_a_bad_receipt_is_refused(tmp_path, capsys):
    p = _receipt(tmp_path, removed=[WRITTEN[0]])
    assert V.main(["--receipt", p, "--target", f"twin={TWIN},aw_e10_twin"]) == 2
    assert "both written and removed" in capsys.readouterr().out


def test_distinct_url_and_windowed_url_counts_are_checked_when_the_receipt_states_them(
        monkeypatch, tmp_path, capsys):
    rc, out, _ = _run(monkeypatch, tmp_path, _store(), _store(), capsys,
                      receipt=_receipt(tmp_path, distinct_urls=5, window_urls=1))
    assert rc == 1 and "distinct urls 6, expected 5" in out and "windowed urls 2, expected 1" in out
