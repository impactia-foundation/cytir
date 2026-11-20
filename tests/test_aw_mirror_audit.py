"""G3, `cyt ops mirror-audit`: twin against prod by CONTENT for a declared class of ids (task 837).

The comparator is task 833's (`cytir.index.mirror_compare`); this module pins the VERB around it:
the buckets, the declarations, the exit codes (0 / 3 / 4), the receipt, and that prod is read only
by id, paced and budgeted, and never in a replay.

THE ROWS are the six real rows of `fixtures/mirror_guards/rows.v1.json` (task 833): three
`prod_only_old` (the twin rewrote them in place; prod holds the 09-22 row) and three `equal`.
Prod is a fake Qdrant that stores what a real one does (dense unit-normalised, fp16, JSON payload).

What each group pins:
  buckets      EQUAL / DIFF-declared / DIFF-undeclared / missing-twin / missing-prod / missing-both
  declarations a declared id may differ ONLY on its set's keys; a new key is a finding; an expired
               set declares nothing; the tracked file is valid; a changed source is a refusal
  exit codes   0 only with >= 1 id compared and no undeclared finding; 3 findings; 4 refused
  refusals     replay (zero prod calls), budget, read error, empty class, nothing compared, a twin
               that is prod, a receipt that cannot be written
  read-only    no PUT, no DELETE, no scroll; at most 500 ids per call; the call count is the receipt's
"""
from __future__ import annotations

import copy
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import types
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from cytir import cli
from cytir.index import mirror_compare as MC
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.ops import mirror_audit as MA

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mirror_guards" / "rows.v1.json"
TRACKED = HERE.parent / "data" / "mirror-audit-declared.v1.json"
PROD = "http://127.0.0.1:%d" % PG.PROD_PORT
TWIN = "http://twin.invalid:16457"          # never resolved: urlopen is the fake below
#: the tracked declarations are operator data: the public cut does not ship them, its tests skip these
needs_tracked = pytest.mark.skipif(not TRACKED.exists(), reason="data/mirror-audit-declared.v1.json is not in the public cut")
NOW = dt.datetime(2026, 10, 10, 3, 50, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def rows():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("AW_PROD_READ_BUDGET", raising=False)
    monkeypatch.delenv("AW_PROD_URL", raising=False)
    monkeypatch.delenv("AW_REPLAY", raising=False)
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0)
    PG.reset()


# ------------------------------------------------------------------------- a fake Qdrant
class _Resp:
    def __init__(self, doc):
        self._b = json.dumps(doc).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


class FakeQdrant:
    """`urllib.request.urlopen` stand-in. Stores dense unit-normalised in fp16, payload through a
    JSON round trip. Every request is logged; `fail_prod` raises a URLError on a prod POST."""

    def __init__(self):
        self.store: dict[tuple[str, str], dict[str, dict]] = {}
        self.log: list[tuple[str, str, str, dict]] = []
        self.fail_prod = False

    def put(self, base, collection, rows: dict):
        coll = self.store.setdefault((base, collection), {})
        for pid, r in rows.items():
            vec = {}
            for name, v in (r.get("vector") or {}).items():
                a = np.asarray(v, dtype=np.float64)
                n = np.linalg.norm(a)
                vec[name] = [float(x) for x in (a / n if n else a).astype(np.float16)]
            coll[pid] = {"payload": json.loads(json.dumps(r["payload"])), "vector": vec}

    def calls(self, base, method="POST"):
        return [e for e in self.log if e[0] == base and e[1] == method]

    def __call__(self, req, timeout=None):
        full, method = req.full_url, req.get_method()
        body = json.loads(req.data) if req.data else None
        p = urllib.parse.urlparse(full)
        base, path = f"{p.scheme}://{p.netloc}", p.path
        self.log.append((base, method, path, body or {}))
        if path == "/aliases":
            al = [{"alias_name": "aw_prod", "collection_name": "aw_e9"}] if base == PROD else []
            return _Resp({"status": "ok", "result": {"aliases": al}})
        parts = path.strip("/").split("/")
        if method == "POST" and parts[2:] == ["points"]:
            if self.fail_prod and base == PROD:
                raise urllib.error.URLError("connection reset")
            coll = self.store.setdefault((base, parts[1]), {})
            out = []
            for pid in body["ids"]:
                if pid in coll:
                    out.append({"id": pid, "payload": copy.deepcopy(coll[pid]["payload"]),
                                "vector": {n: coll[pid]["vector"][n] for n in body["with_vector"]
                                           if n in coll[pid]["vector"]}})
            return _Resp({"status": "ok", "result": out})
        raise AssertionError(f"unexpected request {method} {full}")


@pytest.fixture()
def qd(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# ------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def cmp_e9():
    return MC.Comparator("aw_e9")


def _stamped(entry, cmp_e9, source="twin"):
    """What the drain writes to prod for this row: the source row in prod's stored form, stamped."""
    src = entry[source]
    return {"payload": cmp_e9.target_form(src["payload"]), "vector": copy.deepcopy(src["vector"])}


def _unstamped(entry, source):
    """What a flip-carried prod row holds: prod's form WITHOUT the BM25 epoch stamp."""
    src = entry[source]
    return {"payload": PP.Projector.for_collection("aw_e9").project(src["payload"]),
            "vector": copy.deepcopy(src["vector"])}


def _settings(tmp_path):
    return types.SimpleNamespace(work_root=str(tmp_path / "work"), queue_dir=str(tmp_path / "queue"),
                                 news_work=str(tmp_path / "news"))


def _ids_file(tmp_path, ids, name="ids.txt"):
    p = tmp_path / name
    p.write_text("\n".join(ids) + "\n")
    return str(p)


def _run(tmp_path, argv, *, settings=None, **kw):
    lines: list[str] = []
    s = settings or _settings(tmp_path)
    if not (tmp_path / "declared.json").exists():
        _write_declared(tmp_path, _declared_doc())
    base = ["--twin-url", TWIN, "--twin-collection", "aw_e8", "--prod-url", PROD,
            "--prod-collection", "aw_e9", "--receipt-dir", str(tmp_path / "receipts"),
            "--artifact-dir", str(tmp_path / "artifacts"), "--declared", str(tmp_path / "declared.json")]
    if "--declared" in argv:
        base = base[:-2]
    kw.setdefault("sleep", lambda secs: None)          # the recheck wait is not waited for
    code = MA.run(base + argv, now=lambda: NOW, settings=s, out=lines.append, **kw)
    return code, lines


def _receipt(tmp_path):
    files = sorted((tmp_path / "receipts").glob("*.json"))
    assert files, "every run writes a receipt"
    return json.loads(files[-1].read_text())


def _declared_doc(sets=(), global_keys=()):
    return {"schema": MA.DECLARED_SCHEMA, "not_compared": [{"what": "bm25", "why": "pruned on prod"}],
            "global_allowed_keys": [{"key": k, "reason": "r"} for k in global_keys], "sets": list(sets)}


def _write_declared(tmp_path, doc):
    (tmp_path / "declared.json").write_text(json.dumps(doc))


def _results_set(tmp_path, ids, *, allowed, set_id="s1", expires=None, name="res.jsonl.gz", **extra):
    path = tmp_path / name
    with gzip.open(path, "wt") as fh:
        for i in ids:
            fh.write(json.dumps({"id": i, "class": "PROD-ONLY-OLD"}) + "\n")
    s = {"id": set_id, "reason": "a reason", "ruling": "a ruling", "allowed_keys": sorted(allowed),
         "expires": expires, "source": {"kind": "results-jsonl-gz", "path": str(path),
                                         "classes": ["PROD-ONLY-OLD"], "expected_ids": len(ids)}}
    s.update(extra)
    return s


def _world(qd, cmp_e9, rows):
    """Twin holds the new rows of all six; prod holds the snapshot row for the stale three and the
    twin's (stamped) row for the equal three."""
    old, eq = rows["prod_only_old"], rows["equal"]
    qd.put(TWIN, "aw_e8", {e["id"]: e["twin"] for e in (*old, *eq)})
    qd.put(PROD, "aw_e9", {e["id"]: _stamped(e, cmp_e9, "copy") for e in old})
    qd.put(PROD, "aw_e9", {e["id"]: _stamped(e, cmp_e9, "twin") for e in eq})
    return [e["id"] for e in old], [e["id"] for e in eq]


# ===================================================================== buckets and exit codes
def test_equal_rows_pass_with_exit_0_and_one_prod_call(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq)])
    assert code == MA.EXIT_OK
    r = _receipt(tmp_path)
    assert r["verdict"] == "PASS" and r["exit_code"] == 0
    assert r["counts"]["EQUAL"] == 3 and r["undeclared"] == 0 and r["compared"] == 3
    assert r["prod_calls"] == 1 and len(qd.calls(PROD)) == 1 and r["complete"] is True
    assert any("PASS" in ln for ln in lines)


def test_stale_rows_are_undeclared_diffs_exit_3_with_the_keys(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old + eq)])
    assert code == MA.EXIT_DIVERGED
    r = _receipt(tmp_path)
    assert r["verdict"] == "DIVERGED"
    assert r["counts"]["DIFF-undeclared"] == 3 and r["counts"]["EQUAL"] == 3
    assert {e["id"] for e in r["examples"]} == set(old)
    keys = {k for e in r["examples"] for k in e["keys"]}
    assert {"last_seen", "dense"} <= keys and r["by_key"]["last_seen"] >= 1
    assert any("UNDECLARED" in ln and old[0] in ln for ln in lines)


def test_a_declared_divergence_is_counted_and_passes(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    probe, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    seen = {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]}
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, old, allowed=seen)]))
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old + eq)], settings=_settings(tmp_path))
    assert code == MA.EXIT_OK
    r = _receipt(tmp_path)
    assert r["counts"]["DIFF-declared"] == 3 and r["counts"]["DIFF-undeclared"] == 0
    assert r["declared_sets"]["s1"] == {"DIFF-declared": 3}


def test_a_new_key_on_a_declared_id_is_still_a_finding(tmp_path, qd, rows, cmp_e9):
    old, _ = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    seen = {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]}
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, old, allowed=seen - {"dense"})]))
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    assert code == MA.EXIT_DIVERGED
    r = _receipt(tmp_path)
    assert r["counts"]["DIFF-undeclared"] >= 1
    assert all("dense" in e["undeclared_keys"] for e in r["examples"] if "dense" in e["keys"])


def test_global_allowed_keys_and_expiry(tmp_path, qd, rows, cmp_e9):
    old, _ = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    seen = {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]}
    _write_declared(tmp_path, _declared_doc(global_keys=sorted(seen)))
    assert _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])[0] == MA.EXIT_OK
    # an expired set declares nothing: the same ids are findings again, and the receipt says why
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, old, allowed=seen, expires="2026-10-09")]))
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    assert code == MA.EXIT_DIVERGED
    assert _receipt(tmp_path)["expired_declarations"] == ["s1"]
    assert any("EXPIRED" in ln for ln in lines)


def test_missing_on_one_side_is_a_finding_missing_on_both_is_not(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    only_twin, only_prod, nowhere = "11111111-1111-5111-8111-111111111111", \
        "22222222-2222-5222-8222-222222222222", "33333333-3333-5333-8333-333333333333"
    qd.put(TWIN, "aw_e8", {only_twin: rows["equal"][0]["twin"]})
    qd.put(PROD, "aw_e9", {only_prod: _stamped(rows["equal"][0], cmp_e9)})
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq + [only_twin, only_prod, nowhere])])
    assert code == MA.EXIT_DIVERGED
    c = _receipt(tmp_path)["counts"]
    assert (c["missing-prod"], c["missing-twin"], c["missing-both"], c["EQUAL"]) == (1, 1, 1, 3)
    ex = {e["id"]: e["keys"] for e in _receipt(tmp_path)["examples"]}
    assert ex[only_twin] == ["<absent-on-prod>"] and ex[only_prod] == ["<absent-on-twin>"]


def test_allow_missing_declares_an_absence(tmp_path, qd, rows, cmp_e9):
    _world(qd, cmp_e9, rows)
    gone = "11111111-1111-5111-8111-111111111111"
    qd.put(TWIN, "aw_e8", {gone: rows["equal"][0]["twin"]})
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, [gone], allowed=["last_seen"],
                                                           allow_missing=True)]))
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, [gone] + [rows["equal"][0]["id"]])])
    assert code == MA.EXIT_OK and _receipt(tmp_path)["counts"]["DIFF-declared"] == 1


def test_a_flip_carried_unstamped_row_needs_the_flag(tmp_path, qd, rows):
    eq = rows["equal"]
    qd.put(TWIN, "aw_e8", {e["id"]: e["twin"] for e in eq})
    qd.put(PROD, "aw_e9", {e["id"]: _unstamped(e, "twin") for e in eq})
    ids = _ids_file(tmp_path, [e["id"] for e in eq])
    code, _ = _run(tmp_path, ["--class", "ids", ids])
    assert code == MA.EXIT_DIVERGED
    assert {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]} == {"rule_epochs"}
    code, _ = _run(tmp_path, ["--class", "ids", ids, "--allow-unstamped-rule-epochs"])
    assert code == MA.EXIT_OK and _receipt(tmp_path)["counts"]["EQUAL"] == 3


# ===================================================================== the classes
def _artifact(tmp_path, pack, *, at, ids, roles=("twin", "prod")):
    d = tmp_path / "artifacts"
    d.mkdir(exist_ok=True)
    doc = {"schema": MA.INPLACE_SCHEMA, "pack_id": pack, "manifest_sha": "x"}
    for role in roles:
        doc[role] = {"collection": "aw_e8", "at": at, "count": len(ids), "ids": sorted(ids)}
    (d / f"{pack}-inplace-ids.json").write_text(json.dumps(doc))


def test_inplace_classes_select_by_date_and_skip_the_unsettled(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    _artifact(tmp_path, "p-old", at="2026-10-07T10:00:00Z", ids=[eq[0]])
    _artifact(tmp_path, "p-yesterday", at="2026-10-09T10:00:00Z", ids=[eq[1]])
    _artifact(tmp_path, "p-today", at="2026-10-10T00:10:00Z", ids=[eq[2]])
    _artifact(tmp_path, "p-fresh", at="2026-10-10T03:30:00Z", ids=[old[0]])    # 20 min before NOW
    code, _ = _run(tmp_path, ["--class", "inplace-since", "2026-10-09"])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_OK and r["checked"] == 2           # yesterday + today, not old, not fresh
    assert r["class"]["inplace"][0]["deferred_unsettled"] == 1
    code, _ = _run(tmp_path, ["--class", "inplace-today"])
    assert _receipt(tmp_path)["checked"] == 1
    code, _ = _run(tmp_path, ["--class", "inplace-since", "2026-10-07", "--settle-min", "0"])
    assert _receipt(tmp_path)["checked"] == 4                  # old + yesterday + today + fresh
    assert code == MA.EXIT_DIVERGED                            # p-fresh holds a stale row


def test_declared_class_reads_every_declared_set(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    seen = {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]}
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, old, allowed=seen)]))
    code, _ = _run(tmp_path, ["--class", "declared"])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_OK and r["checked"] == 3 and r["counts"]["DIFF-declared"] == 3


def test_classes_are_unioned_and_ordered_without_duplicates(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    f = _ids_file(tmp_path, eq + [eq[0]])
    code, _ = _run(tmp_path, ["--class", "ids", f, "--class", "ids", f])
    assert code == MA.EXIT_OK and _receipt(tmp_path)["checked"] == 3


def test_a_json_id_list_is_read(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(eq))
    assert _run(tmp_path, ["--class", "ids", str(p)])[0] == MA.EXIT_OK


# ===================================================================== refusals (exit 4)
def test_a_replay_sends_no_query_to_prod_and_claims_no_result(tmp_path, qd, rows, cmp_e9, monkeypatch):
    _, eq = _world(qd, cmp_e9, rows)
    monkeypatch.setenv("AW_REPLAY", "1")
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq)])
    assert code == MA.EXIT_REFUSED
    assert qd.calls(PROD) == [] and qd.calls(TWIN) == []        # not even the twin: nothing is claimed
    r = _receipt(tmp_path)
    assert r["verdict"] == "REFUSED" and "replay" in r["refusal"] and r["counts"] == {}
    assert not any("PASS" in ln for ln in lines)


def test_the_prod_budget_refuses_the_run(tmp_path, qd, rows, cmp_e9, monkeypatch):
    _world(qd, cmp_e9, rows)
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "2")
    PG.check(PROD, "POST", "/collections/aw_e9/points")          # the process already spent one call
    ids = [f"{i:08x}-0000-5000-8000-000000000000" for i in range(501)]     # 2 batches of <= 500
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, ids)])
    assert code == MA.EXIT_REFUSED
    r = _receipt(tmp_path)
    assert "prod read refused" in r["refusal"] and "budget" in r["refusal"] and "read error" not in r["refusal"]
    assert r["prod_calls"] == 2 and len(qd.calls(PROD)) == 1
    assert r["counts"] == {}                                    # a refused run claims no counts


def test_a_class_over_the_budget_is_refused_before_any_call(tmp_path, qd, rows, cmp_e9, monkeypatch):
    _world(qd, cmp_e9, rows)
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "1")
    ids = [f"{i:08x}-0000-5000-8000-000000000000" for i in range(501)]
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, ids)])
    assert code == MA.EXIT_REFUSED and qd.log == []
    assert "no call was made" in _receipt(tmp_path)["refusal"]
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, ids), "--max-ids", "500"])
    r = _receipt(tmp_path)                                    # the cap makes it fit: one call, then "nothing compared"
    assert "no call was made" not in (r["refusal"] or "") and r["prod_calls"] == 1 and r["skipped"] == 1


def test_a_prod_read_error_is_a_refusal_never_a_pass(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    qd.fail_prod = True
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq)])
    assert code == MA.EXIT_REFUSED and "read error" in _receipt(tmp_path)["refusal"]


def test_an_empty_or_unknown_class_is_a_refusal(tmp_path, qd):
    assert _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, [])])[0] == MA.EXIT_REFUSED
    assert "empty" in _receipt(tmp_path)["refusal"]
    assert _run(tmp_path, ["--class", "inplace-today"])[0] == MA.EXIT_REFUSED      # no artifact dir
    assert _run(tmp_path, ["--class", "nonsense"])[0] == MA.EXIT_REFUSED
    assert _run(tmp_path, [])[0] == MA.EXIT_REFUSED
    assert qd.log == []


def test_nothing_compared_is_a_refusal(tmp_path, qd):
    ids = ["33333333-3333-5333-8333-333333333333"]
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, ids)])
    assert code == MA.EXIT_REFUSED and "nothing was compared" in _receipt(tmp_path)["refusal"]
    assert _receipt(tmp_path)["compared"] == 0


def test_a_twin_that_is_prod_is_refused(tmp_path, qd, rows):
    lines: list[str] = []
    code = MA.run(["--twin-url", PROD, "--twin-collection", "aw_e9", "--prod-url", PROD,
                   "--prod-collection", "aw_e9", "--receipt-dir", str(tmp_path / "r"),
                   "--class", "ids", _ids_file(tmp_path, ["x"])],
                  now=lambda: NOW, settings=_settings(tmp_path), out=lines.append)
    assert code == MA.EXIT_REFUSED and qd.log == []


def test_an_unwritable_receipt_is_a_refusal(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    (tmp_path / "receipts").write_text("a file where the directory should be")
    assert _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq)])[0] == MA.EXIT_REFUSED


def test_the_alias_names_the_prod_collection(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    _write_declared(tmp_path, _declared_doc())
    lines: list[str] = []
    code = MA.run(["--twin-url", TWIN, "--twin-collection", "aw_e8", "--prod-url", PROD,
                   "--receipt-dir", str(tmp_path / "r"), "--declared", str(tmp_path / "declared.json"),
                   "--class", "ids", _ids_file(tmp_path, eq)],
                  now=lambda: NOW, settings=_settings(tmp_path), out=lines.append)
    assert code == MA.EXIT_OK
    assert [e[2] for e in qd.calls(PROD, "GET")] == ["/aliases"]       # a metadata GET, not a query
    assert json.loads(next((tmp_path / "r").glob("*.json")).read_text())["counts"]["EQUAL"] == 3


def test_an_explicit_declared_file_that_does_not_exist_is_a_refusal(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq), "--declared", str(tmp_path / "none.json")])
    assert code == MA.EXIT_REFUSED and qd.log == []


# ===================================================================== caps and the read-only rule
def test_the_cap_skips_loudly(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq), "--max-ids", "2"])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_OK and r["checked"] == 2 and r["skipped"] == 1 and r["complete"] is False
    assert [len(e[3]["ids"]) for e in qd.calls(PROD)] == [2]        # the cap holds on the prod request too
    assert any("SKIPPED 1" in ln for ln in lines)


def test_the_deadline_skips_loudly(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    ticks = iter([0.0, 0.0] + [9999.0] * 10)     # start, first batch in time, then past the deadline
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq + [f"{i:08x}-0000-5000-8000-000000000000" for i in range(600)]),
                              "--deadline-min", "1"], clock=lambda: next(ticks))
    r = _receipt(tmp_path)
    assert r["skipped"] > 0 and r["complete"] is False and r["checked"] == 500


def test_prod_is_read_by_id_only_in_batches_of_at_most_500(tmp_path, qd, rows, cmp_e9):
    _world(qd, cmp_e9, rows)
    ids = [f"{i:08x}-0000-5000-8000-000000000000" for i in range(1203)]
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, ids)])
    prod = qd.calls(PROD)
    assert [len(e[3]["ids"]) for e in prod] == [500, 500, 203]
    assert {e[2] for e in qd.log} <= {"/collections/aw_e9/points", "/collections/aw_e8/points"}
    assert {e[1] for e in qd.log} == {"POST"}                   # no PUT, DELETE, scroll or alias GET
    assert not any("scroll" in e[2] or "filter" in json.dumps(e[3]) for e in qd.log)
    assert _receipt(tmp_path)["prod_calls"] == 3


def test_plan_reads_nothing(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq), "--plan"])
    assert code == MA.EXIT_OK and qd.log == [] and any("PLAN: 3 ids" in ln for ln in lines)
    assert not (tmp_path / "receipts").exists()


def test_the_cli_routes_to_the_verb(tmp_path, qd, rows, cmp_e9, monkeypatch):
    _, eq = _world(qd, cmp_e9, rows)
    monkeypatch.setattr(MA, "run", lambda argv, **kw: 3)
    assert cli._ops_main(["mirror-audit", "--plan"]) == 3


# ===================================================================== the declared file
@needs_tracked
def test_the_tracked_declared_file_is_valid_and_documents_its_exclusions():
    """The SCHEMA is pinned, not the set names or counts: the sets are meant to be removed at the aw_e10 flip."""
    doc = json.loads(TRACKED.read_text())
    assert MA.validate_declared(doc) == []
    assert doc["global_allowed_keys"] == []                      # nothing is allowed globally by default
    assert any("bm25" in e["what"] for e in doc["not_compared"])
    for s_ in doc["sets"]:
        assert s_["source"]["expected_ids"] > 0
        assert s_["reason"].strip() and s_["ruling"].strip() and s_["allowed_keys"]


@needs_tracked
@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(schema="nope"), "schema"),
    (lambda d: d["sets"][0].pop("reason"), "reason"),
    (lambda d: d["sets"][0].update(allowed_keys=[]), "allowed_keys"),
    (lambda d: d["sets"][0]["source"].update(kind="elsewhere"), "source.kind"),
    (lambda d: d["sets"][0]["source"].pop("expected_ids"), "expected_ids"),
    (lambda d: d["sets"][0].update(expires="soon"), "expires"),
    (lambda d: d["global_allowed_keys"].append({"key": "title"}), "reason"),
    (lambda d: d["sets"].append(copy.deepcopy(d["sets"][0])), "duplicate"),
])
def test_the_declared_schema_rejects(mutate, needle):
    doc = json.loads(TRACKED.read_text())
    mutate(doc)
    problems = MA.validate_declared(doc)
    assert problems and any(needle in p for p in problems), problems


def test_a_missing_default_file_is_empty_but_a_requested_one_is_a_refusal(tmp_path):
    assert MA.load_declared(tmp_path / "absent.json", required=False).sets == []
    with pytest.raises(MA.Refused):
        MA.load_declared(tmp_path / "absent.json", required=True)
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(MA.Refused):
        MA.load_declared(tmp_path / "bad.json", required=False)


def _ledger_pack(tmp_path, n=3, *, lie_sha=False, lie_count=False):
    base = tmp_path / "news"
    (base / "pack-d").mkdir(parents=True)
    ids = [f"{i:08x}-0000-5000-8000-000000000001" for i in range(n)]
    raw = "".join(json.dumps({"id": i, "payload": {}}) + "\n" for i in ids).encode()
    blob = gzip.compress(raw)
    sha = hashlib.sha256(blob).hexdigest()
    (base / "pack-d" / f"news-points.{sha}.jsonl.gz").write_bytes(blob)
    if lie_sha:
        (base / "pack-d" / f"news-points.{sha}.jsonl.gz").write_bytes(gzip.compress(raw + b"\n"))
    (base / "ledger.jsonl").write_text(
        json.dumps({"status": "RESTORE-PLAN", "artifact_sha256": sha,
                    "artifact_ids": n + (1 if lie_count else 0)}) + "\n")
    return ids, {"id": "led", "reason": "r", "ruling": "r", "allowed_keys": ["last_seen"],
                 "source": {"kind": "apply-ledger-pack", "base": "news_work", "ledger": "ledger.jsonl",
                            "pack_dir": "pack-d", "expected_ids": n}}


def test_a_ledger_pack_source_yields_the_pack_ids(tmp_path):
    ids, entry = _ledger_pack(tmp_path)
    d = MA.Declared(_declared_doc([entry]), tmp_path / "d.json")
    MA.fill_declared_ids(d, _settings(tmp_path))
    assert d.sets[0].ids == set(ids)


@pytest.mark.parametrize("kw,needle", [({"lie_sha": True}, "sha256"), ({"lie_count": True}, "ledger says")])
def test_a_changed_source_is_a_refusal_not_a_redeclaration(tmp_path, kw, needle):
    _, entry = _ledger_pack(tmp_path, **kw)
    d = MA.Declared(_declared_doc([entry]), tmp_path / "d.json")
    with pytest.raises(MA.Refused, match=needle):
        MA.fill_declared_ids(d, _settings(tmp_path))


def test_a_results_source_with_another_count_is_a_refusal(tmp_path):
    entry = _results_set(tmp_path, ["a", "b"], allowed=["x"])
    entry["source"]["expected_ids"] = 3
    d = MA.Declared(_declared_doc([entry]), tmp_path / "d.json")
    with pytest.raises(MA.Refused, match="distinct ids"):
        MA.fill_declared_ids(d, _settings(tmp_path))


def test_exit_codes_are_the_documented_ones():
    assert (MA.EXIT_OK, MA.EXIT_DIVERGED, MA.EXIT_REFUSED) == (0, 3, 4)


# ===================================================================== review 838 fixes
def test_an_artifact_whose_prod_leg_is_pending_is_not_audited(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    _artifact(tmp_path, "p-pending", at="2026-10-09T10:00:00Z", ids=[old[0]], roles=("twin",))
    _artifact(tmp_path, "p-done", at="2026-10-09T10:00:00Z", ids=[eq[0]])
    code, _ = _run(tmp_path, ["--class", "inplace-since", "2026-10-09"])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_OK and r["checked"] == 1             # the stale row of the pending pack is the drain's state
    assert r["class"]["inplace"][0]["prod_leg_pending"] == 1


def test_a_finding_is_read_again_and_kept_only_if_it_persists(tmp_path, qd, rows, cmp_e9):
    old, eq = _world(qd, cmp_e9, rows)
    healed = {e["id"]: _stamped(e, cmp_e9, "twin") for e in rows["prod_only_old"]}

    def drain_finishes(secs):
        assert secs == 60
        qd.put(PROD, "aw_e9", healed)

    code, lines = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old + eq)], sleep=drain_finishes)
    r = _receipt(tmp_path)
    assert code == MA.EXIT_OK and r["counts"]["EQUAL"] == 6 and r["undeclared"] == 0
    assert r["recheck"] == {"suspects": 3, "cleared": 3, "after_s": 60, "skipped": False}
    assert r["prod_calls"] == 2 and any("RECHECK" in ln for ln in lines)
    # a divergence that persists is still exit 3, and the recheck cost one more call
    _world(qd, cmp_e9, rows)
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_DIVERGED and r["recheck"]["cleared"] == 0 and r["prod_calls"] == 2


def test_recheck_off_and_recheck_over_budget(tmp_path, qd, rows, cmp_e9, monkeypatch):
    old, _ = _world(qd, cmp_e9, rows)
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old), "--recheck-s", "0"])
    assert code == MA.EXIT_DIVERGED and _receipt(tmp_path)["prod_calls"] == 1
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "1")
    PG.reset()
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    r = _receipt(tmp_path)
    assert code == MA.EXIT_DIVERGED and r["recheck"]["skipped"] is True and r["counts"]["DIFF-undeclared"] == 3


def test_the_receipt_records_the_prod_time(tmp_path, qd, rows, cmp_e9):
    _, eq = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, eq)])
    r = _receipt(tmp_path)
    assert r["prod_seconds_sum"] >= r["prod_seconds_max"] >= 0 and "prod_seconds_sum" in r


def test_the_declared_class_rotates_in_slices_that_cover_every_id(tmp_path, qd, rows, cmp_e9):
    ids = [f"{i:08x}-0000-5000-8000-000000000002" for i in range(70)]
    _write_declared(tmp_path, _declared_doc([_results_set(tmp_path, ids, allowed=["last_seen"],
                                                           allow_missing=True)]))
    seen: set[str] = set()
    for day in range(7):
        now = NOW + dt.timedelta(days=day)
        MA.run(["--twin-url", TWIN, "--twin-collection", "aw_e8", "--prod-url", PROD,
                "--prod-collection", "aw_e9", "--receipt-dir", str(tmp_path / f"r{day}"),
                "--declared", str(tmp_path / "declared.json"), "--class", "declared",
                "--declared-slice", "7"], now=lambda: now, settings=_settings(tmp_path), out=lambda l: None,
               sleep=lambda s_: None)
        rec = json.loads(next((tmp_path / f"r{day}").glob("*.json")).read_text())
        sl = rec["class"]["declared_slice"]
        assert sl["n"] == 7 and sl["ids_declared"] == 70
        seen |= {i for i in ids if MA._slice_of(i, 7) == sl["index"]}
        assert rec["ids_in_class"] == sl["ids_in_slice"] < 70
    assert seen == set(ids)                                      # in 7 nights every id is read once


def test_an_id_in_two_sets_belongs_to_the_first(tmp_path, qd, rows, cmp_e9):
    old, _ = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old), "--recheck-s", "0"])
    seen = {k for e in _receipt(tmp_path)["examples"] for k in e["keys"]}
    first = _results_set(tmp_path, old, allowed=seen, set_id="first", name="a.jsonl.gz")
    second = _results_set(tmp_path, old, allowed=["last_seen"], set_id="second", name="b.jsonl.gz")
    _write_declared(tmp_path, _declared_doc([first, second]))
    assert _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])[0] == MA.EXIT_OK
    _write_declared(tmp_path, _declared_doc([second, first]))
    assert _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])[0] == MA.EXIT_DIVERGED


# ===================================================================== per-id keys (task 839)
def _idkeys_set(tmp_path, id_keys, *, base=("last_seen", "published"), set_id="ik", sha=None):
    path = tmp_path / "idkeys.json"
    raw = json.dumps({"ids": id_keys}).encode()
    path.write_bytes(raw)
    return {"id": set_id, "reason": "a reason", "ruling": "a ruling", "allowed_keys": list(base),
            "source": {"kind": "id-keys-json", "path": str(path),
                       "sha256": sha or hashlib.sha256(raw).hexdigest(), "expected_ids": len(id_keys)}}


def test_per_id_keys_allow_exactly_the_listed_keys_of_that_id(tmp_path, qd, rows, cmp_e9):
    old, _ = _world(qd, cmp_e9, rows)
    _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    seen = {e["id"]: [k for k in e["keys"] if k not in ("last_seen", "published")]
            for e in _receipt(tmp_path)["examples"]}
    assert all(seen.values())
    ok = dict(seen)
    ok[old[0]] = [k for k in seen[old[0]] if k != "dense"] or ["title"]       # one id lacks the key it needs
    _write_declared(tmp_path, _declared_doc([_idkeys_set(tmp_path, ok)]))
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    r = _receipt(tmp_path)
    undecl = {e["id"] for e in r["examples"]}
    assert (code == MA.EXIT_DIVERGED) == bool(undecl)
    assert undecl <= {old[0]}                                   # the other ids are covered with their keys
    assert r["counts"]["DIFF-declared"] + r["counts"]["DIFF-undeclared"] == 3
    assert r["counts"]["DIFF-declared"] >= 2


def test_a_changed_per_id_file_is_a_refusal(tmp_path, qd, rows, cmp_e9):
    old, _ = _world(qd, cmp_e9, rows)
    _write_declared(tmp_path, _declared_doc([_idkeys_set(tmp_path, {old[0]: ["dense"]}, sha="0" * 64)]))
    code, _ = _run(tmp_path, ["--class", "ids", _ids_file(tmp_path, old)])
    assert code == MA.EXIT_REFUSED and "sha256" in _receipt(tmp_path)["refusal"]


def _per_id_file(set_entry):
    rel = pathlib.Path(set_entry["source"]["path"]).relative_to("data")
    return (TRACKED.parent / rel).read_bytes()


@needs_tracked
def test_every_id_of_the_38_fixture_is_covered_by_exactly_one_set_with_its_keys():
    """The 38 DIFF-undeclared ids of the first audit run. The per-id file is the only source that is
    readable without the box, so it is the one that must hold them."""
    fx = json.loads((HERE / "fixtures" / "mirror_audit_news626f_38.v1.json").read_text())["ids"]
    doc = json.loads(TRACKED.read_text())
    assert MA.validate_declared(doc) == [] and len(fx) == 38
    owners: dict[str, list] = {i: [] for i in fx}
    # F7 (task 964): three sets are RETIRED to doc["retired"]; the fixture ids stay covered there
    for s_ in doc["sets"] + doc.get("retired", []):
        if s_["source"]["kind"] != "id-keys-json":
            continue
        m = json.loads(_per_id_file(s_))["ids"]
        for i in fx:
            if i in m:
                owners[i].append(set(s_["allowed_keys"]) | set(m[i]))
    for i, keys in fx.items():
        assert len(owners[i]) == 1, i
        assert set(keys) <= owners[i][0], (i, keys)


@needs_tracked
def test_the_per_id_file_matches_its_pinned_sha_and_count():
    doc = json.loads(TRACKED.read_text())
    for s_ in doc["sets"]:
        if s_["source"]["kind"] == "id-keys-json":
            raw = _per_id_file(s_)
            assert hashlib.sha256(raw).hexdigest() == s_["source"]["sha256"]
            assert len(json.loads(raw)["ids"]) == s_["source"]["expected_ids"]
