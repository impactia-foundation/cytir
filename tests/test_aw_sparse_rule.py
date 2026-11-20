"""Task 717 -- the drain writes PROD rows under the aw_e9 BM25 prune rule; the twin stays raw.

Real rows, no live calls. `tests/data/sparse_rule_real_rows.json` holds rows read on 2026-10-07
(read-only) from the twin `aw_e8` (raw) and from prod `aw_e9` (by id), seed 7131, from a sample of
28,964 ids that are on both (task 717's `identity.json`: 19,978 rows where prod ==
rule(twin), 8,971 where prod == twin, 15 where the rule changes nothing, 0 other). Per row the
fixture keeps the twin vector, the prod vector, and, for every term in them, whether the REAL
231 MB term set drops it (`in_drop_set`); the tests rebuild a term set with exactly that
membership over the fixture's terms, so the answer for every term the rows touch is the real one.

What each group would have caught:

* the rule applied to the twin's vector equals, bit for bit, what the aw_e9 rebuild stored;
* the writer door cuts prod's rows and leaves the twin's raw, in the same call, and stamps the
  epoch in the SAME `rule_epochs` code the enum table decodes;
* a row that would carry a dropped term, a missing term set, a wrong term set, a row that cannot
  carry the marker, and a stored row that still holds a dropped term, are each refused;
* the German de-compounded parts survive the rule (task 708 measured 96-97 %).
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import struct
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core import paths as PATHS  # noqa: E402
from cytir.index import payload_profile as PP  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.index import sparse_rule as SR  # noqa: E402

FIXTURE = REPO / "tests" / "data" / "sparse_rule_real_rows.json"
RULE_EPOCHS = {"card": 1, "chunk": 5, "country": 2, "embed": 1, "extract": 1, "tok": 2, "trust": 1}


def f32(x):
    return struct.pack("<f", x)


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def rule_env(fx, tmp_path, monkeypatch):
    """A term set with the REAL membership over the fixture's terms (and the German parts), declared
    for aw_e9 in a temporary copy of the profiles file, found through the operator setting."""
    members = set(fx["in_drop_set"]) | {int(h) for h, dropped in fx["de_part_membership"].items() if dropped}
    f = tmp_path / "drop.u32"
    written = SR.write_drop_terms(f, members)
    profiles = json.loads(PP.PROFILES_PATH.read_text())
    profiles["collections"]["aw_e9"]["sparse_rule"] = dict(
        profiles["collections"]["aw_e9"]["sparse_rule"], terms=written["terms"],
        drop_terms_sha256=written["sha256"])
    pf = tmp_path / "profiles.json"
    pf.write_text(json.dumps(profiles))
    monkeypatch.setattr(PP, "PROFILES_PATH", pf)
    monkeypatch.setenv("AW_SPARSE_RULE_DROP_TERMS", str(f))
    PATHS.reset_settings_cache()
    yield f
    PATHS.reset_settings_cache()


def _capture(monkeypatch):
    """Record every PUT per collection and answer retrieves from what was PUT (a fake store)."""
    puts, store = [], {}

    def api(url, path, body=None, method=None):
        if method == "PUT":
            coll = path.split("/")[2]
            puts.append((url, coll, body["points"]))
            for p in body["points"]:
                store[(coll, p["id"])] = p
            return {}
        coll = path.split("/")[2]
        return [dict(store[(coll, i)], payload=store[(coll, i)]["payload"]) for i in body["ids"]
                if (coll, i) in store]

    monkeypatch.setattr(PSP, "_api", api)
    return puts, store


def _point(row, payload=None):
    return {"id": row["id"],
            "vector": {"dense": [0.0, 1.0], "bm25": {"indices": list(row["twin"]["indices"]),
                                                     "values": list(row["twin"]["values"])}},
            "payload": payload if payload is not None else {"url": "https://x/" + row["id"],
                                                            "rule_epochs": dict(RULE_EPOCHS)}}


# --------------------------------------------------------- real rows: the rule is the rebuild's
def test_the_rule_on_the_twin_vector_is_the_vector_the_rebuild_stored(fx, rule_env):
    drop = SR.SparseRule.for_collection("aw_e9").drop
    identical = [r for r in fx["rows"] if r["class"] == "rule_identical"]
    assert len(identical) >= 45, "the fixture must hold real rows that the rule changes"
    for r in identical:
        ki, kv = SR.prune_postings(r["twin"]["indices"], r["twin"]["values"], drop)
        assert len(ki) < len(r["twin"]["indices"]), f"{r['id']}: the rule must change this row"
        assert ki == r["prod"]["indices"], f"{r['id']}: indices or order differ from the rebuild's row"
        assert [f32(v) for v in kv] == [f32(v) for v in r["prod"]["values"]], r["id"]


def test_a_drain_row_on_prod_is_the_twin_row_unpruned(fx, rule_env):
    """The defect, on real rows: prod holds the twin's raw vector and the rule would cut it."""
    drop = SR.SparseRule.for_collection("aw_e9").drop
    rows = [r for r in fx["rows"] if r["class"] == "prod_unpruned"]
    assert rows
    for r in rows:
        assert r["prod"] == r["twin"]
        ki, _ = SR.prune_postings(r["twin"]["indices"], r["twin"]["values"], drop)
        assert len(r["prod"]["indices"]) - len(ki) >= 1


def test_the_term_set_and_a_python_set_cut_alike(fx, rule_env):
    drop = SR.SparseRule.for_collection("aw_e9").drop
    as_set = set(fx["in_drop_set"]) | {int(h) for h, d in fx["de_part_membership"].items() if d}
    for r in fx["rows"]:
        a = SR.prune_postings(r["twin"]["indices"], r["twin"]["values"], drop)
        b = SR.prune_postings(r["twin"]["indices"], r["twin"]["values"], as_set)
        assert a == b
    rng = random.Random(713)
    probe = [rng.getrandbits(32) for _ in range(500)] + sorted(as_set)[:200]
    assert drop.contains_each(probe) == [t in as_set for t in probe]
    assert drop.contains_each([-1, 2 ** 32, 2 ** 40]) == [False, False, False]


# ----------------------------------------------------------------------- the declaration
def test_the_twin_declares_no_rule_and_prod_does():
    assert SR.declared_epoch("aw_e8") is None and SR.declared_epoch("not-a-declared-collection") is None
    assert SR.declared_epoch("aw_e9") == 1
    assert SR.SparseRule.for_collection("aw_e8") is None


def test_a_box_without_the_term_set_refuses_by_name_and_never_writes_raw(monkeypatch):
    monkeypatch.delenv("AW_SPARSE_RULE_DROP_TERMS", raising=False)
    PATHS.reset_settings_cache()
    with pytest.raises(SR.SparseRuleError, match="sparse_rule_drop_terms"):
        SR.SparseRule.for_collection("aw_e9")
    puts, _ = _capture(monkeypatch)
    with pytest.raises(SR.SparseRuleError):
        PSP.upsert("http://prod", "aw_e9", [{"id": "x", "vector": {}, "payload": {}}], batch=10)
    assert puts == []


def test_a_term_set_that_is_not_the_declared_one_is_refused(rule_env, tmp_path):
    other = tmp_path / "other.u32"
    np.array([1, 2, 3], dtype="<u4").tofile(other)
    entry = json.loads(PP.PROFILES_PATH.read_text())["collections"]["aw_e9"]["sparse_rule"]
    with pytest.raises(SR.SparseRuleError, match="hashes to"):
        SR.DropTerms(other, expect_sha256=entry["drop_terms_sha256"], expect_terms=entry["terms"])
    with pytest.raises(SR.SparseRuleError, match="holds 3 terms"):
        SR.DropTerms(other, expect_sha256=SR._sha256_file(other), expect_terms=4)
    with pytest.raises(SR.SparseRuleError, match="does not exist"):
        SR.DropTerms(tmp_path / "absent.u32", expect_sha256="0" * 64, expect_terms=1)


# ------------------------------------------------------------------------- the writer door
def test_prod_is_written_pruned_and_the_twin_raw_from_the_same_points(fx, rule_env, monkeypatch):
    puts, _ = _capture(monkeypatch)
    rows = [r for r in fx["rows"] if r["class"] == "rule_identical"][:20]
    pts = [_point(r) for r in rows]
    before = json.dumps(pts, sort_keys=True)
    PSP.upsert("http://twin", "aw_e8", pts, batch=7)
    PSP.upsert("http://prod", "aw_e9", pts, batch=7)
    assert json.dumps(pts, sort_keys=True) == before, "the twin leg reuses the same list: raw"
    twin = {p["id"]: p for u, c, b in puts if c == "aw_e8" for p in b}
    prod = {p["id"]: p for u, c, b in puts if c == "aw_e9" for p in b}
    tables = PP.load_tables(expect_sha256=PP.entry_for_collection("aw_e9")["tables_sha256"])
    inverted = {code: json.loads(v) for v, code in tables["rule_epochs"].items()}
    for r in rows:
        assert twin[r["id"]]["vector"]["bm25"] == {"indices": r["twin"]["indices"], "values": r["twin"]["values"]}
        assert twin[r["id"]]["payload"]["rule_epochs"] == RULE_EPOCHS, "the twin carries no marker"
        got = prod[r["id"]]["vector"]["bm25"]
        assert got["indices"] == r["prod"]["indices"]
        assert [f32(v) for v in got["values"]] == [f32(v) for v in r["prod"]["values"]]
        assert prod[r["id"]]["vector"]["dense"] == [0.0, 1.0]
        decoded = inverted[prod[r["id"]]["payload"]["rule_epochs"]]
        assert decoded == dict(RULE_EPOCHS, bm25_prune=1)


def test_every_aw_e9_rule_epochs_combination_has_a_marked_code():
    tables = PP.load_tables(expect_sha256=PP.entry_for_collection("aw_e9")["tables_sha256"])["rule_epochs"]
    plain = [json.loads(v) for v in tables if "bm25_prune" not in v]
    assert len(plain) == 6  # the three aw_e9 dicts, and each plus prov_id=2 (task 1361)
    for combo in plain:
        marked = json.dumps(dict(sorted(dict(combo, bm25_prune=1).items())))
        assert marked in tables, f"{combo}: no code for its marked form"


def test_a_row_whose_every_term_is_dropped_loses_its_bm25_vector(fx, rule_env, monkeypatch):
    puts, _ = _capture(monkeypatch)
    some = fx["in_drop_set"][:3]
    pt = {"id": "p1", "vector": {"dense": [0.1], "bm25": {"indices": some, "values": [1.0, 2.0, 3.0]}},
          "payload": {"rule_epochs": dict(RULE_EPOCHS)}}
    PSP.upsert("http://prod", "aw_e9", [pt], batch=5)
    assert "bm25" not in puts[0][2][0]["vector"] and "dense" in puts[0][2][0]["vector"]


def test_a_row_that_cannot_carry_the_marker_is_refused_before_any_put(fx, rule_env, monkeypatch):
    puts, _ = _capture(monkeypatch)
    row = fx["rows"][0]
    for bad in ({}, {"rule_epochs": 0}, {"rule_epochs": None}):
        with pytest.raises(SR.SparseRuleError, match="rule_epochs"):
            PSP.upsert("http://prod", "aw_e9", [_point(row, dict(bad, url="u"))], batch=5)
    assert puts == []


def test_a_vector_that_would_keep_a_dropped_term_is_refused(fx, rule_env, monkeypatch):
    puts, _ = _capture(monkeypatch)
    monkeypatch.setattr(SR, "prune_postings", lambda i, v, d: (list(i), list(v)))   # a broken cut
    row = next(r for r in fx["rows"] if r["class"] == "rule_identical")
    with pytest.raises(SR.SparseRuleError, match="dropped terms left"):
        PSP.upsert("http://prod", "aw_e9", [_point(row)], batch=5)
    assert puts == []


def test_a_stored_row_that_still_holds_a_dropped_term_fails_the_readback(fx, rule_env, monkeypatch):
    puts, store = _capture(monkeypatch)
    row = next(r for r in fx["rows"] if r["class"] == "rule_identical")
    real_api = PSP._api

    def tampering(url, path, body=None, method=None):
        got = real_api(url, path, body, method)
        if method != "PUT":     # the collection "stores" the raw vector: the write did not prune
            for g in got:
                g["vector"] = {"bm25": {"indices": row["twin"]["indices"], "values": row["twin"]["values"]}}
        return got

    monkeypatch.setattr(PSP, "_api", tampering)
    with pytest.raises(SR.SparseRuleError, match="not in the aw_e9-df-prune form"):
        PSP.upsert("http://prod", "aw_e9", [_point(row)], batch=5)


def test_the_preflight_checks_the_marked_form_before_the_twin_leg(fx, rule_env):
    ok = PSP.payload_profile_preflight("aw_e9", [{"rule_epochs": dict(RULE_EPOCHS)}])
    assert ok["sparse_rule_epoch"] == 1 and ok["rows_checked"] == 1
    twin = PSP.payload_profile_preflight("aw_e8", [{"rule_epochs": dict(RULE_EPOCHS)}])
    assert "sparse_rule_epoch" not in twin
    with pytest.raises(PP.UnknownValueError):       # chunk 7 has no code, marked or not
        PSP.payload_profile_preflight("aw_e9", [{"rule_epochs": dict(RULE_EPOCHS, chunk=7)}])
    with pytest.raises(SR.SparseRuleError):
        PSP.payload_profile_preflight("aw_e9", [{"url": "no rule_epochs"}])


def test_verify_written_form_reads_the_marked_code_on_prod(fx, rule_env, monkeypatch):
    pts = [_point(r) for r in fx["rows"][:6]]
    marked = PP.Projector.for_collection("aw_e9").project(SR.stamp_epoch(pts[0]["payload"], 1))
    plain = PP.Projector.for_collection("aw_e9").project(pts[0]["payload"])
    assert marked["rule_epochs"] != plain["rule_epochs"]
    stored = {p["id"]: PP.Projector.for_collection("aw_e9").project(SR.stamp_epoch(p["payload"], 1)) for p in pts}
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {i: {"payload": stored[i], "vector": {"dense": [0.0, 1.0]}} for i in ids})
    assert Q.verify_written_form("http://prod", "aw_e9", pts)["mismatches"] == 0
    unmarked = {p["id"]: PP.Projector.for_collection("aw_e9").project(p["payload"]) for p in pts}
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {i: {"payload": unmarked[i], "vector": {"dense": [0.0, 1.0]}} for i in ids})
    with pytest.raises(Q.QueueError, match="not in the declared"):
        Q.verify_written_form("http://prod", "aw_e9", pts)


# --------------------------------------------------------------------------- German parts
def test_the_rule_keeps_the_german_decompounded_parts(fx, rule_env):
    """German DRAIN rows (the rows the rule will cut): the de-compounded parts that the text adds
    as postings survive it. Task 708 measured 97.3 %; the whole sample of this task read 96.2 %
    on 7,418 German drain rows (task 717's `german-parts.out.json`)."""
    # tok@2 folds accents at import time (AW_FOLD_ACCENTS), so the part hashes come from a child
    # process that has it set: the test never silently skips and never depends on the parent's env.
    child = (
        "import json,sys\n"
        "from cytir.core import bm25_tokenizer as C\n"
        "assert C._FOLD_ACCENTS and C._decompound('bundesverfassung'), 'compound_split or the fold is off'\n"
        "out = []\n"
        "for r in json.load(sys.stdin):\n"
        "    text = ' '.join(x for x in (r.get('title'), r.get('snippet')) if x)\n"
        "    toks, parts = set(), set()\n"
        "    for tok in C._TOKEN_RE.findall(text.lower()):\n"
        "        if len(tok) < 2: continue\n"
        "        toks.add(C._hash(C._fold(tok)))\n"
        "        for part in C._decompound(tok): parts.add(C._hash(C._fold(part)))\n"
        "    out.append(sorted(parts - toks))\n"
        "print(json.dumps(out))\n")
    env = dict(os.environ, AW_FOLD_ACCENTS="1", PYTHONPATH=str(REPO))
    done = subprocess.run([sys.executable, "-c", child], input=json.dumps(fx["de_rows"]), env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    drop = SR.SparseRule.for_collection("aw_e9").drop
    parts_total = kept = 0
    for new_parts in json.loads(done.stdout):
        for h in new_parts:
            assert str(h) in fx["de_part_membership"], "the fixture must answer every part"
            parts_total += 1
            kept += not drop.contains_each([h])[0]
    assert parts_total >= 1000, f"positive control: {parts_total} part postings"
    assert kept / parts_total >= 0.95, f"{kept}/{parts_total} parts kept"


# ------------------------------------------- review fixes: load before the twin, no silent raw
def _main_env(monkeypatch, tmp_path):
    """Run `Q.main(--apply)` in process with everything but the new startup hook stubbed. `drain`
    is the stand-in for "the twin PUT would follow": a call to it is the twin leg starting."""
    puts = []
    calls = []

    def api(url, path, body=None, method=None):
        if method == "PUT":
            puts.append((url, path))
            return {}
        if path == "/aliases":
            return {"aliases": [{"alias_name": "aw_prod", "collection_name": "aw_e9"}]}
        raise AssertionError("unexpected call " + path)

    monkeypatch.setattr(PSP, "_api", api)
    monkeypatch.setattr(Q, "plan_startup_refusal", lambda: None)
    monkeypatch.setattr(Q.GPG, "startup_refusal", lambda: None)
    monkeypatch.setattr(Q, "apply_startup_refusal", lambda: None)
    monkeypatch.setattr(Q, "drain", lambda *a, **k: calls.append("drain") or [])
    q = tmp_path / "queue"
    q.mkdir()
    return puts, calls, ["--apply", "--queue-dir", str(q)]


def test_an_unset_term_set_exits_2_before_any_twin_write(rule_env, monkeypatch, tmp_path, capsys):
    """The twin leg is written BEFORE the prod leg opens the term set; a missing file there left the
    twin one pack ahead of prod. Resolved at the start of the drain, it stops the run: non-zero
    exit, one [REFUSED] line, no drain, no PUT. A per-pack REFUSED (exit 0) is NOT how a rule failure
    ends: by then the rule has been loaded or the run has stopped, so the cause 'the rule' never
    reaches the per-pack path as a quiet exit 0."""
    puts, calls, argv = _main_env(monkeypatch, tmp_path)
    monkeypatch.delenv("AW_SPARSE_RULE_DROP_TERMS")
    PATHS.reset_settings_cache()
    assert Q.main(argv) == 2
    err = capsys.readouterr().err
    assert err.count("[REFUSED]") == 1 and "sparse_rule_drop_terms" in err
    assert calls == [] and puts == []


def test_a_wrong_term_set_exits_2_before_any_twin_write(rule_env, monkeypatch, tmp_path, capsys):
    puts, calls, argv = _main_env(monkeypatch, tmp_path)
    other = tmp_path / "other.u32"
    np.array([1, 2, 3], dtype="<u4").tofile(other)
    monkeypatch.setenv("AW_SPARSE_RULE_DROP_TERMS", str(other))
    PATHS.reset_settings_cache()
    assert Q.main(argv) == 2
    assert "hashes to" in capsys.readouterr().err
    assert calls == [] and puts == []


def test_a_good_term_set_lets_the_drain_start_and_prints_its_sha(rule_env, monkeypatch, tmp_path, capsys):
    puts, calls, argv = _main_env(monkeypatch, tmp_path)
    assert Q.main(argv) == 0
    assert calls == ["drain"]
    out = capsys.readouterr().out
    assert "[sparse-rule]" in out and SR._sha256_file(rule_env) in out


def test_an_unresolvable_prod_alias_refuses_at_start_too(rule_env, monkeypatch, tmp_path, capsys):
    puts, calls, argv = _main_env(monkeypatch, tmp_path)

    def down(url, path, body=None, method=None):
        raise OSError("connection refused")

    monkeypatch.setattr(PSP, "_api", down)
    assert Q.main(argv) == 2
    assert capsys.readouterr().err.count("[REFUSED]") == 1 and calls == []


def test_a_dry_drain_needs_no_prod_for_the_rule(rule_env, monkeypatch, tmp_path):
    puts, calls, argv = _main_env(monkeypatch, tmp_path)

    def no_prod(*a, **k):
        raise AssertionError("a dry drain makes no prod call for the rule")

    monkeypatch.setattr(PSP, "_api", no_prod)
    assert Q.main([a for a in argv if a != "--apply"]) == 0 and calls == ["drain"]


def test_the_term_set_is_hashed_once_per_run_not_per_upsert(fx, rule_env, monkeypatch):
    puts, _ = _capture(monkeypatch)
    hashed = []
    real = SR._sha256_file
    monkeypatch.setattr(SR, "_sha256_file", lambda p: hashed.append(str(p)) or real(p))
    first = SR.SparseRule.for_collection("aw_e9")
    row = next(r for r in fx["rows"] if r["class"] == "rule_identical")
    for _ in range(3):
        PSP.upsert("http://prod", "aw_e9", [_point(row)], batch=5)
    assert SR.SparseRule.for_collection("aw_e9") is first
    assert len(hashed) == 1, hashed
    assert first.drop.sha256 == real(rule_env)


def _declare(tmp_path, monkeypatch, **extra):
    """Add an `aw_e10`-shaped collection (aw_e9's profile and tables) to the temporary profiles."""
    profiles = json.loads(PP.PROFILES_PATH.read_text())
    entry = {k: v for k, v in profiles["collections"]["aw_e9"].items() if k != "sparse_rule"}
    profiles["collections"]["aw_e10"] = dict(entry, **extra)
    pf = tmp_path / "profiles-e10.json"
    pf.write_text(json.dumps(profiles))
    monkeypatch.setattr(PP, "PROFILES_PATH", pf)


def test_a_declared_collection_without_a_rule_key_is_refused_never_written_raw(fx, rule_env, monkeypatch, tmp_path):
    """After the aw_e10 flip the drain writes `aw_e10`. A profile that declares it but says nothing
    about the rule must stop the prod leg, not write raw sparse vectors."""
    _declare(tmp_path, monkeypatch)
    puts, _ = _capture(monkeypatch)
    with pytest.raises(SR.SparseRuleError, match="without a `sparse_rule` key"):
        SR.SparseRule.for_collection("aw_e10")
    with pytest.raises(SR.SparseRuleError, match="without a `sparse_rule` key"):
        SR.declared_epoch("aw_e10")
    with pytest.raises(SR.SparseRuleError, match="without a `sparse_rule` key"):
        PSP.upsert("http://prod", "aw_e10", [_point(fx["rows"][0])], batch=5)
    with pytest.raises(SR.SparseRuleError):        # and before the twin leg, in the preflight
        PSP.payload_profile_preflight("aw_e10", [{"rule_epochs": dict(RULE_EPOCHS)}])
    assert puts == []


def test_an_explicit_null_is_the_only_way_to_declare_no_rule(fx, rule_env, monkeypatch, tmp_path):
    _declare(tmp_path, monkeypatch, sparse_rule=None)
    puts, _ = _capture(monkeypatch)
    assert SR.SparseRule.for_collection("aw_e10") is None and SR.declared_epoch("aw_e10") is None
    pt = _point(fx["rows"][0])
    PSP.upsert("http://prod", "aw_e10", [pt], batch=5)
    assert puts and puts[0][2][0]["vector"]["bm25"] == pt["vector"]["bm25"]       # raw, on purpose
    # the twin and an undeclared stranger's target keep working
    assert SR.SparseRule.for_collection("aw_e8") is None
    assert SR.SparseRule.for_collection("a-strangers-target") is None


def test_the_startup_hook_refuses_a_prod_collection_with_no_rule_key(rule_env, monkeypatch, tmp_path, capsys):
    puts, calls, argv = _main_env(monkeypatch, tmp_path)
    _declare(tmp_path, monkeypatch)
    monkeypatch.setattr(PSP, "_api", lambda u, p, b=None, method=None: {"aliases": [
        {"alias_name": "aw_prod", "collection_name": "aw_e10"}]})
    assert Q.main(argv) == 2
    assert "without a `sparse_rule` key" in capsys.readouterr().err and calls == []


def test_the_readback_result_is_kept_for_the_receipt(fx, rule_env, monkeypatch):
    puts, store = _capture(monkeypatch)
    SR.LAST_READBACK.clear()
    pts = [_point(r) for r in fx["rows"][:6]]
    PSP.upsert("http://prod", "aw_e9", pts, batch=5)
    rb = SR.LAST_READBACK["aw_e9"]
    assert rb["checked"] == 6 and rb["mismatches"] == 0 and rb["epoch"] == 1
    assert rb["drop_terms_sha256"] == SR._sha256_file(rule_env)
    stored = {p["id"]: PP.Projector.for_collection("aw_e9").project(SR.stamp_epoch(p["payload"], 1)) for p in pts}
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {i: {"payload": stored[i], "vector": {"dense": [0.0, 1.0]}} for i in ids})
    assert Q.verify_written_form("http://prod", "aw_e9", pts)["sparse_rule_readback"] == rb


def test_2938_FINDING_a_later_packs_verify_without_its_own_write_reports_the_earlier_packs_readback(
        fx, rule_env, monkeypatch):
    """Task 2938, today's behaviour, pinned (not a decision to keep it). `LAST_READBACK` is a process-wide dict keyed
    by the physical collection, written after EVERY rule upsert and read by `verify_written_form` into the receipt
    field `sparse_rule_readback`. Pack 1 upserts six rows into aw_e9. Pack 2 (two other rows, which the test
    stores itself: a pure payload patch or a resume writes nothing through `upsert`) is verified on the same
    collection: its receipt carries pack 1's readback (`checked` 6, not 2): a receipt line attributed to the
    wrong pack. No write decision reads it. Clearing the dict (what the test fixture does between tests) removes the key."""
    puts, store = _capture(monkeypatch)
    SR.LAST_READBACK.clear()
    pack1 = [_point(r) for r in fx["rows"][:6]]
    PSP.upsert("http://prod", "aw_e9", pack1, batch=5)
    rb1 = dict(SR.LAST_READBACK["aw_e9"])
    assert rb1["checked"] == 6
    pack2 = [_point(r) for r in fx["rows"][6:8]]
    stored = {p["id"]: PP.Projector.for_collection("aw_e9").project(SR.stamp_epoch(p["payload"], 1)) for p in pack2}
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {
        i: {"payload": stored[i], "vector": {"dense": [0.0, 1.0]}} for i in ids})
    got = Q.verify_written_form("http://prod", "aw_e9", pack2)
    assert got["checked"] == 2 and got["sparse_rule_readback"] == rb1 and got["sparse_rule_readback"]["checked"] == 6
    SR.LAST_READBACK.clear()                          # keyed by collection and never reset by the drain
    assert "sparse_rule_readback" not in Q.verify_written_form("http://prod", "aw_e9", pack2)


# ------------------------------------------------ task 2942: the readback belongs to the leg that wrote, not to the process
def _content_resume_after_an_earlier_pack(fx, rule_env, monkeypatch, tmp_path, *, resume_upserts: bool) -> dict:
    """Pack 1 (a content pack) upserts six rows into aw_e9 through the real writer. Then a prod-only content-pack
    RESUME of pack 2 (two other rows) runs on the same collection in the same process, through the real
    `resume_content_pack_prod_only` and the real `verify_written_form`. Fakes: the lock runner, the TFA protocol,
    the alias check, the layout fingerprint, and `apply_content_pack_side` (`resume_upserts`: it upserts the two
    rows through the real writer, else it writes nothing, like a resume whose chunk checkpoint holds every batch)."""
    import types
    puts, store = _capture(monkeypatch)
    SR.LAST_READBACK.clear()
    PSP.upsert("http://prod", "aw_e9", [_point(r) for r in fx["rows"][:6]], batch=5)
    assert SR.LAST_READBACK["aw_e9"]["checked"] == 6
    pack2 = [_point(r) for r in fx["rows"][6:8]]
    stored = {p["id"]: PP.Projector.for_collection("aw_e9").project(SR.stamp_epoch(p["payload"], 1)) for p in pack2}
    monkeypatch.setattr(Q.PD, "fetch_rows", lambda url, coll, ids, batch=1000, vectors=None: {
        i: {"payload": stored[i], "vector": {"dense": [0.0, 1.0]}} for i in ids})

    class RB:
        LOCKS = ("L1",)
        assert_locks_free = staticmethod(lambda task: None)
        acquire = staticmethod(lambda name, task, why, pid=None: None)
        release = staticmethod(lambda name, task: None)

    def tfa_resume(**kw):
        st = {}
        kw["apply_prod"](st)
        return {"status": "LANDED", **kw["verify_mirrors"](st)}
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q, "TFA", types.SimpleNamespace(
        ProductionWriterLock=lambda **kw: "THE-LOCK", resume_after_verified_twin_gate=tfa_resume,
        ApplyAbort=Q.TFA.ApplyAbort))
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda *a, **k: {"fp": 1})
    monkeypatch.setattr(Q.PSP, "already_present", lambda u, c, ids, **k: list(ids))
    monkeypatch.setattr(Q, "load_content_pack_points", lambda path: pack2)

    def apply_side(role, url, coll, points, tombs, **kw):
        if resume_upserts:
            PSP.upsert(url, coll, points, batch=5)
        return {"written": len(points) if resume_upserts else 0, "tombstoned": 0}
    monkeypatch.setattr(Q, "apply_content_pack_side", apply_side)
    ids = [p["id"] for p in pack2]
    out = Q.resume_content_pack_prod_only(
        pack_id="p-2", manifest={"pack_id": "p-2", "collection": "aw_e8", "pack_points": "unused"},
        state_record={"stage": "twin-write-complete", "pack_ids": ids, "tomb_ids": []},
        twin_url="http://twin.invalid", prod_url="http://prod", task="t", batch=5, pause_s=0,
        artifact_root=tmp_path / "a", prod_collection="aw_e9")
    return out["payload_form"]


def test_2942_a_content_pack_resume_that_writes_nothing_does_not_carry_the_earlier_packs_readback(
        fx, rule_env, monkeypatch, tmp_path):
    """RED on main: the receipt of pack 2 carried pack 1's readback (`checked` 6 for a two-row pack)."""
    form = _content_resume_after_an_earlier_pack(fx, rule_env, monkeypatch, tmp_path, resume_upserts=False)
    assert form["checked"] == 2
    assert "sparse_rule_readback" not in form


def test_2942_a_content_pack_resume_that_upserts_carries_its_own_readback(fx, rule_env, monkeypatch, tmp_path):
    """The normal path, unchanged: the leg that wrote reports what it read back (two rows, not pack 1's six)."""
    form = _content_resume_after_an_earlier_pack(fx, rule_env, monkeypatch, tmp_path, resume_upserts=True)
    assert form["sparse_rule_readback"]["checked"] == 2


def test_2942_the_drain_content_pack_leg_forgets_the_earlier_packs_readback_before_it_writes(
        fx, rule_env, monkeypatch, tmp_path):
    """The drain's own legs (twin, then prod) clear the collection's readback before `apply_content_pack_side`.
    Fakes the transaction protocol and the apply (a stand-in that performs no upsert, a path that does not exist
    today) and records what `LAST_READBACK` holds when the apply starts."""
    import types
    from cytir.index.drain import write_content_pack as WCP
    _capture(monkeypatch)
    SR.LAST_READBACK.clear()
    PSP.upsert("http://prod", "aw_e9", [_point(r) for r in fx["rows"][:6]], batch=5)
    SR.LAST_READBACK["aw_e8"] = {"checked": 6}                  # an earlier pack's value on the twin's collection too
    assert SR.LAST_READBACK["aw_e9"]["checked"] == 6

    class RB:
        LOCKS = ("L1",)
        assert_locks_free = staticmethod(lambda task: None)
        acquire = staticmethod(lambda name, task, why, pid=None: None)
        release = staticmethod(lambda name, task: None)

    def twin_first(**kw):
        st = {}
        for role in ("twin", "prod"):
            kw["apply_target"](role, st)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q, "prod_lock_host", lambda: "a-host")
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(WCP, "TFA", types.SimpleNamespace(
        ProductionWriterLock=lambda **kw: "THE-LOCK", twin_first=twin_first))
    monkeypatch.setattr(WCP.UGS, "reverify_generation_export", lambda *a, **k: None)
    monkeypatch.setattr(WCP.PSP, "already_present", lambda u, c, ids, **k: [])
    seen = {}

    def apply_side(role, url, coll, points, tombs, **kw):
        seen[role] = (coll, SR.LAST_READBACK.get(coll))
        return {"written": 0, "tombstoned": 0, "layout_before": {}, "layout_after": {}}
    monkeypatch.setattr(Q, "apply_content_pack_side", apply_side)
    pts = [_point(r) for r in fx["rows"][6:8]]
    WCP.live_apply_content_pack(
        pack_id="p-2", manifest={"pack_id": "p-2", "collection": "aw_e8"}, twin_url="http://twin.invalid",
        prod_url="http://prod", task="t", batch=5, pause_s=0, artifact_root=tmp_path / "a", points=pts,
        tombstones=[], prod_collection="aw_e9", manifest_sha="m1")
    assert seen == {"twin": ("aw_e8", None), "prod": ("aw_e9", None)}
