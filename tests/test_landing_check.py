"""`cytir.ops.landing_check` (task 701): the landing predicate on a receipt shaped like the real t3 delta receipt."""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.ops import landing_check as L  # noqa: E402

TWIN, PROD = ("http://t", "ct"), ("http://p", "cp")


def _stage(tmp_path, n=10):
    d = tmp_path / "chunks"
    d.mkdir()
    with gzip.open(d / "shard-0000.texts.jsonl.gz", "wt") as fh:
        for i in range(n):
            fh.write(json.dumps({"point_id": f"p{i}", "domain": "a.example" if i < 7 else "b.example"}) + "\n")
    return d


def _receipt(tmp_path, status="LANDED", written=8):
    # the real t3 receipt shape: rows.vectors / projected / excluded_out_of_scope_language, written per side
    side = {"rows": {"vectors": 10, "projected": 8, "skipped_by_projection": 0, "excluded_undated": 0,
                     "excluded_out_of_scope_language": 2, "already_present": 0}, "written": written}
    q = tmp_path / "q" / "pk"
    q.mkdir(parents=True)
    (q / "receipt.json").write_text(json.dumps(
        {"status": status, "mechanical_content": {"word_floor_dropped": 0}, "twin": side, "prod": side}))
    return tmp_path / "q"


class Fake:
    def __init__(self, present, count=1000):
        self.present, self.count = set(present), count

    def install(self, monkeypatch):
        monkeypatch.setattr(L, "points_count", lambda base, c: self.count)
        monkeypatch.setattr(L, "served", lambda base, c, ids: sum(1 for i in ids if i in self.present))


def _ex(tmp_path):
    f = tmp_path / "ex.jsonl"
    f.write_text("".join(json.dumps({"point_id": i}) + "\n" for i in ("p8", "p9")))
    return str(f)


def test_a_landed_balanced_pack_passes(tmp_path, monkeypatch):
    Fake([f"p{i}" for i in range(8)]).install(monkeypatch)
    res = L.check_pack(str(_receipt(tmp_path)), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD,
                       exclude_file=_ex(tmp_path))
    assert res["ok"], res
    assert res["accounting"]["twin"] == {"vectors": 10, "written": 8, "skipped_by_projection": 0,
                                         "excluded_undated": 0, "excluded_out_of_scope_language": 2,
                                         "already_present": 0, "word_floor_dropped": 0, "balanced": True}


def test_an_excluded_id_that_serves_fails_the_check(tmp_path, monkeypatch):
    Fake([f"p{i}" for i in range(9)]).install(monkeypatch)  # p8 is on twin and prod
    res = L.check_pack(str(_receipt(tmp_path)), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD,
                       exclude_file=_ex(tmp_path))
    assert not res["ok"] and res["excluded_found_twin"] == 1 and not res["checks"]["excluded_absent_prod"]


def test_rows_that_vanish_unnamed_fail_the_accounting(tmp_path, monkeypatch):
    Fake([f"p{i}" for i in range(8)]).install(monkeypatch)
    res = L.check_pack(str(_receipt(tmp_path, written=7)), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD,
                       exclude_file=_ex(tmp_path))
    assert not res["checks"]["accounted"] and not res["ok"]


def test_a_pack_without_a_landed_receipt_fails(tmp_path, monkeypatch):
    Fake([]).install(monkeypatch)
    res = L.check_pack(str(_receipt(tmp_path, status="REFUSED")), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD)
    assert not res["checks"]["receipt_landed"] and not res["ok"]
    assert not L.check_pack(str(tmp_path / "none"), "x", "unused", twin=TWIN, prod=PROD)["ok"]


def test_domains_flag_a_domain_that_serves_less_than_it_sealed(tmp_path, monkeypatch):
    serving = {"a.example": (7, 7), "b.example": (2, 3)}
    monkeypatch.setattr(L, "domain_count", lambda base, c, d: serving[d][0 if base == "http://t" else 1])
    res = L.check_domains(str(_stage(tmp_path)), twin=TWIN, prod=PROD)
    by = {r["domain"]: r for r in res["top"]}
    assert not by["a.example"]["check"] and by["b.example"]["check"] and not res["ok"]


def test_an_empty_stage_refuses(tmp_path):
    import pytest
    (tmp_path / "e").mkdir()
    with pytest.raises(SystemExit, match="REFUSED"):
        L.stage_ids(tmp_path / "e")


# ------------------------------------------------ served exclusions are PRESENT, others ABSENT (#572 M1)
SERVED = "already_on_twin_and_prod"


def _rows_file(tmp_path, rows):
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(f)


def _check(tmp_path, monkeypatch, present, rows):
    Fake(present).install(monkeypatch)
    return L.check_pack(str(_receipt(tmp_path)), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD,
                        exclude_file=_rows_file(tmp_path, rows))


def test_served_exclusions_must_be_present_on_both_sides(tmp_path, monkeypatch):
    rows = [{"point_id": "p8", "reason": SERVED}, {"point_id": "p9", "reason": SERVED}]
    res = _check(tmp_path, monkeypatch, [f"p{i}" for i in range(10)], rows)
    assert res["ok"], res
    assert (res["served_excluded_in_file"], res["served_excluded_found_twin"],
            res["served_excluded_found_prod"], res["excluded_other_in_file"]) == (2, 2, 2, 0)


def test_a_served_exclusion_that_is_missing_fails(tmp_path, monkeypatch):
    rows = [{"point_id": "p8", "reason": SERVED}, {"point_id": "p9", "reason": SERVED}]
    res = _check(tmp_path, monkeypatch, [f"p{i}" for i in range(9)], rows)   # p9 is nowhere
    assert not res["ok"] and not res["checks"]["served_excluded_present_twin"]
    assert res["served_excluded_found_twin"] == 1


def test_a_mixed_file_applies_each_rule_to_its_own_group(tmp_path, monkeypatch):
    rows = [{"point_id": "p8", "reason": SERVED}, {"point_id": "p9", "reason": "host_hijacked"}]
    ok = _check(tmp_path, monkeypatch, [f"p{i}" for i in range(9)], rows)    # p8 there, p9 absent
    assert ok["ok"], ok
    assert (ok["served_excluded_in_file"], ok["excluded_other_in_file"]) == (1, 1)
    assert ok["excluded_found_twin"] == 0
    (tmp_path / "second").mkdir()
    bad = _check(tmp_path / "second", monkeypatch, [f"p{i}" for i in range(10)], rows)  # spam id served
    assert not bad["ok"] and bad["excluded_found_twin"] == 1 and bad["served_excluded_found_twin"] == 1


def test_a_row_without_a_reason_keeps_the_absent_rule(tmp_path, monkeypatch):
    res = _check(tmp_path, monkeypatch, [f"p{i}" for i in range(9)], [{"point_id": "p8"}, {"point_id": "p9"}])
    assert not res["ok"] and res["excluded_found_twin"] == 1 and res["served_excluded_in_file"] == 0


def test_the_real_tranche_6_ids_are_a_served_group(tmp_path, monkeypatch):
    real = json.loads((HERE / "testdata" / "task702_tranche6_served_203.json").read_text())
    ids = sorted(real)
    assert len(ids) == 203
    f = _rows_file(tmp_path, [{"point_id": i, "reason": SERVED} for i in ids])
    held = set(ids) | {"p0", "p1"}
    monkeypatch.setattr(L, "points_count", lambda base, c: 1000)
    monkeypatch.setattr(L, "served", lambda base, c, q: sum(1 for i in q if i in held))
    res = L.check_pack(str(_receipt(tmp_path)), "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD,
                       exclude_file=f, sample=2)
    assert res["served_excluded_in_file"] == 203 and res["served_excluded_found_prod"] == 203
    assert res["checks"]["served_excluded_present_twin"] and res["checks"]["excluded_absent_twin"]


# --- task 760: the sample leaves out the rows the content check dropped (shard59 of task 749) ---------------

def _receipt_with_floor_drops(tmp_path, *, failing, named, written):
    """The shape of `us-federal-register-ecfr-v1-shard59/receipt.json`: `mechanical_content` carries
    `word_floor_dropped`, `failing_rows` and a `problem_sample` of the first 20 problems; the written count
    leaves those rows out."""
    side = {"rows": {"vectors": 10, "skipped_by_projection": 0, "excluded_undated": 0,
                     "excluded_out_of_scope_language": 0, "already_present": 0}, "written": written}
    q = tmp_path / "q" / "pk"
    q.mkdir(parents=True)
    mech = {"schema": "aw.mechanical-content-check.v1", "rows_scored": 10, "failing_rows": failing,
            "word_floor_dropped": failing, "verdict": "PASS",
            "problem_sample": {f"p{i}": [f"text below word floor ({9 + i} < 15)"] for i in named}}
    (q / "receipt.json").write_text(json.dumps({"status": "LANDED", "mechanical_content": mech,
                                                "mechanical_content_dropped": failing, "twin": side, "prod": side}))
    return tmp_path / "q"


def test_word_floor_drops_are_not_sampled(tmp_path, monkeypatch):
    # 10 staged, p6..p9 dropped by the word floor and never written: a correct landing
    Fake([f"p{i}" for i in range(6)]).install(monkeypatch)
    res = L.check_pack(str(_receipt_with_floor_drops(tmp_path, failing=4, named=range(6, 10), written=6)),
                       "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD, sample=100)
    assert res["checks"]["sample_served_twin"] and res["checks"]["sample_served_prod"], res
    assert res["ok"], res
    assert res["content_dropped_named"] == 4 and res["sample_unnamed_content_drops_tolerated"] == 0


def test_the_old_sample_would_have_failed_this_landing(tmp_path, monkeypatch):
    """Guard: the sample WITHOUT the exclusion misses the 4 dropped ids (the bug of task 749)."""
    Fake([f"p{i}" for i in range(6)]).install(monkeypatch)
    ids = [i for i, _ in L.stage_ids(_stage(tmp_path))]
    assert L.served("b", "c", ids) == 6          # 4 of the 10 staged ids are not on the target


def test_a_missing_sampled_row_the_receipt_does_not_name_still_fails(tmp_path, monkeypatch):
    Fake([f"p{i}" for i in range(5)]).install(monkeypatch)       # p5 is missing and is not a content drop
    res = L.check_pack(str(_receipt_with_floor_drops(tmp_path, failing=4, named=range(6, 10), written=6)),
                       "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD, sample=100)
    assert not res["checks"]["sample_served_twin"] and not res["ok"]


def test_drops_beyond_the_receipts_20_named_are_tolerated_and_printed(tmp_path, monkeypatch):
    # the receipt names 2 of 4 drops; the other 2 may be in the sample
    Fake([f"p{i}" for i in range(6)]).install(monkeypatch)
    res = L.check_pack(str(_receipt_with_floor_drops(tmp_path, failing=4, named=[6, 7], written=6)),
                       "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD, sample=100)
    assert res["sample_unnamed_content_drops_tolerated"] == 2 and res["checks"]["sample_served_twin"]


def test_more_misses_than_unnamed_drops_fail(tmp_path, monkeypatch):
    Fake([f"p{i}" for i in range(3)]).install(monkeypatch)       # 3 missing sampled rows beyond the 2 named drops
    res = L.check_pack(str(_receipt_with_floor_drops(tmp_path, failing=4, named=[6, 7], written=6)),
                       "pk", str(_stage(tmp_path)), twin=TWIN, prod=PROD, sample=100)
    assert not res["checks"]["sample_served_twin"]
