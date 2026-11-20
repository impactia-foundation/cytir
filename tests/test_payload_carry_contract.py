#!/usr/bin/env python3
"""Real-record tests for `payload_carry_contract.py` -- the contract task 580 lacked.

Every fixture in `fixtures/payload_carry_contract/task580-cache-and-live.json` is a REAL
record, sealed from the build box on 2026-09-12: task 580's own materialiser payload
cache, the payload `aw_e8` holds for the same page, and the first row of task
580's roster-B materialisation -- the row it actually staged and applied.  A
hand-written dict would have proved that the module rejects what the author
imagined; these prove it rejects what shipped.

What each case exists for:

  * `carry_base` on the LIVE payload carries the five scorer-read fields and
    drops `snippet` -- the carry still works, so this is not a blanket refusal.
  * `carry_base` on the CACHED payload raises, and the message names the
    missing fields. That single call is the whole of task 580's defect.
  * the REAL materialised record task 580 wrote fails `assert_scoreable`. The
    rows that reached the collection are the population under test.
  * the four-field `with_payload` list from `roster_579.cand_points()` is
    refused AT THE SEAM, so the projection cannot be substituted silently.
  * an absent cache, an empty cache and an empty population all FAIL. A guard
    that skips on missing input reports green over unchecked ground.
  * `stage_pack_b1` CALLS the refusal, before it loads the embedder. A guard
    with a test and no caller is decoration (verification-doctrine 12).
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent

from cytir.index import payload_carry as PC  # noqa: E402

FIXTURE = json.loads(
    (HERE / "fixtures" / "payload_carry_contract" /
     "task580-cache-and-live.json").read_text(encoding="utf-8"))

LIVE = [p["live_aw_e8_payload"] for p in FIXTURE["pairs"]
        if p["live_aw_e8_payload"]]
CACHED_PROJECTION = [p["cache_580_payload"] for p in FIXTURE["pairs"]
                     if sorted(p["cache_580_payload"]) == ["source_type", "url"]]
CACHED_SYNTHESIZED = [p["cache_580_payload"] for p in FIXTURE["pairs"]
                      if p["cache_580_payload"].get("title")]


def test_fixture_is_the_two_real_shapes():
    """The fixture must actually contain the defect, or every test below is a
    test of nothing."""
    assert LIVE, "no live aw_e8 payload sealed"
    assert len(CACHED_PROJECTION) >= 2, "no projection-shaped cache entry sealed"
    assert CACHED_SYNTHESIZED, "no full cache entry sealed"
    for pl in LIVE:
        assert all(pl.get(f) is not None for f in PC.REQUIRED_CARRY)
    for pl in CACHED_PROJECTION:
        assert sorted(pl) == ["source_type", "url"]


# ----------------------------------------------------------------- carry_base
def test_carry_base_carries_the_live_payload():
    for pl in LIVE:
        base = PC.carry_base(pl, url=pl.get("url", ""))
        for f in PC.REQUIRED_CARRY:
            assert base[f] == pl[f], f"{f} not carried"
        for f in PC.NOT_CARRIED:
            assert f not in base, f"{f} must not be carried"


def test_carry_base_refuses_task_580s_cached_projection():
    pl = CACHED_PROJECTION[0]
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.carry_base(pl, url=pl["url"])
    msg = str(e.value)
    for f in ("title", "lang", "domain", "trust_tier"):
        assert f in msg, f"the refusal must name {f}"
    assert "PROJECTION" in msg


def test_carry_base_accepts_the_synthesized_fetch_payload():
    """Task 580's `--synthesize` seam composed a COMPLETE payload from the
    fetch record.  It is not the defect, and must not be refused: every one of
    those pages was dropped later, by `no_staged_point_ids`."""
    pl = CACHED_SYNTHESIZED[0]
    base = PC.carry_base(pl, url=pl["url"])
    assert base["title"] and base["trust_tier"] is not None


def test_family_required_published_for_news():
    news = dict(LIVE[0], source_type="news")
    news.pop("published", None)
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.carry_base(news, url=news.get("url", ""))
    assert "published" in str(e.value)
    ok = dict(news, published="2026-09-01")
    assert PC.carry_base(ok)["published"] == "2026-09-01"


# ------------------------------------------------------- the row that shipped
def test_the_real_row_task_580_applied_is_refused():
    rec = FIXTURE["materialized_580_record"]
    assert rec.get("title") is None and rec.get("trust_tier") is None
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_scoreable([rec], label="task580 roster-B record")
    msg = str(e.value)
    assert "title 0.0%" in msg and "trust_tier 0.0%" in msg


def test_a_live_population_passes():
    v = PC.assert_scoreable(LIVE, label="aw_e8 rows")
    assert v["verdict"] == "PASS" and v["rows_checked"] == len(LIVE)


def test_a_half_good_population_still_fails():
    rec = FIXTURE["materialized_580_record"]
    pop = LIVE * 20 + [rec] * 3           # 98.8% titled, floor is 99%
    with pytest.raises(PC.PayloadCarryError):
        PC.assert_scoreable(pop, label="half good")


# ------------------------------------------------------------- fails closed
def test_empty_population_fails():
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_scoreable([], label="nothing")
    assert "0 rows" in str(e.value)


def test_missing_cache_fails(tmp_path):
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_payload_cache(str(tmp_path / "nope.jsonl.gz"))
    assert "does not exist" in str(e.value)


def test_empty_cache_fails(tmp_path):
    p = tmp_path / "empty.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("")
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_payload_cache(str(p))
    assert "0 entries" in str(e.value)


# --------------------------------------------------------------- the cache
def _cache(tmp_path, payloads, name):
    p = tmp_path / name
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for pl in payloads:
            fh.write(json.dumps({"url": pl.get("url", "x"),
                                 "payload": pl}) + "\n")
    return str(p)


def test_task_580s_cache_shape_is_refused(tmp_path):
    path = _cache(tmp_path, CACHED_PROJECTION, "payloads.jsonl.gz")
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_payload_cache(path)
    assert "cannot carry" in str(e.value)


def test_a_full_cache_passes(tmp_path):
    path = _cache(tmp_path, LIVE, "payloads.jsonl.gz")
    assert PC.assert_payload_cache(path)["verdict"] == "PASS"


# ---------------------------------------------------------------- the seam
def test_the_projection_read_itself_is_refused():
    """The exact `with_payload` list `roster_579.cand_points()` sends."""
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_full_payload_request(
            ["url", "source_type", "anchor", "document_id"],
            where="cand_points scroll")
    assert "PROJECTION" in str(e.value)
    PC.assert_full_payload_request(True, where="harvest")   # must not raise


def test_cand_points_still_sends_the_projection_this_module_refuses():
    """A live cross-check, not a re-statement: if `roster_579.cand_points`
    is ever changed to read full payloads, this test says so, and the fixture
    above stops being the shape of a real defect."""
    roster = HERE.parent / "tasks" / "579-scripts" / "roster_579.py"
    if not roster.is_file():
        pytest.skip("the task tree is absent (public cut)")
    src = roster.read_text()
    assert '"with_payload": ["url", "source_type", "anchor",' in src


# --------------------------------------------------------------- one home
def test_the_apply_gate_and_the_stage_share_one_contract():
    from cytir.gates import gate_payload_contract as G
    assert G.CONTRACT is PC.CONTRACT


# ---------------------------------------------------------------- wired in
def test_stage_pack_b1_refuses_before_it_loads_the_embedder():
    """Not `is it importable` -- WHERE is it called.  The value of this guard
    is that it costs seconds instead of GPU-hours, and a call that drifted
    below the embedder load would still pass a naive 'is it wired' check."""
    # the package module holds main(); the old script path is only a shim (task 1528)
    origin = importlib.util.find_spec("cytir.index.stage_pack_b1").origin
    src = pathlib.Path(origin).read_text()
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    guard = embed = None
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "assert_scoreable":
            guard = node.lineno if guard is None else min(guard, node.lineno)
        if isinstance(f, ast.Name) and f.id in ("load_embedder",
                                                "tei_doc_encoder"):
            embed = node.lineno if embed is None else min(embed, node.lineno)
    assert guard is not None, \
        "stage_pack_b1.main() does not call payload_carry_contract.assert_scoreable"
    assert embed is not None, "no embedder load found -- test is stale"
    assert guard < embed, \
        f"the carry refusal (line {guard}) must come before the embedder " \
        f"load (line {embed}), or it is an acceptance check, not a gate"


def test_there_is_no_waiver_on_the_cli():
    """No `--force`, no `--allow-missing`, no `--skip`: the CLI must have no
    argument that lets the run being judged soften the standard."""
    src = (HERE.parent / "cytir" / "index" / "payload_carry.py").read_text()
    tree = ast.parse(src)
    flags = [a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "add_argument"
             for a in n.args if isinstance(a, ast.Constant)]
    assert flags, "no CLI parsed -- test is stale"
    for f in flags:
        assert not any(w in f for w in ("force", "skip", "allow", "ignore",
                                        "waive", "no-check")), \
            f"{f} is a waiver the run could grant itself"


# ------------------------------------------------- record vs serving contract
def test_record_contract_is_the_carried_subset_at_the_same_floors():
    """A pre-projection record has no `snippet` (it is in NOT_CARRIED) and its
    `country` is re-derived by `project_record`. Applying the serving contract
    to a materialised record would fail every honest pack on two fields it is
    not supposed to have yet -- and a gate that refuses correct work teaches
    its operator to reach for a waiver."""
    assert set(PC.RECORD_CONTRACT) < set(PC.CONTRACT)
    assert "snippet" not in PC.RECORD_CONTRACT
    assert "country" not in PC.RECORD_CONTRACT
    for f, floor in PC.RECORD_CONTRACT.items():
        assert PC.CONTRACT[f] == floor, f"{f} floor differs between the two"


def test_a_correctly_carried_record_passes_the_record_contract():
    """The pair that proves the split is not a loophole: the SAME real payload
    that task 580 reduced to `{url, source_type}`, carried properly, clears
    RECORD_CONTRACT without `snippet` or `country` being present."""
    rows = [dict(PC.carry_base(pl, url=pl["url"]),
                 url=pl["url"], text="x") for pl in LIVE]
    for r in rows:
        assert "snippet" not in r
    v = PC.assert_scoreable(rows, label="carried records",
                            contract=PC.RECORD_CONTRACT)
    assert v["verdict"] == "PASS"
    with pytest.raises(PC.PayloadCarryError):
        PC.assert_scoreable(rows, label="carried records")   # serving contract


def test_the_580_record_fails_the_record_contract_too():
    """The split must not let the defect through: 580's row is refused at the
    looser bar as well, on `title` and `trust_tier`."""
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_scoreable([FIXTURE["materialized_580_record"]],
                            label="task580 row", contract=PC.RECORD_CONTRACT)
    assert "title" in str(e.value) and "trust_tier" in str(e.value)


# ------------------------------------------------ country is policy-conditioned
def _row(**over):
    """A live aw_e8 payload with fields overridden -- real shape, one change."""
    row = dict(LIVE[0])
    row.pop("country", None)
    row.update(over)
    return {k: v for k, v in row.items() if v is not None}


def test_an_unstamped_class_pack_passes_without_country():
    """Task 629 B01's real shape: 445 `docs` rows on vendor hosts. The ratified
    policy stamps no country on `docs`, and 99.88% of the index's own docs rows
    carry none; a floor that refused them refused what the policy prescribes."""
    pop = ([_row(source_type="docs", domain="sunfounder.com")] * 425
           + [_row(source_type="docs", domain="upesy.com")] * 14
           + [_row(source_type="docs", domain="toybox.com")] * 6)
    v = PC.assert_scoreable(pop, label="B01 shape")
    country = next(r for r in v["fields"] if r["field"] == "country")
    assert country["present"] == 1.0
    assert "country_for_record" in country["counts"]


def test_a_location_bound_pack_missing_country_still_fails():
    """The regression the floor exists for: rows the policy WOULD stamp."""
    pop = [_row(source_type="gov-court", domain="www.bger.ch")] * 10
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_scoreable(pop, label="courts without country")
    assert "country 0.0%" in str(e.value)


def test_a_registered_card_missing_country_still_fails():
    """KEEP_EXISTING is a stamp the payload owns; its absence is a loss."""
    pop = [_row(source_type="entity_card", domain="x.ch",
                register_id="CHE-100.000.001")] * 10
    with pytest.raises(PC.PayloadCarryError):
        PC.assert_scoreable(pop, label="cards without country")


def test_only_country_is_policy_conditioned():
    """Widening this set is a change to the standard, so a test names it."""
    assert set(PC.POLICY_CONDITIONED) == {"country"}
    pop = [_row(source_type="docs", domain="sunfounder.com", title=None)] * 10
    with pytest.raises(PC.PayloadCarryError) as e:
        PC.assert_scoreable(pop, label="docs without title")
    assert "title 0.0%" in str(e.value)
