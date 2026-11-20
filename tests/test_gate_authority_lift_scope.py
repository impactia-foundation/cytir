"""Regression tests for the composite gate's authority-lift SCOPING (task 394 / 393-F5).

The defect: `gate_authority.compare` returned `0 if (zero_reg and lift_win)`
unconditionally, so a CONTENT/corpus A/B — identical flags on both arms, corpus
the only variable — could never satisfy criterion (2) and exited 1 no matter how
clean it was. Task 393 shipped 8,964 cards with zero regressions across 1,404
queries and still printed `authority-lift=FAIL`, forcing a founder ruling for
what ADR-043's content bar already answered.

The fix must be exactly two-sided, and both sides are pinned here:

  * arms DIFFER on some `AW_AUTHORITY*` var -> criterion (2) is enforced exactly
    as before. NOT weakened: it is the flood insurance for the authority layer,
    and a true A/B with no lift still FAILS.
  * arms IDENTICAL -> `authority-lift=N/A`, excluded from the verdict and the
    exit code — but the zero-regression criterion keeps its teeth (a corpus run
    that regresses a gold still exits 1).

Run: ``python3 tests/test_gate_authority_lift_scope.py`` (or via pytest).
"""
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile

from tests.helpers.repo_files import repo_file_or_skip

GATE = pathlib.Path(__file__).resolve().parent.parent / "cytir" / "gates" / "authority.py"

_spec = importlib.util.spec_from_file_location("aw_gate_authority_under_test", GATE)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


# --------------------------------------------------------------------------- #
# leg fixtures — the minimum shape `compare` reads: one class, one gold query.
# `gold_auth` >= LIFT_MIN_AUTH puts the gold in the lift pool, which is what
# makes "did the lift criterion fire?" observable at all.
# --------------------------------------------------------------------------- #

def _leg(rank, authority_env, version=G.GATE_SET_VERSION, legacy_env=None):
    leg = {
        "label": "x",
        "env": legacy_env if legacy_env is not None else {"AW_COLLECTION": "aw_b1"},
        "sets": {
            "authoritative": {
                "n": 1, "n_gold": 1,
                "h1": 1 if rank == 1 else 0,
                "h10": 1 if (rank and rank <= 10) else 0,
                "mrr": round(1.0 / rank, 4) if rank else 0.0,
                "per_q": {"g1": {"urls": [f"https://example.org/{rank}"],
                                 "rank": rank, "gold_auth": 0.9}},
            }
        },
        "cache_miss": 0,
    }
    if version is not None:
        leg["gate_set_version"] = version
    if authority_env is not None:
        leg["authority_env"] = authority_env
    return leg


@contextlib.contextmanager
def _compared(off, on):
    """Run the real `--compare` path over two leg dicts; yield (exit_code, stdout)."""
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "off.json"), os.path.join(d, "on.json")
        for p, leg in ((pa, off), (pb, on)):
            with open(p, "w") as fh:
                json.dump(leg, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = G.compare(pa, pb)
        yield code, buf.getvalue()


# --------------------------------------------------------------------------- #
# 1. TRUE A/B — the check must NOT be weakened
# --------------------------------------------------------------------------- #

def test_true_ab_without_lift_still_fails():
    """AW_AUTHORITY moved, nothing improved -> FAIL, exactly as before the fix."""
    off = _leg(3, {"AW_AUTHORITY": "0"})
    on = _leg(3, {"AW_AUTHORITY": "1", "AW_AUTHORITY_W": "0.5"})
    with _compared(off, on) as (code, out):
        assert code == 1, out
        assert "authority-lift=FAIL (no measurable lift)" in out, out
        assert "ENFORCED (true AW_AUTHORITY A/B)" in out, out
        assert "N/A" not in out.split("VERDICT")[-1], out


def test_true_ab_with_lift_passes():
    """AW_AUTHORITY moved and a lift-pool gold improved -> PASS."""
    off = _leg(3, {"AW_AUTHORITY": "0"})
    on = _leg(1, {"AW_AUTHORITY": "1"})
    with _compared(off, on) as (code, out):
        assert code == 0, out
        assert "authority-lift=PASS" in out, out


def test_a_single_differing_knob_is_enough_to_enforce():
    """AW_AUTHORITY itself equal, only a sub-knob moved -> still a true A/B."""
    off = _leg(3, {"AW_AUTHORITY": "1", "AW_AUTHORITY_W": "0.5"})
    on = _leg(3, {"AW_AUTHORITY": "1", "AW_AUTHORITY_W": "0.9"})
    with _compared(off, on) as (code, out):
        assert code == 1, out
        assert "authority-lift=FAIL" in out, out


def test_novel_authority_knob_is_captured_not_whitelisted():
    """A knob invented after this file was written must still count as a lever.

    `capture_authority_env` reads the prefix, not a hand-maintained list — the
    blind spot that would let a real A/B be silently scored as an identical pair.
    """
    env = {"AW_AUTHORITY_BRAND_NEW_KNOB": "7", "AW_HUBFLOOR": "1", "PATH": "/bin"}
    assert G.capture_authority_env(env) == {"AW_AUTHORITY_BRAND_NEW_KNOB": "7"}
    off = _leg(3, {"AW_AUTHORITY_BRAND_NEW_KNOB": "7"})
    on = _leg(3, {"AW_AUTHORITY_BRAND_NEW_KNOB": "9"})
    with _compared(off, on) as (code, out):
        assert code == 1, out
        assert "authority-lift=FAIL" in out, out


# --------------------------------------------------------------------------- #
# 2. CORPUS-ONLY — the 393 case
# --------------------------------------------------------------------------- #

def test_corpus_only_clean_run_exits_zero_with_na():
    """Identical flags, zero regressions -> exit 0 and an explicit N/A line."""
    flags = {"AW_AUTHORITY": "1", "AW_AUTHORITY_W": "0.5"}
    with _compared(_leg(3, flags), _leg(3, dict(flags))) as (code, out):
        assert code == 0, out
        assert "authority-lift=N/A (arms identical on AW_AUTHORITY)" in out, out
        assert "REPORT-ONLY, not scored" in out, out
        assert "per-class zero-regression=PASS" in out, out


def test_corpus_only_with_no_authority_flags_at_all_is_na():
    """The literal 393 shape: neither arm sets any AW_AUTHORITY* variable."""
    with _compared(_leg(3, {}), _leg(3, {})) as (code, out):
        assert code == 0, out
        assert "authority-lift=N/A" in out, out


def test_corpus_only_still_fails_on_a_regression():
    """N/A must not make the gate toothless — zero-regression keeps its teeth."""
    flags = {"AW_AUTHORITY": "1"}
    with _compared(_leg(2, flags), _leg(7, dict(flags))) as (code, out):
        assert code == 1, out
        assert "per-class zero-regression=FAIL" in out, out
        assert "authority-lift=N/A" in out, out
        assert "RANK-REG" in out, out


# --------------------------------------------------------------------------- #
# 3. Pre-v2 legs (task 318/365/393 artifacts) stay replayable
# --------------------------------------------------------------------------- #

_LEGACY_KEYS = {k: None for k in G._LEGACY_AUTHORITY_KEYS}


def _legacy_leg(rank, **auth):
    env = dict(_LEGACY_KEYS, AW_COLLECTION="aw_b1", **auth)
    return _leg(rank, None, version=None, legacy_env=env)


def test_pre_v2_identical_legs_replay_to_na():
    """No version stamp, no authority_env: the 393 artifact shape -> N/A, exit 0."""
    with _compared(_legacy_leg(3), _legacy_leg(3)) as (code, out):
        assert code == 0, out
        assert "authority-lift=N/A" in out, out
        assert "pre-v2 artifacts" in out, out
        assert "legacy 6-key whitelist" in out, out


def test_pre_v2_true_ab_still_enforces_lift():
    """A pre-v2 A/B is still readable as an A/B through the legacy whitelist."""
    off = _legacy_leg(3, AW_AUTHORITY="0")
    on = _legacy_leg(3, AW_AUTHORITY="1")
    with _compared(off, on) as (code, out):
        assert code == 1, out
        assert "authority-lift=FAIL" in out, out


def test_cross_version_compare_is_refused():
    """Two legs scored on different yardsticks are not a regression signal."""
    with _compared(_legacy_leg(3), _leg(3, {})) as (code, out):
        assert code == 3, out
        assert "gate-set version mismatch" in out, out


def test_gate_set_version_is_stamped_and_v4():
    # v4 (task 602, 2026-09-17, ADR-107 am.3): the 24 contaminated freshness
    # rows leave data/eval_authoritative.yaml (478 -> 454). A REMOVAL, so it is
    # a version bump and not a correction -- see gate_authority.py's
    # GATE_SET_VERSION table. v3 (task 531) was the rfc-editor.org gold_url fix.
    assert G.GATE_SET_VERSION == "v4"
    names = [n for n, _ in G.SETS]
    assert "entity_card_golds" in names, names
    assert "entity_cards" in names, "the task-163 return-band set must survive"


def test_the_v4_row_set_is_the_one_the_version_claims():
    """The version number is only worth what the set behind it counts.

    1,396 rows / 1,065 gold-bearing is the v4 population; a change that moves
    it without moving the version is the failure the stamp exists to prevent.
    """
    repo_file_or_skip("data/eval_authoritative.yaml")   # the withheld gold sets
    rows = gold = 0
    for _name, path in G.SETS:
        if os.path.exists(path):
            for r in G.load_rows(path):
                rows += 1
                gu, gd = G.gold_of(r)
                if gu or gd:
                    gold += 1
    if os.path.exists(G.SLICEB):
        for r in G.load_sliceb(G.SLICEB):
            rows += 1
            gu, gd = G.gold_of(r)
            if gu or gd:
                gold += 1
    assert (rows, gold) == (1396, 1065), (
        f"the scored population is {rows} rows / {gold} gold-bearing; v4 was "
        "sealed at 1,396 / 1,065. Moving it is a GATE_SET_VERSION bump "
        "(gate_authority.py's table), never a silent edit.")


# --------------------------------------------------------------------------- #
# 4. The v2 gold bundle itself
# --------------------------------------------------------------------------- #

def test_entity_card_golds_load_and_are_url_golded():
    repo_file_or_skip("data/eval_entity_card_golds.yaml")
    path = dict(G.SETS)["entity_card_golds"]
    rows = G.load_rows(path)
    assert 12 <= len(rows) <= 20, f"gold count out of the adjudicated band: {len(rows)}"
    seen = set()
    for r in rows:
        gu, gd = G.gold_of(r)
        assert gu, f"{r.get('id')} has no gold url — it would score as gold=0"
        assert r.get("adjudication") in ("unique", "strong"), r.get("id")
        assert r.get("provenance") in ("intent-paraphrase", "site-verbatim"), r.get("id")
        assert r.get("notes"), f"{r.get('id')} has no adjudication rationale"
        assert r["query"] not in seen, f"duplicate query {r['query']!r}"
        seen.add(r["query"])


def test_entity_card_golds_never_name_their_own_company():
    """A gold that contains the company name tests name-matching, not the
    activity channel this set exists to measure (the ADR-060 taxonomy case is
    precisely that cards match name tokens and nothing else)."""
    repo_file_or_skip("data/eval_entity_card_golds.yaml")
    rows = G.load_rows(dict(G.SETS)["entity_card_golds"])
    for r in rows:
        q = r["query"].lower()
        for dom in r.get("gold_domains") or []:
            stem = dom.split(".")[0].replace("-", "")
            assert stem not in q.replace(" ", "").replace("-", ""), \
                f"{r['id']}: query names its own target ({stem})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all {len(fns)} gate lift-scope tests passed")
