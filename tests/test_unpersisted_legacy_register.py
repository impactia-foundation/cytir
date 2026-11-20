#!/usr/bin/env python3
"""[test] ADR-105 / task 581 -- the debt is declared, or the suite fails.

The guard in `cytir.corpus.persisted` refuses a row whose page is not on
disk. That protects the FUTURE. This suite is about the PAST: production
already holds rows whose inputs nobody kept, and the failure mode this project
keeps repeating is that such a population stops being mentioned and therefore
stops existing (corpus-doctrine 2).

So: every provenance family production holds must either resolve entirely to a
persisted store, or appear in `data/unpersisted-legacy-register.v1.json` with a
date and a disposition. A family that quietly acquires unpersisted pages fails
this suite; a register entry that outlives its debt fails it too.

The register grants nothing. `persisted_inputs.py` never reads it, and no
applier is relaxed by an entry in it -- asserted below, because a debt register
that turns into a waiver list is worse than no register (verification-doctrine
12).

CI has no build host, so the instrument is the tracked census
`data/prod-persisted-input-census.v1.json`, produced by
the task-581 scripts prod_provenance_census, unpersisted_census and seed_register
and re-runnable from the sealed twin snapshot.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

import cytir.corpus.persisted as PI  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
# The public cut withholds both files; the whole module skips there (task 2411).
CENSUS_PATH = repo_file_or_skip("data/prod-persisted-input-census.v1.json")
REGISTER_PATH = repo_file_or_skip("data/unpersisted-legacy-register.v1.json")

DEBT_TERMINALS = ("none", "l2_only")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(scope="module")
def census():
    assert CENSUS_PATH.exists(), (
        f"{CENSUS_PATH.name} is missing. It is the instrument this suite "
        "reads; without it the register cannot be checked against anything, "
        "and an unchecked register is a list of claims.")
    doc = json.loads(CENSUS_PATH.read_text())
    assert doc["schema"] == "aw.prod-persisted-input-census.v1"
    return doc


@pytest.fixture(scope="module")
def register():
    return PI.load_register(str(REGISTER_PATH))


# -- the register describes the census, exactly ----------------------------

def test_every_family_with_unpersisted_pages_is_registered(census, register):
    reg = {e["provenance"] for e in register["families"]}
    missing = [f["provenance"] for f in census["families"]
               if any(f["pages_by_terminal"].get(t) for t in DEBT_TERMINALS)
               and f["provenance"] not in reg]
    assert missing == [], (
        f"{len(missing)} provenance family/families hold production pages with "
        f"no persisted L1 and are not in {REGISTER_PATH.name}: {missing[:10]}. "
        "Re-run the task-581 seed_register script, or persist the pages. "
        "A family that acquires this debt silently is the whole failure this "
        "register exists to stop.")


def test_no_register_entry_outlives_its_debt(census, register):
    by_fam = {f["provenance"]: f for f in census["families"]}
    stale = []
    for e in register["families"]:
        f = by_fam.get(e["provenance"])
        if f is None:
            stale.append((e["provenance"], "not in the census at all"))
        elif not any(f["pages_by_terminal"].get(t) for t in DEBT_TERMINALS):
            stale.append((e["provenance"], "census shows it fully persisted"))
    assert stale == [], (
        f"{len(stale)} register entry/entries no longer describe a debt: "
        f"{stale[:10]}. Remove them -- a register nobody prunes stops being "
        "read.")


def test_the_register_and_the_census_agree_on_the_numbers(census, register):
    by_fam = {f["provenance"]: f for f in census["families"]}
    for e in register["families"]:
        f = by_fam[e["provenance"]]
        assert e["pages_none"] == f["pages_by_terminal"].get("none", 0)
        assert e["points_none"] == f["points_by_terminal"].get("none", 0)
        assert e["pages_l2_only"] == f["pages_by_terminal"].get("l2_only", 0)
        assert e["points_l2_only"] == f["points_by_terminal"].get("l2_only", 0)


def test_register_totals_are_the_sum_of_its_entries(register):
    t = register["totals"]
    for key in ("pages_none", "points_none", "pages_l2_only", "points_l2_only"):
        assert t[key] == sum(e[key] for e in register["families"]), key
    assert t["families"] == len(register["families"])


def test_every_entry_is_dated_and_has_a_disposition(register):
    for e in register["families"]:
        assert DATE.match(e["registered_on"]), e
        assert e["disposition"] in {"refetch", "chain_adr093",
                                    "refetch_and_chain"}, e
        # a disposition must match the debt it claims to describe
        if e["disposition"] == "refetch":
            assert e["pages_none"] and not e["pages_l2_only"], e
        if e["disposition"] == "chain_adr093":
            assert e["pages_l2_only"] and not e["pages_none"], e
        if e["disposition"] == "refetch_and_chain":
            assert e["pages_none"] and e["pages_l2_only"], e


def test_the_work_list_is_sized_not_just_named(register):
    wl = register["work_list"]
    assert set(wl) >= {"refetch", "chain_adr093", "promote_unpromoted_stores"}
    up = register["unpromoted_stores"]
    assert DATE.match(up["measured_on"])
    assert up["instrument"].endswith("store_discovery_581.py")
    assert up["by_top_level"], "the unpromoted-store walk reported nothing"
    assert up["largest"], "no unpromoted store was named"


# -- the register is not a waiver ------------------------------------------

def test_persisted_inputs_never_consults_the_register_when_judging_a_row():
    """A debt register that becomes a bypass list is worse than no register."""
    src = (REPO / "cytir" / "corpus" / "persisted.py").read_text()
    body = src.split("def audit_pages", 1)[1].split("# ----", 1)[0]
    for name in ("load_register", "registered_families", "REGISTER_SCHEMA",
                 "DEFAULT_REGISTER"):
        assert name not in body, (
            f"audit_pages references {name}: the verdict on a row must not "
            "depend on whether somebody wrote its family into the register.")


def test_no_applier_imports_the_register():
    appliers = ["../cytir/index/pack_delta.py", "../cytir/index/surface_delta.py",
                "freshness_ingest.py",
                "weekly_delta.py", "../cytir/index/apply_delta.py"]
    for rel in appliers:
        p = REPO / "pipeline" / rel
        if not p.exists():
            continue
        src = p.read_text()
        assert "load_register" not in src and \
            "unpersisted-legacy-register" not in src, rel


# -- the census itself has to be honest about its instrument ---------------

def test_the_census_names_its_instrument_and_its_holes(census):
    inst = census["instrument"]
    assert inst["snapshot"], "the census does not say what it measured"
    assert DATE.match(census["measured_on"])
    assert inst["index_built_at"]
    assert inst["store_roster"], "the store roster is not recorded"
    # An index built over unreadable files biases toward REFUSING, and the
    # count travels so nobody reads a refusal as proof of absence.
    assert "index_unreadable_files" in inst


def test_the_census_terminals_conserve(census):
    tot = census["prod_pages_by_terminal"]
    assert sum(tot.values()) == census["prod_pages"]
    fam_pages = sum(f["pages"] for f in census["families"])
    assert fam_pages == census["prod_pages"]
    for f in census["families"]:
        assert sum(f["pages_by_terminal"].values()) == f["pages"], f["provenance"]
        assert sum(f["points_by_terminal"].values()) == f["points"], f["provenance"]
