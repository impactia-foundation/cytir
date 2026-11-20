#!/usr/bin/env python3
"""Task 548 / 1433 -- the curation register decides a source's stamps, not a flag.

`data/allowlist.yaml` declared `impactia.org` as `tier: D` -- "Authoritative
only about itself" -- while the twin served that host's 145 rows as
`source_type: official`, `trust_tier: 1.0`, because the small-site ingester
(retired, task 1433) defaulted to `--tier A --source-type official --trust 1.0`
and nothing compared the two (task 548).

These cases assert only the MECHANISM of `cytir.core.common.resolve_register_stamps`:
the register is read, a contradiction is refused, an override has to say why.
They assert nothing about which tier impactia.org OUGHT to hold -- that is the
founder's ruling. The cases are the retired module's own test cases, re-pointed
(same inputs, same expected values), plus one parity case pinned from the
retired implementation.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import cytir.core.common as AC

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
ALLOWLIST = REPO / "data" / "allowlist.yaml"
# ADR-104 (2026-09-08) moved impactia.org to tier A, so the live register no longer
# holds a tier-D entry; the contradiction tests use a sealed tier-D fixture instead.
TIER_D_FIXTURE = HERE / "testdata" / "task548_tier_d_register.yaml"
TIER_D_DOMAIN = "tier-d-fixture.example"


def stamps(domain, *, allowlist=str(ALLOWLIST), tier=None, source_type=None,
           trust=None, override_register=""):
    return AC.resolve_register_stamps(
        domain, allowlist=allowlist, tier=tier, source_type=source_type,
        trust=trust, override_reason=override_register)


# --- one home for the table --------------------------------------------------

def test_the_tier_table_has_exactly_one_home():
    """It shipped as three byte-identical copies (`fetch_corpus`, `cc_ingest`,
    `fetch_hn_cc`). Identical today is not parity, so this is a SOURCE scan of
    every Python module in the repo outside tests/ and tasks/ rather than an
    identity check on named imports: a fourth copy in a module nobody thought
    to list is the failure this exists to catch."""
    skip = {"tests", "tasks", ".git", ".claude", "node_modules", "venv", ".venv",
            "build"}  # setuptools build/lib copy made by the cut rehearsal's install
    redefiners = []
    for path in sorted(REPO.glob("*/**/*.py")):
        rel = path.relative_to(REPO)
        if skip & set(rel.parts) or rel.as_posix() == "cytir/core/common.py":
            continue                       # cytir/core/common.py: the one legal home
        if path.name == "aw_common.py":
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in (
                        "TIER_TRUST", "TIER_SOURCE"):
                    redefiners.append(f"{rel}:{node.lineno} {tgt.id}")
    assert redefiners == [], (
        "the allowlist tier table is defined outside cytir/core/common.py: "
        f"{redefiners}. Import it instead (engineering doctrine 1).")


def test_tier_stamps_is_the_registers_own_mapping():
    assert AC.tier_stamps("D") == {"tier": "D", "trust_tier": 0.5,
                                   "source_type": "entity-self"}
    assert AC.tier_stamps("A") == {"tier": "A", "trust_tier": 1.0,
                                   "source_type": "official"}


# --- the register is read ----------------------------------------------------

def test_the_registered_entry_is_what_ships():
    tier, st, trust, basis = stamps("impactia.org")
    reg = AC.allowlist_entry("impactia.org", str(ALLOWLIST))
    assert reg["tier"] == tier            # whatever the register says today
    assert (st, trust) == (AC.TIER_SOURCE[reg["tier"]],
                           AC.TIER_TRUST[reg["tier"]])
    assert "register" in basis and str(ALLOWLIST) in basis


def test_www_prefix_does_not_hide_the_entry():
    """A join that misses is a join that reports 'unregistered' and then
    stamps whatever the caller wanted (corpus doctrine 5)."""
    assert AC.allowlist_entry("www.impactia.org", str(ALLOWLIST)) is not None
    assert AC.allowlist_entry("IMPACTIA.ORG", str(ALLOWLIST)) is not None


# --- a contradiction is refused ---------------------------------------------

def test_the_historical_invocation_is_now_refused():
    """`--tier A --source-type official --trust 1.0` on a tier-D register entry
    -- the command line behind the twin's stamps."""
    with pytest.raises(SystemExit) as e:
        stamps(TIER_D_DOMAIN, tier="A", source_type="official", trust=1.0,
               allowlist=str(TIER_D_FIXTURE))
    msg = str(e.value)
    assert "REFUSED" in msg and "register" in msg
    assert "--override-register" in msg


def test_an_override_needs_a_reason_and_the_reason_is_carried():
    tier, st, trust, basis = stamps(
        TIER_D_DOMAIN, tier="A", source_type="official", trust=1.0,
        allowlist=str(TIER_D_FIXTURE),
        override_register="task 548 acceptance test")
    assert (tier, st, trust) == ("A", "official", 1.0)
    assert basis.startswith("OVERRIDE") and "task 548 acceptance test" in basis


def test_restating_the_register_is_not_an_override():
    reg = AC.allowlist_entry("impactia.org", str(ALLOWLIST))
    tier, st, trust, basis = stamps(
        "impactia.org", tier=reg["tier"],
        source_type=AC.TIER_SOURCE[reg["tier"]],
        trust=AC.TIER_TRUST[reg["tier"]])
    assert not basis.startswith("OVERRIDE")


def test_an_unregistered_domain_with_no_stamps_is_refused():
    """Silently defaulting an unknown host to `official` / 1.0 is how a source
    admits itself at the top tier without anyone deciding."""
    with pytest.raises(SystemExit) as e:
        stamps("not-in-the-register.example")
    assert "REFUSED" in str(e.value)


def test_an_unregistered_domain_with_explicit_stamps_says_so():
    tier, st, trust, basis = stamps(
        "not-in-the-register.example", tier="C", source_type="docs", trust=0.72)
    assert (tier, st, trust) == ("C", "docs", 0.72)
    assert "not in the register" in basis


# --- parity with the retired implementation ---------------------------------
# Expected values computed ONCE by running `resolve_stamps` from the retired
# ingester (git blob 1d4ccc5bff9c7a0083f407097cf4a3e07b8ebbe7 on origin/main,
# 2026-10-30) against this allowlist. {AL} stands for the allowlist path.
_PARITY_ALLOWLIST = """allowlist:
- domain: a.example
  tier: A
- domain: b.example
  tier: B
- domain: c.example
  tier: C
- domain: d.example
  tier: D
"""
_REFUSE_UNREG = ("REFUSED: {d} has no entry in {AL} and no "
                 "--tier/--source-type/--trust was given. An unregistered "
                 "source has no curation tier to stamp; add it to the register "
                 "(a PR Tim merges) or state the stamps explicitly.")
_PARITY = [
    ("a.example", {}, ("A", "official", 1.0, "register {AL} tier=A")),
    ("www.B.example", {}, ("B", "reference", 0.82, "register {AL} tier=B")),
    ("c.example", {}, ("C", "docs", 0.72, "register {AL} tier=C")),
    ("d.example", {}, ("D", "entity-self", 0.5, "register {AL} tier=D")),
    ("d.example", dict(tier="D", source_type="entity-self", trust=0.5),
     ("D", "entity-self", 0.5, "register {AL} tier=D (restated)")),
    ("d.example", dict(tier="A", source_type="official", trust=1.0,
                       override_register="why"),
     ("A", "official", 1.0, "OVERRIDE of register tier=D: why")),
    ("d.example", dict(trust=0.9, override_register="r2"),
     ("D", "entity-self", 0.9, "OVERRIDE of register tier=D: r2")),
    ("x.example", dict(tier="C", source_type="docs", trust=0.72),
     ("C", "docs", 0.72, "explicit (domain not in the register)")),
    ("x.example", dict(trust=0.3),
     ("A", "official", 0.3, "explicit (domain not in the register)")),
    ("x.example", {}, "EXIT:" + _REFUSE_UNREG.replace("{d}", "x.example")),
    ("d.example", dict(tier="A"),
     "EXIT:REFUSED: d.example is registered as tier D (entity-self / 0.5) and "
     "this invocation contradicts it: {{'tier': ('D', 'A')}}. The register is "
     "changed by a PR Tim merges, not by a command line. Pass "
     "--override-register REASON to stamp differently anyway, and the reason "
     "goes in the receipt."),
]


@pytest.mark.parametrize("domain,kw,want", _PARITY)
def test_parity_with_the_retired_resolve_stamps(tmp_path, domain, kw, want):
    al = tmp_path / "al.yaml"
    al.write_text(_PARITY_ALLOWLIST)
    if isinstance(want, str):
        with pytest.raises(SystemExit) as e:
            stamps(domain, allowlist=str(al), **kw)
        assert "EXIT:" + str(e.value) == want.format(AL=str(al))
    else:
        got = stamps(domain, allowlist=str(al), **kw)
        assert got == tuple(w.format(AL=str(al)) if isinstance(w, str) else w
                            for w in want)
