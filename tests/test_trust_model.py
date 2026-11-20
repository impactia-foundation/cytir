#!/usr/bin/env python3
"""Task 440 — tests for earned-vs-inherited trust.

Every test in the first group FAILS against the pre-439 code, where
`trust_tier = CLASS_TRUST[cls]` regardless of how the class was reached. That is
the point: they discriminate the fix from the defect rather than restating it.

Run: python -m pytest tests/test_trust_model.py -q
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.acquire import prep_discovery_queue as pdq  # noqa: E402
from cytir.acquire.trust_model import (ENDORSED, INHERITED_CEILING, INHERITED_FLOOR,  # noqa: E402
                         SELF_SIGNAL, inherited_trust, resolve_trust)

# discovery_ingest pulls httpx/trafilatura; only its table is needed here.
CLASS_TRUST = {"gov": 0.85, "health": 0.82, "edu": 0.8, "ref": 0.78,
               "tech": 0.75, "news": 0.6, "other": 0.55, "commercial": 0.5}
MISSION_CLASSES = ("gov", "health", "edu", "ref", "tech")


# --- group 1: these FAIL against the pre-439 inheritance --------------------

@pytest.mark.parametrize("cls", MISSION_CLASSES)
def test_single_endorsement_never_confers_mission_trust(cls):
    """THE DEFECT. One inbound link from one trusted seed used to grant the full
    mission tier -- task 203's heathrow.com at edu@0.8. It now lands exactly on
    POLICY v2 §4.4's vendor-content class."""
    got = resolve_trust(CLASS_TRUST[cls], ENDORSED, 1)
    assert got == INHERITED_FLOOR, (cls, got)
    assert got < CLASS_TRUST[cls]


@pytest.mark.parametrize("cls", MISSION_CLASSES)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 100, 10_000])
def test_inherited_trust_never_reaches_self_signal_trust(cls, n):
    """Hearsay is never priced at the credential's value, at ANY corroboration.
    The ceiling is structural, not a tuned threshold: no number of inbound links
    turns 'people link this' into 'this host is a primary authority'."""
    assert resolve_trust(CLASS_TRUST[cls], ENDORSED, n) < CLASS_TRUST[cls]


def test_corroboration_raises_trust_monotonically():
    """Pre-439 this sequence was constant. Independent endorsers must compound,
    the way ENDORSER_HOST_CAP's comment already says they should."""
    seq = [inherited_trust(n) for n in (1, 2, 4, 8, 16)]
    assert seq == sorted(seq)
    assert seq[0] < seq[-1]
    assert seq[0] == INHERITED_FLOOR and seq[-1] == INHERITED_CEILING


def test_inherited_band_sits_below_every_mission_tier():
    assert INHERITED_CEILING < min(CLASS_TRUST[c] for c in MISSION_CLASSES)


def test_missing_basis_is_treated_as_endorsed():
    """A queue built before 439 carries no basis field. The safe reading of a
    missing credential is that there is none -- never the mission tier."""
    assert resolve_trust(CLASS_TRUST["gov"], None, None) == INHERITED_FLOOR


# --- group 2: non-regression -- what must NOT change ------------------------

@pytest.mark.parametrize("cls", MISSION_CLASSES)
def test_self_signal_trust_is_untouched(cls):
    """A credential in the hostname still earns the full tier. The fix removes
    unearned trust; it must not disturb earned trust."""
    assert resolve_trust(CLASS_TRUST[cls], SELF_SIGNAL, 0) == CLASS_TRUST[cls]


@pytest.mark.parametrize("cls", ["news", "other", "commercial"])
def test_endorsement_never_promotes_a_sub_band_class(cls):
    """news 0.6 / other 0.55 / commercial 0.5 already sit at or below the band.
    Being linked must not raise them into it."""
    assert resolve_trust(CLASS_TRUST[cls], ENDORSED, 50) <= CLASS_TRUST[cls]


# --- group 3: the basis + corroboration signals -----------------------------

def test_class_basis_reads_the_branch_that_fired():
    assert pdq.class_basis("gov TLD/keyword") == SELF_SIGNAL
    assert pdq.class_basis("edu/research keyword") == SELF_SIGNAL
    assert pdq.class_basis("unclassified") == SELF_SIGNAL
    assert pdq.class_basis("endorsed-by universities-research seed(s)") == ENDORSED


def test_corroboration_counts_distinct_registrables_not_links():
    """One endorser's many hosts are one voice. Pre-439 nothing counted at all."""
    endorse = {"edges": {
        "cs.uni-a.example": {"class": "universities-research",
                             "targets": {"http://t.example/1": 500,
                                         "http://t.example/2": 500}},
        "lib.uni-a.example": {"class": "universities-research",
                              "targets": {"http://t.example/3": 500}},
        "uni-b.example": {"class": "universities-research",
                          "targets": {"http://t.example/4": 1}},
    }}
    hcr, _ = pdq.endorser_regs_by_class(endorse)
    # 1000 links from uni-a's two hosts = ONE registrable; uni-b = a second.
    assert pdq.n_corroborating_regs("t.example", "edu", hcr) == 2


def test_sibling_endorsement_does_not_corroborate():
    """A sibling host's endorsement is evidence about the registrable, not about
    this host. Task 440 measured 11,968 hosts holding a high-trust class with no
    direct endorsement of that class at all."""
    endorse = {"edges": {"uni.example": {
        "class": "universities-research",
        "targets": {"http://blog.target.example/x": 5}}}}
    hcr, _ = pdq.endorser_regs_by_class(endorse)
    assert pdq.n_corroborating_regs("blog.target.example", "edu", hcr) == 1
    assert pdq.n_corroborating_regs("shop.target.example", "edu", hcr) == 0


# --- group 4: doctrine -- the rule may not name a host ----------------------

def test_task_203_denylist_is_gone():
    """ADR-043 A1 §7 forbids accreted special-case gates; Tim 2026-07-28: 'fix
    error CLASSES ... no host special-cased'. The 90-host COMMERCIAL_DENY caught
    503 of 90,598 inherited-high-trust hosts and none of task 438's six measured
    gov-tier commercial hosts."""
    assert not hasattr(pdq, "COMMERCIAL_DENY")
    assert not hasattr(pdq, "_is_commercial_host")


def test_trust_rule_names_no_host():
    """Mechanical guard against re-accreting a denylist: no hostname literal may
    appear anywhere in the trust rule outside its documentation."""
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("cytir", "acquire", "trust_model.py").read_text()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    hostlike = re.compile(r"\b[a-z0-9][a-z0-9-]*\.(com|ch|org|net|de|fr|it|edu|gov|io)\b")
    offenders = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and n.value not in docstrings and hostlike.search(n.value)]
    assert not offenders, offenders


# --- group 5: generalization on HELD-OUT chaff ------------------------------
# Task 203's denylist provably missed all of these: task 438's fresh sample
# found them carrying mission-tier trust. Counts are the MEASURED number of
# distinct endorsing registrables from the real edge graph (endorse_after.json,
# 5,622 seeds / 641,471 edges). The rule floors every one of them below every
# mission tier while naming none of them.
HELD_OUT_CHAFF = [
    ("miniorange.com", "gov", 1), ("comarketing-news.fr", "gov", 1),
    ("presseportal.de", "gov", 1), ("legal500.com", "gov", 4),
    ("triviso.com", "edu", 1), ("openid.net", "edu", 1),
    ("chemeurope.com", "edu", 4), ("rothoblaas.ru.com", "edu", 8),
    ("chordie.com", "edu", 11), ("justpaste.it", "health", 1),
]


@pytest.mark.parametrize("host,cls,n", HELD_OUT_CHAFF)
def test_held_out_chaff_loses_mission_trust(host, cls, n):
    got = resolve_trust(CLASS_TRUST[cls], ENDORSED, n)
    assert got <= INHERITED_CEILING < CLASS_TRUST[cls], (host, got)


def test_the_recall_cost_is_real_and_pinned():
    """HONESTY TEST. Corroboration does NOT separate authorities from
    aggregators: chordie.com carries 11 independent university endorsements
    while git-scm.com carries 1. So genuine non-.edu authorities are demoted
    too. They stay discoverable and keep their class -- only the unearned
    boost goes -- but the cost is real and is pinned here rather than hidden."""
    # measured from the same graph
    assert resolve_trust(CLASS_TRUST["edu"], ENDORSED, 8) < CLASS_TRUST["edu"]   # numpy.org
    assert resolve_trust(CLASS_TRUST["edu"], ENDORSED, 6) < CLASS_TRUST["edu"]   # jmlr.org
    # but never to zero, and never below the sub-band classes: still findable
    assert resolve_trust(CLASS_TRUST["edu"], ENDORSED, 6) >= INHERITED_FLOOR
