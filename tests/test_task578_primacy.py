"""Task 578 — tests for the two authority-signal serving levers.

(a) AW_REFCANON_TARGETS  — the DEMOTED class set of the 404 REF-CANON lever,
                           parameterised (default "reference" == 404 exactly).
(d) AW_PRIMACY           — PRIMACY-KEEP: refund the ADR-036 general-intent
                           relgate / trust-floor attenuation to a
                           relevance-competitive, title-central PRIMARY source.

Same loading pattern as `test_task404_levers.py` / `test_task406_levers.py`:
the lever code is imported from the serving modules (S10 step 3, task 1083: no source
slicing), with the env set BEFORE a fresh import so the module-level flag constants
resolve to the tested configuration.

The payload records are SEALED, not fetched: every one below was read from the
twin `aw_e3` on 2026-09-11 by the task-578 qd_lookup_578 script and is
copied here verbatim, so this module has no network dependency and cannot
silently change when the collection does (memory
`fixes-land-in-pipeline-with-guard`: real records, sealed inputs).

Run: python3 -m pytest on this file.
"""
import json
import os
import pathlib
import re
import sys

import pytest
import yaml

from tests.helpers.repo_files import repo_file_or_skip  # deploy/ compose only

APP = pathlib.Path(__file__).resolve().parent.parent
RETRIEVAL = APP / "cytir" / "serve" / "retrieval.py"
# S10 step 1 (task 1061): the lexical helpers moved to textmatch.py; the lever block stays
# in retrieval.py, so the chunks come from both files.
COMPOSE = APP / "deploy" / "docker-compose.yaml"

# --- sealed records: the ho-ai-002 / ho-ai-004 / ho-sw-010 displacements of
# --- task 577 §4, read from aw_e3 on 2026-09-11 (population.v1.json).
REC = {
    "deepmind": {"url": "https://deepmind.google/models/gemini/",
                 "source_type": "official-docs", "trust_tier": 0.9,
                 "domain": "deepmind.google", "title": "Gemini — Google DeepMind"},
    "wiki_gemini": {"url": "https://en.wikipedia.org/wiki/Google_Gemini",
                    "source_type": "reference", "trust_tier": 0.7,
                    "domain": "en.wikipedia.org", "title": "Google Gemini"},
    "hf_qwen": {"url": "https://huggingface.co/Qwen", "source_type": "model-hub",
                "trust_tier": 0.85, "domain": "huggingface.co",
                "title": "Qwen (Alibaba Cloud) — Hugging Face"},
    "willison": {"url": "https://simonwillison.net/2025/Apr/29/qwen-3",
                 "source_type": "blog", "trust_tier": 0.8,
                 "domain": "simonwillison.net",
                 "title": "Qwen 3 offers a case study in how to effectively "
                          "release a model"},
    "askubuntu": {"url": "https://askubuntu.com/questions/761879",
                  "source_type": "community-qa", "trust_tier": 0.55,
                  "domain": "askubuntu.com",
                  "title": "Why doesn't the latest Ubuntu edition come with "
                           "the latest stable Linux kernel version?"},
}


def _load(env=None, decompound=None):
    """Import the serving modules under `env` and return the `retrieval` namespace (S10 step 3,
    task 1083: the lever code lives in `query.py`/`levers.py` now, so the tests import it instead
    of slicing source). Settings are read at import, so the `aw_server` modules are purged first
    and again after, and a fresh import parses the env this test just set. `decompound` replaces
    the German splitter in `textmatch` (the real one needs an optional package)."""
    import importlib
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in (str(root), str(root / "server")):
        if p not in sys.path:
            sys.path.insert(0, p)

    def _purge():
        for m in [m for m in sys.modules if m == "cytir.serve" or m.startswith("cytir.serve.")]:
            del sys.modules[m]

    saved = {}
    for k, v in (env or {}).items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    _purge()
    try:
        r = importlib.import_module("cytir.serve.retrieval")
        sys.modules["cytir.serve.textmatch"]._decompound = decompound or (lambda t: [])
        return vars(r)
    finally:
        _purge()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _row(pid, rec, rel, pen=0.0):
    """A scored row in `_apply_refcanon`'s shape: [pid, payload, rel_core,
    penalties, trust_tier]."""
    return [pid, dict(rec), rel, pen, rec["trust_tier"]]


# ------------------------------------------------------------- inert by default
def test_defaults_are_byte_identical():
    """Merging the CODE alone must not change any score anywhere."""
    ns = _load({})
    assert ns["_PRIMACY"] is False
    assert ns["_PRIMACY_KEEP"] == 0.0
    assert ns["_REFCANON_TARGETS"] == frozenset({"reference"})
    assert ns["_REFCANON_MAXBETA"] == 0.25
    assert ns["_REFCANON_MINCENT"] == 0.0
    # the refund is the identity at KEEP = 0
    for s in (0.0, 0.25, 0.5, 1.0):
        assert ns["_primacy_refund"](s) == s


def test_prod_compose_ships_neither_lever():
    """THE GUARD. `deploy/docker-compose.yaml` is what production runs. If this
    file ever gains one of task 578's knobs without a POLICY PR, this fails —
    the lever is a ratified serving-policy change, never a side effect of
    merging the code that implements it."""
    env = yaml.safe_load(repo_file_or_skip(COMPOSE).read_text())["services"]["aw-api"]["environment"]
    for k in ("AW_PRIMACY", "AW_PRIMACY_KEEP", "AW_PRIMACY_BETA",
              "AW_PRIMACY_MINCENT", "AW_PRIMACY_EXTRA_CLASSES",
              "AW_REFCANON_TARGETS", "AW_REFCANON_MAXBETA",
              "AW_REFCANON_MINCENT", "AW_REFCANON_EXTRA_CLASSES"):
        assert k not in env, (
            f"{k} is set in prod's compose file. Task 578's levers are "
            f"MEASURED, not deployed; a serving-policy change is Tim's, by PR.")


# --------------------------------------------------------- the primary classes
@pytest.mark.parametrize("st,want", [
    # the location-bound set, by enumeration and by prefix rule
    ("official", True), ("official-docs", True), ("official-news", True),
    ("gov", True), ("gov-stat", True), ("gov-stats", True), ("gov-tax", True),
    ("gov-geo", True), ("gov-court", True), ("gov-law", True),
    ("gov-health", True), ("case_law", True), ("central-bank", True),
    ("registry_record", True),
    # the two families AW_PRIMACY_EXTRA_CLASSES adds by default
    ("docs", True), ("model-hub", True),
    # the secondary classes the population's displacers come from
    ("reference", False), ("blog", False), ("community-qa", False),
    ("hn-discussion", False), ("news", False), ("tech", False),
    ("health", False),
    # the card lane keeps its own boost; it is NOT refunded here
    ("entity_card", False),
    # defensive
    ("", False), (None, False), ("  OFFICIAL  ", True),
])
def test_is_primary_class(st, want):
    ns = _load({})
    assert ns["_is_primary_class"](st) is want


def test_unenumerated_gov_family_defaults_IN():
    """The prefix rule, for the same reason `_is_location_bound` has one
    (task 259 defect 1): a gov-* family nobody enumerated must not silently
    lose the refund."""
    ns = _load({})
    assert ns["_is_primary_class"]("gov-transport-2031") is True
    assert ns["_is_primary_class"]("official-register-x") is True


def test_extra_classes_is_configurable_and_replaces_the_default():
    ns = _load({"AW_PRIMACY_EXTRA_CLASSES": "entity_card"})
    assert ns["_is_primary_class"]("entity_card") is True
    assert ns["_is_primary_class"]("docs") is False     # replaced, not merged
    ns = _load({"AW_PRIMACY_EXTRA_CLASSES": ""})
    assert ns["_is_primary_class"]("docs") is False
    assert ns["_is_primary_class"]("official") is True  # location-bound stands


# ------------------------------------------------------------------ the refund
@pytest.mark.parametrize("keep,scale,want", [
    (0.0, 0.4, 0.4),        # inert
    (1.0, 0.4, 1.0),        # full exemption
    (0.5, 0.4, 0.7),        # halfway back to 1.0
    (1.0, 0.0, 1.0),        # a fully-stripped trust term is fully refunded
    (0.5, 1.0, 1.0),        # nothing to refund
])
def test_primacy_refund_arithmetic(keep, scale, want):
    ns = _load({"AW_PRIMACY": "1", "AW_PRIMACY_KEEP": str(keep)})
    assert ns["_primacy_refund"](scale) == pytest.approx(want)


@pytest.mark.parametrize("keep", ["-1.0", "2.0"])
def test_primacy_refund_is_clipped(keep):
    """An out-of-range KEEP may never AMPLIFY a trust term above its
    unattenuated value, nor invert the attenuation."""
    ns = _load({"AW_PRIMACY": "1", "AW_PRIMACY_KEEP": keep})
    out = ns["_primacy_refund"](0.4)
    assert 0.4 <= out <= 1.0


# -------------------------------------------------- REF-CANON's target set (a)
def _pool():
    """The ho-ai-002 shape: an official-docs primary and a reference page, plus
    the blog and Q&A displacers of ho-ai-004 / ho-sw-010."""
    return [_row("p1", REC["deepmind"], 1.00),
            _row("p2", REC["wiki_gemini"], 0.92),
            _row("p3", REC["willison"], 0.88),
            _row("p4", REC["askubuntu"], 0.70)]


def _fire(ns, rows, rh_beta=0.0):
    return ns["_apply_refcanon"](rows, ns["_content_terms"]("google gemini", "en"),
                                 "en", True, rh_beta, False)


def test_default_targets_demote_reference_only():
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35"})
    rows = _pool()
    assert _fire(ns, rows) == ["p2"]
    assert rows[1][3] == pytest.approx(-2.0)
    assert rows[2][3] == 0.0 and rows[3][3] == 0.0


def test_widened_targets_demote_blog_and_qa():
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35",
                "AW_REFCANON_TARGETS": "reference,blog,community-qa"})
    rows = _pool()
    assert _fire(ns, rows) == ["p2", "p3", "p4"]
    assert rows[0][3] == 0.0            # the primary is never demoted


def test_widened_targets_never_demote_a_primary_class():
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35",
                # a mis-typed policy value naming a PRIMARY class
                "AW_REFCANON_TARGETS": "reference,docs"})
    rows = _pool() + [_row("p5", dict(REC["deepmind"], source_type="docs"), 0.9)]
    out = _fire(ns, rows)
    # the lever does what the policy says — which is exactly why the POLICY is
    # the ratified artifact. Recorded so the behaviour is not a surprise.
    assert "p5" in out


def test_trigger_widening_arms_on_a_docs_canonical():
    """`docs` is not location-bound, so without EXTRA_CLASSES a documentation
    page cannot arm the lever — 16 of the population's addressable pairs."""
    rows = [_row("p1", dict(REC["deepmind"], source_type="docs"), 1.00),
            _row("p2", REC["wiki_gemini"], 0.92)]
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35"})
    assert _fire(ns, rows) == []
    rows = [_row("p1", dict(REC["deepmind"], source_type="docs"), 1.00),
            _row("p2", REC["wiki_gemini"], 0.92)]
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35",
                "AW_REFCANON_EXTRA_CLASSES": "docs"})
    assert _fire(ns, rows) == ["p2"]


def test_intent_gate_is_the_measured_blocker():
    """63 of the 126 addressable golds carry rh_beta >= 0.25 (rhbeta.v1.json):
    on those queries REF-CANON cannot fire at all at the shipped MAXBETA."""
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35"})
    assert _fire(ns, _pool(), rh_beta=0.60) == []
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35", "AW_REFCANON_MAXBETA": "0.9"})
    assert _fire(ns, _pool(), rh_beta=0.60) == ["p2"]


def test_head_off_keeps_the_lever_inert_at_any_maxbeta():
    """`head_on=False` must stay a hard stop: without the intent head there is
    no evidence of intent at all, and a blanket reference demotion is exactly
    what the 404 block comment forbids."""
    ns = _load({"AW_REFCANON": "1", "AW_REFCANON_W": "-2.0",
                "AW_REFCANON_BETA": "0.35", "AW_REFCANON_MAXBETA": "0.9"})
    rows = _pool()
    assert ns["_apply_refcanon"](rows, ns["_content_terms"]("google gemini", "en"),
                                 "en", False, 0.0, False) == []


def test_effective_config_echoes_the_v2_head_load_state():
    # S10 step 2 (task 1071): the echo is a projection of the typed settings; retrieval supplies
    # the load state.
    settings_src = (RETRIEVAL.parent / "settings.py").read_text()
    assert '"rankhead_v2": {"requested": s.query.rankhead_v2,' in settings_src
    assert '"loaded": rankhead_v2_loaded,' in settings_src
    assert "rankhead_v2_loaded=_RANKHEAD_V2_MODEL is not None" in RETRIEVAL.read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
