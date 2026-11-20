#!/usr/bin/env python3
"""Task 548 -- the site-name identity check, replayed against the real incident.

`identity.dense` cannot see this defect: it compares the embedded surface to
the row's own title, and a stripped title matches a stripped surface perfectly.
So `impactia.org/la-fondation/` shipped with a present, non-empty, perfectly
conformant title of `La Fondation` and no `impactia` posting at any weight,
fell from rank 1 to 40, and took the `names` gold class from 5 to 2 h@1 with
every existing guard green.

Every case below replays `tests/testdata/task548_site_name.json` -- the
`task427_postconsent` stage's OWN 20 impactia.org rows, record payload plus the
exact text it handed the embedder, sealed once off the build host's HDD by
task 548's fixture-sealing script. No test here opens a socket,
reads a collection, or invents a surface.

Run: python3 -m pytest tests/test_aw_index_representation_contract_site_name.py
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent

import cytir.core.common as AC                    # noqa: E402
from cytir.index import representation_contract as RC      # noqa: E402

FIXTURE = HERE / "testdata" / "task548_site_name.json"
FAMILY = "postconsent_html"

# The five rows whose SEALED SURFACE carries no `impactia` token anywhere --
# not the thirteen whose titles lost the suffix. The distinction is the point:
# eight of those thirteen are rescued by their own body text and need no fix.
SURFACE_LOSSES = {
    "https://impactia.org/aiia-festival/",
    "https://impactia.org/la-fondation/",
    "https://impactia.org/new-generation/",
    "https://impactia.org/programme-de-mentora-my-mentor-is-a-woman/",
    "https://impactia.org/women-in-ai/",
}


@pytest.fixture(scope="module")
def rows():
    doc = json.loads(FIXTURE.read_text())
    assert doc["schema"] == "aw.task548-site-name-fixture.v1"
    assert doc["host"] == "impactia.org"
    return doc["rows"]


@pytest.fixture(scope="module")
def fam():
    return RC.load_registry()["families"][FAMILY]


def checks_for(row, fam):
    surface = row["sealed_text"]
    return sorted({v["check"]
                   for v in RC.check_row(row, fam, surface, surface)})


# --- the label, which both the guard and the composer read -------------------

def test_the_label_comes_from_one_function():
    """`site_entity_label` is a mirror contract: the composer folds it into the
    surface and the guard looks for it there. A second reading of "what is this
    site called" would let the guard pass a row the composer could not build."""
    assert AC.site_entity_label("https://impactia.org/la-fondation/") == "Impactia"
    assert AC.site_entity_label("https://www.dardagny.ch/dechets") == "Dardagny"


def test_a_generic_label_is_unassertable_not_absent():
    """`admin.ch` names the state, not a body. The guard must say so under its
    OWN check name rather than fail every federal page for lacking a name it
    was never going to have."""
    assert AC.site_entity_label("https://www.admin.ch/gov/de/start.html") is None


def test_unassertable_reports_itself_and_does_not_read_as_a_loss(fam):
    row = {"url": "https://www.admin.ch/gov/de/start.html", "title": "Bundesrat",
           "source_type": "official", "text": "Der Bundesrat."}
    got = checks_for({**row, "sealed_text": "Bundesrat. Der Bundesrat."}, fam)
    assert "identity.site_name_unassertable" in got
    assert not any(c.startswith("identity.site_name.") for c in got)


# --- the incident ------------------------------------------------------------

def test_the_fixture_is_the_lane_s_own_rows(rows):
    assert len(rows) == 20
    assert all(r["url"].startswith("https://impactia.org/") for r in rows)
    # every staged row carried no site_name at all -- the mechanism itself
    assert all(not r.get("site_name") for r in rows)


def test_la_fondation_is_refused(rows, fam):
    row = next(r for r in rows if r["url"] == "https://impactia.org/la-fondation/")
    assert row["title"] == "La Fondation"
    got = checks_for(row, fam)
    assert "identity.site_name.dense" in got
    assert "identity.site_name.sparse" in got


def test_exactly_the_five_surface_losses_are_refused(rows, fam):
    refused = {r["url"] for r in rows
               if any(c.startswith("identity.site_name.") for c in checks_for(r, fam))}
    assert refused == SURFACE_LOSSES


def test_a_stripped_title_rescued_by_its_body_is_not_refused(rows, fam):
    """`Conseil de Fondation` lost the suffix too, and its body says `impactIA`.
    BM25 does not care which part of the surface a posting came from, so this
    row is not a defect -- and a guard that keyed on the TITLE would have put
    it, and seven others, into a delta that needed none."""
    row = next(r for r in rows
               if r["url"] == "https://impactia.org/conseil-de-fondation/")
    assert "impactia" not in row["title"].lower()
    assert not any(c.startswith("identity.site_name.") for c in checks_for(row, fam))


def test_a_title_that_kept_the_suffix_passes(rows, fam):
    row = next(r for r in rows if r["title"] == "impactIA Academy")
    assert not any(c.startswith("identity.site_name.") for c in checks_for(row, fam))


def test_the_composed_repair_clears_every_refusal(rows, fam):
    """The fix, on the real rows: resolve `site_name` through ADR-061's own
    fallback and recompose. Nothing else changes -- same body, same title
    field semantics, same family."""
    for r in rows:
        if r["url"] not in SURFACE_LOSSES:
            continue
        site = AC.site_entity_label(r["url"])
        repaired = AC.compose_index_text(
            {"text": r["text"], "title": r["title"], "site_name": site,
             "source_type": r["source_type"]})
        got = sorted({v["check"] for v in
                      RC.check_row({**r, "site_name": site}, fam, repaired, repaired)})
        assert not any(c.startswith("identity.site_name") for c in got), (r["url"], got)
        assert "impactia" in repaired.lower()


# --- the declaration is opt-in ----------------------------------------------

def test_a_family_that_does_not_declare_it_is_not_checked(rows):
    """Opt-in, like every other dimension here: a family that has never carried
    a site name must not be failed over a rule nobody measured for it."""
    silent = {k: v for k, v in RC.load_registry()["families"]["f2_html"].items()}
    assert "site_name_identity" not in silent
    row = next(r for r in rows if r["url"] == "https://impactia.org/la-fondation/")
    got = sorted({v["check"] for v in
                  RC.check_row(row, silent, row["sealed_text"], row["sealed_text"])})
    assert not any(c.startswith("identity.site_name") for c in got)


def test_the_declared_source_types_bound_the_check(rows, fam):
    """A news article on a foundation's domain is not named by the foundation;
    the declaration names which source types the rule reaches."""
    row = next(r for r in rows if r["url"] == "https://impactia.org/la-fondation/")
    out_of_scope = {**row, "source_type": "news"}
    got = sorted({v["check"] for v in RC.check_row(
        out_of_scope, fam, row["sealed_text"], row["sealed_text"])})
    assert not any(c.startswith("identity.site_name") for c in got)


def test_the_registry_entry_names_its_evidence():
    """A constant without its evidence is a future re-derivation."""
    fam = RC.load_registry()["families"][FAMILY]
    assert "site-name-audit.json" in fam["evidence"]
    assert fam["site_name_identity"]["channels"] == ["dense", "sparse"]


# --- the site-ingest row's refusal -------------------------------------------
# The retired small-site ingester (task 1433) composed its row with
# `AC.resolve_title_and_site_name` + `AC.compose_index_text` and checked it with
# `RC.check_row` over the composed text. `_site_row` below is that composition,
# so these cases keep guarding the `site_ingest_html` family at the same seam.

def _site_row(url, html, body, *, domain, tier, source_type, trust, lang):
    import trafilatura
    meta = trafilatura.extract_metadata(html)
    clean_title = (getattr(meta, "title", None) or "").strip() or url
    title, site_name = AC.resolve_title_and_site_name(
        html, clean_title, url, source_type)
    composed = AC.compose_index_text(
        {"text": body[:6000], "title": title, "site_name": site_name,
         "source_type": source_type})
    return {"url": url, "title": title, "site_name": site_name,
            "snippet": " ".join(body.split()[:30]), "text": composed,
            "lang": lang, "domain": domain, "tier": tier,
            "source_type": source_type, "trust_tier": trust}


def _violations(rec, family):
    return RC.check_row(rec, family, rec["text"], rec["text"])


def test_the_ingester_repairs_a_stripped_title_at_the_source():
    """The `site_ingest_html` family's rows had the same defect at the source:
    a title that lost its site suffix. The composition repairs it."""
    html = ("<html><head><title>La Fondation - impactIA</title></head>"
            "<body><h1>La Fondation</h1></body></html>")
    stripped = ("<html><head><title>La Fondation</title></head>"
                "<body><h1>La Fondation</h1></body></html>")
    site_fam = RC.load_registry()["families"]["site_ingest_html"]
    for markup in (html, stripped):
        p = _site_row(
            "https://impactia.org/la-fondation/", markup,
            "Notre mission est de guider les acteurs.",
            domain="impactia.org", tier="B", source_type="entity-self",
            trust=0.5, lang="fr")
        assert "impactia" in (p["site_name"] or "").lower()
        assert "impactia" in p["text"].lower()
        assert _violations(p, site_fam) == []


def test_the_ingester_refuses_a_row_it_cannot_repair():
    """When the label is unassertable the row must not pretend: it reports
    `identity.site_name_unassertable`, which is a different fact from
    "the site name is missing"."""
    site_fam = RC.load_registry()["families"]["site_ingest_html"]
    p = _site_row(
        "https://www.admin.ch/gov/de/start.html",
        "<html><head><title>Start</title></head><body>Start</body></html>",
        "Der Bundesrat informiert.", domain="admin.ch", tier="A",
        source_type="official", trust=1.0, lang="de")
    got = [v["check"] for v in _violations(p, site_fam)]
    assert got == ["identity.site_name_unassertable"]


# --- task 560: the domain-derived label can be accented where the domain
#     itself cannot (a DNS name has no accents) -----------------------------

def test_an_accented_site_name_is_not_a_false_refusal(fam):
    """fr-gov-027 (task 560): www.assemblee-nationale.fr's own title says
    'Les groupes politiques - Assemblée nationale' -- the site name IS in
    the surface, spelled correctly with its accent. `site_entity_label`
    derives 'Assemblee-nationale' from the ASCII hostname (no accent, by
    construction), and the OLD plain-`carries` comparison folded punctuation
    but never accents, so 'assemblee nationale' was never a substring of
    'assemblée nationale' and every row on this host was refused as missing
    a site name it actually carried. This is a checker false-positive, not
    ADR-061 bug #1's title-suffix loss -- confirmed by hand before the fix
    (representation_contract.carries returns False here; the composed text
    contains the accented name verbatim)."""
    row = {"url": "https://www.assemblee-nationale.fr/dyn/les-groupes-politiques",
           "title": "Les groupes politiques - Assemblée nationale",
           "source_type": "official",
           "text": "Les groupes politiques - Assemblée nationale. "
                   "Les groupes politiques\nLes groupes rassemblent des "
                   "députés selon leurs affinités politiques."}
    surface = row["text"]
    got = checks_for({**row, "sealed_text": surface}, fam)
    assert not any(c.startswith("identity.site_name.") for c in got), got


def test_carries_foldaccents_is_scoped_not_a_norm_change():
    """The fix is a dedicated comparison for this one check, not a change to
    `carries`/`_norm` (whose docstring specifically protects 'Küsnacht' from
    becoming 'k snacht' under accent folding elsewhere in the contract)."""
    assert RC.carries_foldaccents(
        "Les groupes politiques - Assemblée nationale", "Assemblee-nationale")
    assert not RC.carries(
        "Les groupes politiques - Assemblée nationale", "Assemblee-nationale")
    # Küsnacht is untouched: plain `carries` still treats the umlaut as its
    # own token, exactly as its docstring promises.
    assert not RC.carries("Küsnacht is a town near Zurich", "kusnacht")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
