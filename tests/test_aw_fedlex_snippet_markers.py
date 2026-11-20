"""Task 2733: fedlex snippets must not carry Akoma Ntoso anchor markup.

The sealed row below is the shape the epoch-4 fedlex pack carries (OR art. 336c, as the
2026-11-12 agent probe saw it served). The snippet is cut from the pack `text`, which still
holds the eId scaffolding; the index surfaces were stripped in task 564, the snippet was not.
"""
from __future__ import annotations

import re
from collections import defaultdict
from types import SimpleNamespace

from cytir.index import b1_projection as BP
from cytir.index import representation_contract as RC

MARKER = re.compile(r"\{#[a-z_0-9/]+\}")
RAW = ("Art. 336 c {#art_336_c}\n\nArt. 336 c\n\nparagraph {#art_336_c/para_1}\n\n1\n\n"
       "listIntroduction {#art_336_c/para_1/listintro}\n\n"
       "Nach Ablauf der Probezeit darf der Arbeitgeber das Arbeitsverhältnis nicht kündigen.")
URL = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336_c"


def _project(domain, text):
    src = SimpleNamespace(id="x", vector={"dense": [0.1]}, payload={
        "url": URL, "anchor": "art_336_c", "title": "OR — Art. 336 c", "lang": "de",
        "domain": domain, "source_type": "official", "trust_tier": 1.0})
    gate = SimpleNamespace(verdict=lambda _h, _r, _p=None: (True, "ok"))
    point, _ = BP.project_record(src, "fedlex", {"texts": [text], "dense_text": text,
                                                 "sparse_text": text}, {}, {}, gate, defaultdict(int))
    return point["payload"]["snippet"]


def test_fedlex_snippet_is_clean_and_keeps_the_law():
    snip = _project("fedlex.admin.ch", RAW)
    assert not MARKER.search(snip) and "listIntroduction" not in snip
    assert "paragraph" not in snip and "Nach Ablauf der Probezeit" in snip


def test_other_domains_keep_their_text_untouched():
    assert "{#art_336_c}" in _project("example.org", RAW)


def test_contract_refuses_marker_in_snippet_or_title():
    fam = {"dense": "body_only", "sparse": "body_only", "refused_index_markup": ["akn_anchor_id"]}
    bad = {"text": "Nach Ablauf.", "snippet": "Art. 336 c {#art_336_c} Art. 336 c", "title": "OR {#x}"}
    checks = {v["check"] for v in RC.check_row(bad, fam)}
    assert {"markup.snippet", "markup.title"} <= checks
    ok = {"text": "Nach Ablauf.", "snippet": "Nach Ablauf.", "title": "OR — Art. 336 c"}
    assert not {v["check"] for v in RC.check_row(ok, fam)} & {"markup.snippet", "markup.title"}
