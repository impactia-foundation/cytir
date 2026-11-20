"""Task 2737: projected snippets skip the `## Record identity` provenance header.

Rows are BUILT in the producers' template shape (labels copied from cytir/extract/{bfs_dam_current,
bfs_pxweb,swisstopo,pubmed}.py with real-shaped values); no pack row was read.
"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from cytir.index import b1_projection as BP
from cytir.index import representation_contract as RC
from cytir.index.snippet_surface import strip_provenance_header

ROWS = {
    "bfs.admin.ch": (
        "# Bevölkerung der Schweiz 2023\n\n## Record identity\n\n"
        "- Canonical asset: https://www.bfs.admin.ch/asset/de/34567890\n- BFS DAM asset ID: 34567890\n"
        "- API UUID: 0a1b2c3d-1111-2222-3333-444455556666\n- Language: de\n- Content ID: 99\n\n"
        "## Description\n\nDie ständige Wohnbevölkerung der Schweiz betrug Ende 2023 9 048 900 Personen."),
    "pxweb.bfs.admin.ch": (
        "# Ständige Wohnbevölkerung nach Alter\n\n## Record identity\n\n- PXWeb database ID: `px-x-0102010000_101`\n"
        "- Language: `de`\n- API endpoint: `https://x/api`\n- Parent response: `r.json`\n"
        "- Parent response SHA-256: `ab12`\n- Retrieved at: `2026-10-01`\n\n## Dimensions\n\n### 1. Alter\n"),
    "geo.admin.ch": (
        "# Matterhorn\n\n## Record identity\n\n- SwissNAMES3D UUID: `abc`\n- Name UUID: `def`\n"
        "- Release: `2026-10`\n- Release datetime: `2026-10-01`\n- Parent archive member: `a/b.csv`\n"
        "- Parent archive SHA-256: `ff00`\n\n## Classification\n\n- Object type: Gipfel"),
    "pubmed.ncbi.nlm.nih.gov": (
        "# Effects of aspirin on stroke\n\n## Record identity\n\n- PMID: `12345`\n- Parent archive: `pubmed.xml.gz`\n"
        "- Parent archive SHA-256: `ee11`\n\n## Abstract\n\nAspirin reduces recurrent stroke risk."),
}
LEAD = {"bfs.admin.ch": "# Bevölkerung der Schweiz 2023 ## Description",
        "pxweb.bfs.admin.ch": "# Ständige Wohnbevölkerung nach Alter ## Dimensions",
        "geo.admin.ch": "# Matterhorn ## Classification",
        "pubmed.ncbi.nlm.nih.gov": "# Effects of aspirin on stroke ## Abstract"}


def _project(domain, text):
    src = SimpleNamespace(id="x", vector={"dense": [0.1]}, payload={
        "url": f"https://{domain}/r/1", "anchor": "a", "title": "T", "lang": "de",
        "domain": domain, "source_type": "official", "trust_tier": 1.0})
    gate = SimpleNamespace(verdict=lambda _h, _r, _p=None: (True, "ok"))
    point, _ = BP.project_record(src, "fam", {"texts": [text], "dense_text": text,
                                              "sparse_text": text}, {}, {}, gate, defaultdict(int))
    return point["payload"]["snippet"]


@pytest.mark.parametrize("domain", sorted(ROWS))
def test_snippet_starts_at_title_then_content(domain):
    snip = _project(domain, ROWS[domain])
    assert snip.startswith(LEAD[domain]), snip


@pytest.mark.parametrize("domain", sorted(ROWS))
def test_guard_snippet_never_starts_with_template_label(domain):
    snip = _project(domain, ROWS[domain])
    assert "Record identity" not in snip and not snip.startswith("## Record identity")
    # a header-first row without a title line is cut too
    bare = ROWS[domain].split("\n\n", 1)[1]
    assert not _project(domain, bare).startswith("## Record identity")


def test_plain_and_law_rows_unchanged():
    plain = "Der Bundesrat hat entschieden.\n\n## Record identity is a phrase here."
    assert strip_provenance_header(plain) == plain
    mid = "# T\n\nIntro.\n\n## Record identity\n\n- PMID: `1`\n"
    assert strip_provenance_header(mid) == mid  # not leading: untouched
    law = "Art. 336 c {#art_336_c}\n\nNach Ablauf der Probezeit darf der Arbeitgeber nicht kündigen."
    assert strip_provenance_header(law) == law
    assert "Nach Ablauf" in _project("fedlex.admin.ch", law) and "{#" not in _project("fedlex.admin.ch", law)


def test_header_only_row_keeps_title():
    assert strip_provenance_header("# T\n\n## Record identity\n\n- PMID: `1`\n") == "# T\n\n"


def test_contract_class_flags_header_in_snippet():
    fam = {"dense": "body_only", "sparse": "body_only", "refused_index_markup": ["record_identity_header"]}
    bad = {"text": "x", "snippet": "## Record identity\n- PMID: 1", "title": "T"}
    assert "markup.snippet" in {v["check"] for v in RC.check_row(bad, fam)}
    ok = {"text": "x", "snippet": "# T ## Abstract Aspirin", "title": "T"}
    assert "markup.snippet" not in {v["check"] for v in RC.check_row(ok, fam)}


def test_bullets_without_trailing_blank_line_are_stripped():
    t = "# T\n\n## Record identity\n\n- A: 1\n- B: 2\n## Description\n\nbody"
    assert strip_provenance_header(t) == "# T\n\n## Description\n\nbody"


def test_crlf_header_is_stripped():
    t = "# T\r\n\r\n## Record identity\r\n\r\n- A: 1\r\n\r\n## Description\r\n\r\nbody"
    assert strip_provenance_header(t) == "# T\r\n\r\n## Description\r\n\r\nbody"


def test_real_pack_form_without_hashes_or_blank_line_is_stripped():
    """Task 2734: the sealed pack text is `Record identity\n- ...`, not the extractor template."""
    text = ("Record identity\n- Canonical asset: https://www.bfs.admin.ch/asset/en/1\n- Language: en\n\n"
            "Description\n- Subtitle: Weekly number of deaths")
    assert strip_provenance_header(text) == "Description\n- Subtitle: Weekly number of deaths"
    assert "Record identity" not in _project("bfs.admin.ch", text)
