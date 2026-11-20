"""Which point-id scheme does each family's sealed pack use, and does compose check THAT one?

`specs/aw-index-serve-gate.md` §1.3 inferred from the code that `compose.verify_identity` checks
only the legacy mint (`aw_common.mint_passage_point_id`) while a b1-minted pack (fedlex:
`cytir.core.ids.mint_point_id(canonical_url, anchor)`) would reproduce 0/N there. Step 2's first test
settles it on a REAL sealed fedlex pack: the offline golden tape of `cc/1948/195`
(`tests/fixtures/fedlex_offline/`, recorded live 2026-09-30), run through the real
`fedlex_exact` family door, then asked of `verify_identity` exactly as `compose_pack` asks it.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.core import ids as ID  # noqa: E402
from cytir.core.common import canonical_url  # noqa: E402
from cytir.index import compose  # noqa: E402
import test_aw_family_fedlex as FX  # noqa: E402  (the sealed tape + its client, not copied)


@pytest.fixture
def fedlex_stage(monkeypatch, tmp_path):
    """The fedlex pack's rows, staged the way `family_stage` stages them: one
    `chunks/shard-0000.texts.jsonl.gz`, rows untouched."""
    want = FX._expected()
    FX._run(monkeypatch, tmp_path / "run", today=want["last_seen_pinned"])
    rows = [json.loads(x) for x in
            (tmp_path / "run" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = tmp_path / "stage" / "chunks"
    chunks.mkdir(parents=True)
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return chunks, rows


def test_fedlex_rows_carry_the_i1_tier_b_mint(fedlex_stage):
    """Task 1035: the pack's declared id is `mint_point_id(canonical_url, sha256 of the text)` (I1,
    tier B). It was `(canonical_url, anchor)`, the b1 scheme, before the door was ported."""
    _chunks, rows = fedlex_stage
    assert len(rows) == FX._expected()["rows"] > 0
    for r in rows:
        assert r["point_id"] == ID.mint_point_id(canonical_url(r["url"]), ID.content_digest(r["text"]))


def test_the_legacy_check_REJECTS_a_b1_minted_fedlex_pack(fedlex_stage):
    """SETTLED (task 696 step 2): the spec's inference holds. Under the default scheme the check
    cannot even form an anchor (fedlex rows carry no document_id/chunk_ordinal); under the
    raw-anchor scheme it reproduces 0/N. Either way compose would refuse a fedlex pack today."""
    chunks, rows = fedlex_stage
    with pytest.raises((compose.ComposeError, KeyError)):
        compose.verify_identity(str(chunks))
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: %d/%d" % (len(rows), len(rows))):
        compose.verify_identity(str(chunks), anchor_fn=compose.raw_anchor_field)


# --- the declaration: compose reads the family's own scheme ---------------------------------------

def test_fedlex_declares_the_i1_tier_scheme_and_reproduces_63_of_63(fedlex_stage):
    chunks, rows = fedlex_stage
    assert compose.id_scheme_for_family("fedlex_exact") == compose.ID_SCHEME_I1_TIER
    got = compose.verify_identity(str(chunks), scheme=compose.id_scheme_for_family("fedlex_exact"))
    assert got == {"checked": len(rows), "matched": len(rows), "mismatches": []} and len(rows) == 63


def test_an_i1_declared_family_still_refuses_a_wrong_id(fedlex_stage, tmp_path):
    _chunks, rows = fedlex_stage
    bad = tmp_path / "bad" / "chunks"
    bad.mkdir(parents=True)
    rows = [dict(r) for r in rows]
    rows[3]["point_id"] = "00000000-0000-5000-8000-000000000000"
    with gzip.open(bad / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: 1/63"):
        compose.verify_identity(str(bad), scheme=compose.ID_SCHEME_I1_TIER)


def test_every_other_family_keeps_the_legacy_scheme():
    for family in ("eric_official_pdf", "cc_authority_frontier_w7", "dila_juri", "dila_kali",
                   "dila_jorf", "us_federal_register_ecfr", "govuk_content",
                   "de_openlegaldata_v1", "stackoverflow", "anything-new"):
        assert compose.id_scheme_for_family(family) == compose.ID_SCHEME_LEGACY
    assert set(compose.FAMILY_ID_SCHEME) == {
        "fedlex_exact", "loda_exact", "scratch_weko_docs_exact", "scratch_bav_docs_exact",
        "scratch_bazl_docs_exact", "scratch_elcom_docs_exact"}


# --- loda_exact: the third scheme ---------------------------------------------------------------

@pytest.fixture
def loda_stage(tmp_path):
    """The three real LEGI/LODA fixture articles through the real `loda_exact` family door, staged
    as one chunk shard the way `family_stage` stages a pack (rows untouched)."""
    import shutil
    import test_aw_family_loda as LX
    root = tmp_path / "view"
    branch = root / "legi" / "global" / "code_et_TNC_en_vigueur" / "TNC_en_vigueur" / "JORF"
    branch.mkdir(parents=True)
    for i, f in enumerate(sorted(LX.FIX.glob("*.xml"))):
        shutil.copy(f, branch / f"LEGIARTI9{i:011d}.xml")
    LX._family(root, tmp_path / "fam")
    rows = [json.loads(x) for x in
            (tmp_path / "fam" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = tmp_path / "stage" / "chunks"
    chunks.mkdir(parents=True)
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return chunks, rows


def test_the_plain_legacy_check_rejects_a_loda_pack(loda_stage):
    """SETTLED (task 696 step 2, found while building the identity test): loda_exact's staged rows
    reproduce under NEITHER plain check -- the id is minted over the markdown, which the row holds
    only as `"<title>\\n\\n<markdown>"`."""
    chunks, rows = loda_stage
    with pytest.raises((compose.ComposeError, KeyError)):
        compose.verify_identity(str(chunks))
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: 3/3"):
        compose.verify_identity(str(chunks), anchor_fn=compose.raw_anchor_field)
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: 3/3"):
        compose.verify_identity(str(chunks), scheme=compose.ID_SCHEME_B1)


def test_loda_declares_the_titled_legacy_scheme_and_reproduces_3_of_3(loda_stage):
    chunks, rows = loda_stage
    assert compose.id_scheme_for_family("loda_exact") == compose.ID_SCHEME_LEGACY_TITLED
    got = compose.verify_identity(str(chunks), scheme=compose.id_scheme_for_family("loda_exact"))
    assert got == {"checked": 3, "matched": 3, "mismatches": []} and len(rows) == 3


def test_a_truncated_or_retitled_loda_row_is_refused_not_reminted(loda_stage, tmp_path):
    _chunks, rows = loda_stage
    bad = tmp_path / "bad" / "chunks"
    bad.mkdir(parents=True)
    rows = [dict(r) for r in rows]
    rows[0]["text"] = rows[0]["text"][:-5]                 # a truncated body: the id cannot reproduce
    rows[1]["title"] = "another title"                     # the text no longer starts with its title
    with gzip.open(bad / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: 2/3"):
        compose.verify_identity(str(bad), scheme=compose.ID_SCHEME_LEGACY_TITLED)


def test_an_unknown_scheme_is_refused_not_defaulted(fedlex_stage):
    chunks, _ = fedlex_stage
    with pytest.raises(compose.ComposeError, match="unknown id scheme"):
        compose.verify_identity(str(chunks), scheme="b2")
