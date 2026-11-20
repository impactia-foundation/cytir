#!/usr/bin/env python3
"""`extract_legi_loda_full.py` on REAL DILA LEGI/LODA article XML.

Task 633. Fixtures under `tests/fixtures/legi_loda_full/` are real
LEGIARTI XML files copied verbatim from the DILA LEGI dump view
(the TNC_en_vigueur tree, 2026-09-22) -- sealed inputs, not live calls (memory
`fixes-land-in-pipeline-with-guard`). Real ARRETE/DECRET/LOI/ORDONNANCE
articles at both sides of the quality floor, plus one genuinely ABROGE (not
in force) article and one ABROGE_DIFF (deferred-abrogation, still applicable)
article, so the tests exercise the actual DILA nature/ETAT vocabulary rather
than a synthetic approximation of it.

What these tests would have caught:
  * the old `ingest_dila_loda.py` NATURE allow-list silently re-appearing and
    dropping ARRETE again (task 633's whole point is admitting it);
  * a kept article losing its `chunk.law` source-native anchor (`article:
    LEGIARTI...`) -- the property that makes the URL fragment reachable;
  * `evaluate_article` accepting an ETAT the DILA schema never marks
    in-force, or rejecting ABROGE_DIFF (task 322's deferred-abrogation fix);
  * the nature-sensitive quality floor collapsing to one constant and letting
    short ARRETE noise back in, or over-filtering real DECRET/LOI bodies;
  * a shard row missing `point_id`/`text` -- the exact two fields
    task 413a's embed_shards script reads.
"""
from __future__ import annotations

import gzip
import json
import os

import pytest

from cytir.sources.families import loda as X

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "legi_loda_full")


def _read(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# evaluate_article: ETAT gate
# ---------------------------------------------------------------------------

def test_a_genuinely_abroge_article_is_rejected_not_in_force():
    # sample_loi.xml: real LOI article, ETAT=ABROGE (not VIGUEUR/ABROGE_DIFF).
    accept, reason = X.evaluate_article(_read("sample_loi.xml"))
    assert accept is None
    assert reason == "not_in_force:ABROGE"


def test_abroge_diff_is_in_force_task_322_fix():
    # sample_abroge_diff.xml: ETAT=ABROGE_DIFF, deferred abrogation -- still
    # applicable law today. Must clear the ETAT gate (it is short, so it is
    # then rejected by the body floor, not by ETAT -- assert the REASON, not
    # just non-acceptance, so a floor regression can't hide an ETAT regression).
    accept, reason = X.evaluate_article(_read("sample_abroge_diff.xml"))
    assert accept is None
    assert reason.startswith("below_body_floor")  # NOT not_in_force


# ---------------------------------------------------------------------------
# evaluate_article: the nature cap is LIFTED (task 633's whole point)
# ---------------------------------------------------------------------------

def test_arrete_is_admissible_the_old_nature_cap_is_lifted():
    """A real, in-force ARRETE article with a substantive body is KEPT.

    The retired ingest_dila_loda script's KEEP_NATURE would have dropped this
    unconditionally regardless of body length. This is the one assertion
    that would fail first if task 633's nature cap silently crept back in.
    """
    accept, reason = X.evaluate_article(_read("sample_arrete_kept.xml"))
    assert reason == "kept"
    assert accept["nature"] == "ARRETE"
    assert accept["state"] == "VIGUEUR"


def test_short_arrete_still_falls_to_the_quality_floor():
    # sample_arrete.xml is a real in-force ARRETE, but its body ("Les
    # territoires des communes...") is well under the 450-char short-form
    # floor -- lifting the NATURE cap must not also lift the QUALITY floor.
    accept, reason = X.evaluate_article(_read("sample_arrete.xml"))
    assert accept is None
    assert reason == "below_body_floor:ARRETE"


def test_decret_kept_and_short_decret_still_gated():
    kept, reason_kept = X.evaluate_article(_read("sample_decret_kept.xml"))
    assert reason_kept == "kept"
    assert kept["nature"] == "DECRET"

    short, reason_short = X.evaluate_article(_read("sample_decret.xml"))
    assert short is None
    assert reason_short == "below_body_floor:DECRET"


def test_loi_kept_at_the_lower_statutory_floor():
    accept, reason = X.evaluate_article(_read("sample_loi_kept.xml"))
    assert reason == "kept"
    assert accept["nature"] == "LOI"
    assert X.min_body_for("LOI") == X.MIN_BODY_STATUTORY == 200


def test_ordonnance_short_body_gated_at_statutory_floor():
    accept, reason = X.evaluate_article(_read("sample_ordonnance.xml"))
    assert accept is None
    assert reason == "below_body_floor:ORDONNANCE"


@pytest.mark.parametrize("nature,expected_floor", [
    ("LOI", 200), ("ORDONNANCE", 200),
    ("DECRET", 450), ("DÉCRET", 450),
    ("ARRETE", 450), ("ARRÊTÉ", 450),
    ("SOMETHING_NEW_DILA_ADDS", 450),  # unknown nature -> conservative floor
])
def test_min_body_for_every_nature_class(nature, expected_floor):
    assert X.min_body_for(nature) == expected_floor


# ---------------------------------------------------------------------------
# build_records: the two aligned outputs (L1 + embed shard)
# ---------------------------------------------------------------------------

def _built(name: str):
    body = _read(name)
    accept, reason = X.evaluate_article(body)
    assert reason == "kept", f"fixture {name} must be a KEPT case for this test"
    l1, rows, _source = X.build_records(accept, body=body, archive_member=name,
                                        archive_sha256="deadbeef" * 8,
                                        source_release="test-release-v1")
    # these fixtures are short articles: ONE row each (task 706: a long article gives several)
    assert len(rows) == 1
    return l1, rows[0]


def test_l1_record_is_the_adr105_shape_persist_l1_records_needs():
    l1, _shard = _built("sample_arrete_kept.xml")
    assert l1["schema"] == "aw.l1-markdown.v1"
    assert l1["url"].startswith("https://www.legifrance.gouv.fr/loda/article_lc/")
    assert l1["url"] == l1["canonical_url"]
    assert l1["markdown"] and l1["markdown"].strip()
    assert l1["anchor"] == f"article:{l1['source_record_id']}"
    # persist_l1_records.normalize() reads url/markdown by DEFAULT field
    # names -- a rename here would silently need new --url-field/--text-field
    # flags at persist time. Assert the defaults still match.
    from cytir.corpus import persist_l1_records as PL
    normalized = PL.normalize(l1, origin="dila-legi-loda-full-v1")
    assert normalized["url"] == l1["url"]
    assert normalized["markdown"] == l1["markdown"]


def test_shard_record_has_exactly_what_embed_shards_reads():
    _l1, shard = _built("sample_decret_kept.xml")
    # task 413a's embed_shards read_texts() reads exactly these two
    # fields off each row; everything else is provenance the binder wants.
    assert isinstance(shard["point_id"], str) and shard["point_id"]
    assert isinstance(shard["text"], str) and shard["text"].strip()
    assert shard["provenance"] == "loda"
    assert shard["license"] == "Etalab-2.0"
    assert shard["consent_state"] == "dump"


def test_l1_and_shard_share_the_same_point_id_and_anchor():
    l1, shard = _built("sample_loi_kept.xml")
    assert l1["point_id"] == shard["point_id"]
    assert l1["anchor"] == shard["anchor"]
    assert l1["url"] == shard["url"]


def test_embed_text_carries_the_article_identity_prefix():
    # embedding-surface-is-a-contract (memory): a sub-chunk (here, the WHOLE
    # article -- chunk.law never sub-splits without the unratified .sub
    # policy) must carry its own identity in the embedded text, not just the
    # payload. Assert the composed text is not bare body -- title leads it.
    l1, shard = _built("sample_arrete_kept.xml")
    assert shard["text"].startswith(l1["title"])


# ---------------------------------------------------------------------------
# end-to-end: run() over the fixtures directory (a "TNC_en_vigueur"-shaped tree)
# ---------------------------------------------------------------------------

def _real_article_id(xml_bytes: bytes) -> str:
    import re
    m = re.search(rb"<ID>(LEGIARTI[0-9]+)</ID>", xml_bytes)
    assert m, "fixture must carry a real META_COMMUN/ID"
    return m.group(1).decode()


@pytest.fixture()
def tnc_tree(tmp_path):
    """Arrange fixtures under a TNC_en_vigueur/ path segment, filed under their
    REAL LEGIARTI id -- `walk_tnc` requires `LEGIARTI*.xml`, matching the real
    dump's own on-disk naming, which the descriptive `sample_*.xml` fixture
    names deliberately don't (readability at rest, real shape when walked)."""
    tnc = tmp_path / "LEGI" / "TNC_en_vigueur"
    tnc.mkdir(parents=True)
    for name in os.listdir(FIXTURES):
        body = _read(name)
        (tnc / f"{_real_article_id(body)}.xml").write_bytes(body)
    return tmp_path


def test_run_end_to_end_over_a_real_fixture_tree(tnc_tree, tmp_path):
    out_l1 = tmp_path / "out.l1.jsonl"
    shard_dir = tmp_path / "shards"
    summary = X.run(root=str(tnc_tree), out_l1=str(out_l1),
                    shard_dir=str(shard_dir), rows_per_shard=3,
                    release="test-release-v1", log=lambda *a, **k: None)

    n_fixtures = len(os.listdir(FIXTURES))
    assert summary["scanned"] == n_fixtures
    # exactly the 3 "_kept" fixtures pass both gates (arrete/decret/loi;
    # ordonnance shares LOI's 200-char statutory floor, no separate fixture)
    assert summary["kept"] == 3
    assert set(summary["kept_by_nature"]) == {"ARRETE", "DECRET", "LOI"}

    l1_lines = out_l1.read_text(encoding="utf-8").splitlines()
    assert len(l1_lines) == 3
    for line in l1_lines:
        rec = json.loads(line)
        assert rec["schema"] == "aw.l1-markdown.v1"

    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".texts.jsonl.gz"))
    assert shard_files  # at least one shard written
    total_shard_rows = 0
    seen_pids = set()
    for fn in shard_files:
        with gzip.open(shard_dir / fn, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                assert "point_id" in row and "text" in row
                seen_pids.add(row["point_id"])
                total_shard_rows += 1
    assert total_shard_rows == 3
    assert len(seen_pids) == 3  # no collisions


def test_run_conserves_rows_scanned_equals_kept_plus_every_reject_reason(tnc_tree, tmp_path):
    # ADR-086 discipline in miniature: admitted + rejected must equal input.
    summary = X.run(root=str(tnc_tree), out_l1=str(tmp_path / "o.jsonl"),
                    shard_dir=str(tmp_path / "s"), rows_per_shard=100,
                    release="test-release-v1", log=lambda *a, **k: None)
    total_rejected = sum(summary["reject_reasons"].values())
    assert summary["kept"] + total_rejected == summary["scanned"]
