#!/usr/bin/env python3
"""Task 536: restore `published`, and stamp the density instrument.

`testdata/task536_sealed_stage_records.json` holds REAL rows, copied
verbatim (long text fields truncated) from disk on the build box:

  * `pubmed_pack` -- the first row of `_work/task417/pubmed/
    exact-production-field-aware-pack-v3-413a-stage-portable-v3/
    shard-0000.texts.jsonl.gz`, the sealed stage record `assembly_binder`
    actually staged for `https://pubmed.ncbi.nlm.nih.gov/21/`. It carries no
    `published` -- pubmed's stage record never did (task 535 sec.3.2).
  * `pubmed_registry_snapshot_row` -- the SAME url's row from the sealed
    `aw_e3` membership snapshot (`_preservation/aw_e3-membership-20260814/
    membership-00012.jsonl.gz`): `published: "1975"`. This is the exact
    artifact task 515's `build_serving_metadata_registry.py` turns into
    the URL-keyed sqlite `aw_serving_metadata.ServingMetadataRegistry` reads.
  * `ca_canadaca_chunk4` -- the first row of `_work/task427/chunk4-run-0001/
    ca_canadaca_cc/chunk2-00000-of-00001.jsonl.gz`, a real epoch-4
    `struct-md-768w-o0-...@4` chunk: `rule_epochs: {"chunk": 4}`,
    `chunk_n_words: 129`, `chunk_rule` all present on the SEALED STAGE
    record. Task 535 sec.2.2 found the served aw_e8 payload carries none of
    them -- `build_aw_b1.project_record` never read them.

Two defects, one mechanism each:

1. `published` was invisible to `aw_serving_metadata.hydrate()` because it was
   never in `REGISTRY_FIELDS` -- so even with a registry built from the exact
   snapshot that holds the 1975 stamp, pubmed's stage record (identity + text,
   no serving payload) had nowhere to recover it from.
2. `chunk_n_words` / `chunk_rule` / `rule_epochs.chunk` were computed by the
   chunker, sealed to the stage record, and then DISCARDED by
   `project_record`'s `out_payload` construction, which stamped a constant
   baseline (`ID.EPOCHS["chunk"] == 1`) regardless of what actually produced
   the row -- rule-vs-data-drift, one field further.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

import cytir.core.common as AC  # noqa: E402
from cytir.index import serving_metadata as SM  # noqa: E402
from cytir.index import b1_slice as B  # noqa: E402
from cytir.index import b1_projection as BP  # noqa: E402
from cytir.consent import build_gate as BG  # noqa: E402

FIXTURES = json.loads(
    (pathlib.Path(__file__).resolve().parent / "testdata"
     / "task536_sealed_stage_records.json").read_text())


class _Shim:
    """Exactly what `assembly_binder._SrcShim` hands the projection."""

    def __init__(self, rec, dim=8):
        self.id = rec["point_id"]
        self.payload = dict(rec)
        self.vector = {"dense": [0.1] * dim}


def _gate():
    return BG.ConsentGate(mode="trust", grace_days=30, cache_path="",
                         log=lambda *a: None)


def _project(rec, registry=None, origin="task417_families", published_map=None):
    stats = defaultdict(int)
    surface = rec.get("text") or rec.get("dense_text")
    point, meta = BP.project_record(
        _Shim(rec), origin=origin,
        corpus_row={"text": surface} if surface else None,
        published_map=published_map or {}, conf_map={}, gate=_gate(),
        stats=stats, metadata_registry=registry)
    return point, meta, stats


def _registry(tmp_path, rows, name="reg.sqlite"):
    db = tmp_path / name
    con = sqlite3.connect(str(db))
    cols = ", ".join(f"{f} TEXT" for f in SM.REGISTRY_FIELDS)
    con.execute(f"CREATE TABLE url_meta (k TEXT PRIMARY KEY, canon TEXT, "
                f"points INTEGER, {cols}) WITHOUT ROWID")
    for canon, vals in rows.items():
        con.execute(
            f"INSERT INTO url_meta VALUES (?,?,?,"
            f"{','.join('?' * len(SM.REGISTRY_FIELDS))})",
            (SM.url_key(canon), canon, 1,
             *[json.dumps(vals[f]) if vals.get(f) is not None else None
               for f in SM.REGISTRY_FIELDS]))
    con.commit()
    con.close()
    return SM.ServingMetadataRegistry(str(db))


# --------------------------------------------------------------- REGISTRY_FIELDS

def test_published_is_a_registry_field():
    """The defect, pinned as a fact: without this, nothing below is possible."""
    assert "published" in SM.REGISTRY_FIELDS


# ---------------------------------------------------------- published: the fix

def test_a_pubmed_row_with_a_snapshot_stamp_recovers_it(tmp_path):
    """A REAL pubmed stage record + the REAL snapshot row for the SAME url.

    Before task 536 this stayed unpublished forever: `published` was not in
    `REGISTRY_FIELDS`, so `hydrate()` could not see it even though the exact
    artifact holding the 1975 stamp was the registry's own input.
    """
    rec = FIXTURES["pubmed_pack"]
    snap = FIXTURES["pubmed_registry_snapshot_row"]
    assert "published" not in rec  # the stage record never carried one
    assert snap["published"] == "1975"
    canon = AC.canonical_url(rec["url"])
    assert canon == AC.canonical_url(snap["url"])

    reg = _registry(tmp_path, {canon: {"published": snap["published"],
                                       "title": snap["title"],
                                       "source_type": snap["source_type"],
                                       "trust_tier": snap["trust_tier"],
                                       "lang": snap["lang"]}})
    point, _meta, stats = _project(rec, registry=reg)
    assert point["payload"]["published"] == "1975"
    assert stats["meta_url_filled_published"] == 1


def test_a_row_the_snapshot_never_stamped_stays_absent(tmp_path):
    """Never invented. A registry hit that carries every OTHER field but no
    `published` leaves the row exactly as unpublished as it arrived --
    corpus-doctrine 8, absence is measured, never inferred."""
    rec = FIXTURES["pubmed_pack"]
    canon = AC.canonical_url(rec["url"])
    reg = _registry(tmp_path, {canon: {"title": "A Title", "lang": "en",
                                       "source_type": "reference",
                                       "trust_tier": 0.85}})
    point, _meta, stats = _project(rec, registry=reg)
    assert "published" not in point["payload"]
    assert stats["meta_unresolved_published"] == 1


def test_a_url_the_registry_has_never_seen_stays_absent_without_a_registry():
    """The pinned pre-536 behaviour for the no-registry path: unaffected."""
    rec = FIXTURES["pubmed_pack"]
    point, _meta, _stats = _project(rec, registry=None)
    assert "published" not in point["payload"]


def test_a_native_published_value_is_never_overwritten_by_the_registry(tmp_path):
    """Rule 1, same as every other REGISTRY_FIELDS entry: the record's own
    value always wins over the snapshot's."""
    rec = dict(FIXTURES["pubmed_pack"], published="2020-01-01")
    canon = AC.canonical_url(rec["url"])
    reg = _registry(tmp_path, {canon: {"published": "1975"}})
    point, _meta, stats = _project(rec, registry=reg)
    assert point["payload"]["published"] == "2020-01-01"
    assert "meta_url_filled_published" not in stats


def test_news_published_map_wired_through_the_binder_flag():
    """`assembly_binder.py`'s `--news-published` sidecar (distinct from the
    registry: dates for pages the sealed snapshot never held). A NEWS_PACKS
    row with no native `published` and no registry hit still resolves from
    the map keyed by canonical url -- the map `_stage_partition` builds via
    `_news_published_map` from `B.load_published_map`."""
    news_rec = {"url": "https://www.swissinfo.ch/eng/some-story/123",
                "point_id": "aaaaaaaa-0000-0000-0000-000000000001",
                "text": "a news story"}
    canon = AC.canonical_url(news_rec["url"])
    point, _meta, _stats = _project(
        news_rec, origin="aw_dswissinfo_cov",
        published_map={canon: "2026-08-01"})
    assert point["payload"]["published"] == "2026-08-01"


# ------------------------------------------------------- chunk instrument: the fix

def test_a_sealed_chunk4_records_own_epoch_is_no_longer_overwritten():
    """Before task 536: `out_payload["rule_epochs"]` was always
    `{**ID.EPOCHS, "tok": tok_epoch}` -- a constant baseline (`chunk: 1`)
    regardless of what actually produced the row. This REAL record declares
    `rule_epochs.chunk: 4` on its own sealed stage record; the served payload
    must say so too, or the fossil record lies (task 535 sec.2.2)."""
    rec = FIXTURES["ca_canadaca_chunk4"]
    assert rec["rule_epochs"]["chunk"] == 4  # the sealed record's OWN claim
    point, _meta, _stats = _project(rec)
    assert point["payload"]["rule_epochs"]["chunk"] == 4


def test_chunk_n_words_and_chunk_rule_pass_through_from_the_sealed_record():
    rec = FIXTURES["ca_canadaca_chunk4"]
    point, _meta, _stats = _project(rec)
    pl = point["payload"]
    assert pl["chunk_n_words"] == rec["chunk_n_words"] == 129
    assert pl["chunk_rule"] == rec["chunk_rule"]
    assert pl["rule_epochs"]["chunk"] == 4
    # every other EPOCHS key is untouched -- the set of keys never changes,
    # only the values this fix is allowed to override (task 514's own
    # regression guard depends on the key SET staying == ID.EPOCHS.keys())
    assert set(pl["rule_epochs"].keys()) == set(B.ID.EPOCHS.keys())


def test_an_absent_chunk_declaration_still_means_the_epochs_baseline():
    """Absent on the incoming record means the baseline, never a guess --
    same rule the `tok` epoch already followed (test_task515_carried_sparse)."""
    rec = dict(FIXTURES["ca_canadaca_chunk4"])
    rec.pop("rule_epochs")
    rec.pop("chunk_n_words")
    rec.pop("chunk_rule")
    point, _meta, _stats = _project(rec)
    assert point["payload"]["rule_epochs"]["chunk"] == B.ID.EPOCHS["chunk"]
    assert "chunk_n_words" not in point["payload"]
    assert "chunk_rule" not in point["payload"]


def test_prov_id_passes_through_when_the_sealed_record_carries_one():
    """ADR-089's discriminator (task 535 sec.2.2): once this reaches the
    payload, a re-materialised row is no longer indistinguishable from the
    production row it replaced."""
    rec = FIXTURES["pubmed_pack"]
    assert rec.get("prov_id")
    point, _meta, _stats = _project(rec)
    assert point["payload"]["prov_id"] == rec["prov_id"]
