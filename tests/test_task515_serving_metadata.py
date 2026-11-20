#!/usr/bin/env python3
"""The serving-metadata hydration contract (task 515).

`testdata/task515_sealed_stage_records.json` holds two REAL sealed
stage records, copied verbatim (bodies truncated) from the shards the binder
staged for task 514:

  * `task417_pack` -- the first row of
    `_work/task417/arxiv/field-aware-pack-v2-413a-stage-portable-v3/shard-0000.texts.jsonl.gz`
  * `task472_materializer` -- the first row of
    `_work/task472/cc-materializer-1024-embed-stage-v1/shard-0000.records.jsonl.gz`

`test_pack_record_ships_no_serving_signals_without_a_registry` is the test that
FAILS against the pinned-at-ec97ef9c projection: it asserts that the payload
`project_record` produces from a real pack record carries no `title`, no
`trust_tier` and no `source_type` -- which is the defect, stated as a fact --
and the tests below it assert that the same record WITH a registry does.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys

import pytest
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
     / "task515_sealed_stage_records.json").read_text())


class _Shim:
    """Exactly what `assembly_binder._SrcShim` hands the projection."""

    def __init__(self, rec, dim=8):
        self.id = rec["point_id"]
        self.payload = dict(rec)
        self.vector = {"dense": [0.1] * dim}


def _gate():
    return BG.ConsentGate(mode="trust", grace_days=30, cache_path="",
                         log=lambda *a: None)


def _project(rec, registry=None, origin="task417_families"):
    stats = defaultdict(int)
    # `assembly_binder._stage_partition` reads the record's OWN text surface:
    # `rec.get("text") or rec.get("dense_text")`. Reading only `text` here
    # would send a card down the no-text tail in the test while the binder
    # sends it down the corpus join in production -- the exact divergence task
    # 515 sec.3.2 found in the binder itself (339,307 cards, no posting).
    surface = rec.get("text") or rec.get("dense_text")
    point, meta = BP.project_record(
        _Shim(rec), origin=origin,
        corpus_row={"text": surface} if surface else None,
        published_map={}, conf_map={}, gate=_gate(), stats=stats,
        metadata_registry=registry)
    return point, meta, stats


def _registry(tmp_path, rows, domains=None, min_share=0.90):
    db = tmp_path / "reg.sqlite"
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
    dpath = None
    if domains is not None:
        dpath = tmp_path / "domains.json"
        dpath.write_text(json.dumps({"domains_map": domains}))
    return SM.ServingMetadataRegistry(str(db),
                                      domain_registry_path=str(dpath) if dpath else None,
                                      min_domain_share=min_share)


# ---------------------------------------------------------------- the defect

def test_pack_record_ships_no_serving_signals_without_a_registry():
    """The pinned projection's behaviour, pinned as a fact.

    This is the state task 514 shipped: 11,273,954 rows (36.5%) with no
    trust_tier and no source_type. If this test ever starts failing because
    the payload now HAS them, the projection changed and the change must be
    deliberate.
    """
    rec = FIXTURES["task417_pack"]
    point, _meta, _stats = _project(rec, registry=None)
    assert point is not None
    pl = point["payload"]
    assert "title" not in pl
    assert "trust_tier" not in pl
    assert "source_type" not in pl
    assert "lang" not in pl
    # ... while the identity and the URL come through fine, which is why no
    # count check could see it
    assert pl["url"] == rec["url"]
    assert pl["content_digest"]


def test_materializer_record_ships_no_serving_signals_without_a_registry():
    rec = FIXTURES["task472_materializer"]
    point, _meta, _stats = _project(rec, registry=None)
    pl = point["payload"]
    for f in ("title", "trust_tier", "source_type", "lang"):
        assert f not in pl, f


# ------------------------------------------------------------------- the fix

def test_url_registry_restores_the_serving_signals(tmp_path):
    rec = FIXTURES["task417_pack"]
    canon = AC.canonical_url(rec["url"])
    reg = _registry(tmp_path, {canon: {
        "title": "Sparsity-certifying Graph Decompositions",
        "lang": "en", "source_type": "reference", "trust_tier": 0.7,
        "country": "INTL", "license": "arXiv-metadata-CC0"}})
    point, _meta, stats = _project(rec, registry=reg)
    pl = point["payload"]
    assert pl["source_type"] == "reference"
    assert pl["trust_tier"] == 0.7
    assert pl["lang"] == "en"
    assert pl["title"]
    assert stats["meta_url_hit"] == 1
    assert stats["meta_row_missing_serving_signal"] == 0


def test_pack_title_comes_from_source_metadata_before_the_registry(tmp_path):
    """Rule 2: a record-native surface beats the snapshot.

    The pack shape carries the L1 parent's own title under `source_metadata`,
    which the projection never looked at. It is fresher than a snapshot title
    by construction -- the snapshot describes the points being replaced.
    """
    rec = FIXTURES["task417_pack"]
    canon = AC.canonical_url(rec["url"])
    reg = _registry(tmp_path, {canon: {
        "title": "A STALE TITLE FROM THE OLD CHUNKING",
        "source_type": "reference", "trust_tier": 0.7}})
    point, _meta, stats = _project(rec, registry=reg)
    assert point["payload"]["title"] == rec["source_metadata"]["title"]
    assert stats["meta_title_from_source_metadata"] == 1


def test_a_record_native_stamp_is_never_overwritten(tmp_path):
    """Rule 1. An aw_b1-native stage record (task427, task505, rii, news)
    must project byte-identically with and without a registry."""
    rec = dict(FIXTURES["task417_pack"],
               title="RECORD TITLE", source_type="official", trust_tier=1.0,
               lang="de")
    canon = AC.canonical_url(rec["url"])
    reg = _registry(tmp_path, {canon: {
        "title": "REGISTRY TITLE", "source_type": "reference",
        "trust_tier": 0.5, "lang": "en"}})
    with_reg, _m, _s = _project(rec, registry=reg)
    without, _m2, _s2 = _project(rec, registry=None)
    assert with_reg["payload"]["title"] == "RECORD TITLE"
    assert with_reg["payload"]["source_type"] == "official"
    assert with_reg["payload"]["trust_tier"] == 1.0
    assert with_reg["payload"]["lang"] == "de"
    assert with_reg["payload"] == without["payload"]


def test_domain_fallback_only_above_the_declared_share(tmp_path):
    """Rule 4. `rfc-editor.org` is the real case: the re-materialization moved
    from `/info/rfcNNN` to `/rfc/rfcN.txt`, so no URL in the snapshot matches
    and the domain stamp (official @ 0.96) is all there is."""
    rec = dict(FIXTURES["task417_pack"],
               url="https://www.rfc-editor.org/rfc/rfc1.txt")
    domains = {"rfc-editor.org": {
        "source_type": {"value": "official", "share": 0.96, "distinct": 2},
        "trust_tier": {"value": 1.0, "share": 0.96, "distinct": 3},
        "lang": {"value": "en", "share": 1.0, "distinct": 1},
        "license": {"value": None, "share": 0.96, "distinct": 2}}}

    strict = _registry(_mk(tmp_path / "a"), {}, domains, min_share=0.99)
    point, _m, stats = _project(rec, registry=strict)
    assert "source_type" not in point["payload"]
    assert stats["meta_unresolved_source_type"] == 1
    assert stats["meta_row_missing_serving_signal"] == 1

    loose = _registry(_mk(tmp_path / "b"), {}, domains, min_share=0.90)
    point2, _m2, stats2 = _project(rec, registry=loose)
    # `www.` is folded exactly as canonical_url folds it
    assert point2["payload"]["source_type"] == "official"
    assert point2["payload"]["trust_tier"] == 1.0
    assert stats2["meta_domain_filled_source_type"] == 1


def test_entity_card_source_type_is_never_inherited_through_a_join(tmp_path):
    """`entity_card` activates the ADR-060 card contract in `project_record`
    (paired dense/sparse corpus channels + a bundle_digest). A re-chunked
    passage that happens to share a URL with a card satisfies none of it, so
    inheriting the label raises rather than restoring a card. Refused and
    counted, never silently dropped."""
    rec = FIXTURES["task472_materializer"]
    canon = AC.canonical_url(rec["url"])
    reg = _registry(_mk(tmp_path / "c"), {canon: {
        "title": "Some Entity", "source_type": "entity_card",
        "trust_tier": 0.9, "lang": "en"}})
    point, _meta, stats = _project(rec, registry=reg)
    assert "source_type" not in point["payload"]
    assert stats["meta_refused_source_type_entity_card"] == 1
    # everything else on the row still lands
    assert point["payload"]["trust_tier"] == 0.9
    assert point["payload"]["title"] == "Some Entity"


def test_the_serving_domain_follows_canonical_urls_www_fold(tmp_path):
    """`_host_of` returns the URL netloc verbatim; every production ingester
    stamped the canonicalized host. The two agree on a record carrying its own
    `domain` and diverge on exactly the sealed record that carries none --
    2,638,905 staged rows, and `retrieval.py` looks the field up against
    tables (authority 18,312 hosts, hub-floor 20) that hold not one
    `www.`-prefixed key."""
    rec = dict(FIXTURES["task472_materializer"],
               url="https://www.fedlex.admin.ch/eli/cc/2022/491/de")
    point, _meta, _stats = _project(rec, registry=None)
    assert point["payload"]["domain"] == "fedlex.admin.ch"
    assert SM.serving_domain("https://www.rfc-editor.org/rfc/rfc1.txt") == \
        "rfc-editor.org"
    # and the fold is canonical_url's, not a second implementation beside it
    assert SM.serving_domain("http://WWW.Example.ORG/x?utm_source=a") == \
        "example.org"


def test_a_production_www_domain_is_preserved_through_the_registry(tmp_path):
    """Nine of production's 361,149 domains really are `www.`-prefixed
    (www.zh.ch and the cantonal Gesetzessammlungen, 63,150 points). Folding
    those would be a second drift in the other direction, so the exact value
    the snapshot holds for that URL wins over the fold."""
    rec = dict(FIXTURES["task472_materializer"],
               url="https://www.zh.ch/de/politik-staat/gesetze-beschluesse.html")
    canon = AC.canonical_url(rec["url"])
    reg = _registry(_mk(tmp_path / "d"), {canon: {
        "domain": "www.zh.ch", "source_type": "official", "trust_tier": 1.0,
        "title": "Gesetze", "lang": "de"}})
    point, _meta, _stats = _project(rec, registry=reg)
    assert point["payload"]["domain"] == "www.zh.ch"
    # ... and without the registry the fold applies
    plain, _m, _s = _project(rec, registry=None)
    assert plain["payload"]["domain"] == "zh.ch"


def test_url_key_is_a_mirror_contract():
    """The builder imports the key from the reader. A registry written under
    one definition and read under another is a silent 100% miss."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "tasks" / "515-scripts"))
    BR = pytest.importorskip("build_serving_metadata_registry")
    assert BR.url_key is SM.url_key
    assert tuple(BR.FIELDS) == tuple(SM.REGISTRY_FIELDS)
    assert SM.url_key("https://example.org/a") != SM.url_key("https://example.org/b")
    assert SM.url_key("https://example.org/a") == SM.url_key("https://example.org/a")


def _mk(p):
    p.mkdir(parents=True, exist_ok=True)
    return p
