#!/usr/bin/env python3
"""URL-group supersede for a sealed pack (ADR-067 am.1 point 4, task 712) and the resume
re-verification of an export (change 2).

REAL ROWS. `fixtures/loda-old-generation-rows.json` (read from the build-box twin 2026-10-05, no
text): real ids, real urls, real `provenance`/`tier`/`anchor` of five loda urls (old generation, the
cut row of the new generation) and one real entity card. Two things are placed by the tests and are
not twin facts: the card is given the url of an old-row group (to prove a card at the same url
survives), and snippets are set where a test needs words (the fixture holds none). The drain-level
tests use the real 3-row sealed pack of the task 629 test fixture (the one `test_promote_queue` uses).

The qdrant is an in-memory fake that answers exactly the calls the drain makes (id retrieve, url
scroll, delete, collection config); an unknown call fails the test. No network, no write outside
`tmp_path`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core import paths as P  # noqa: E402
from cytir.index import point_delete as PD  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import url_group_supersede as UGS  # noqa: E402

REAL_VOLATILE = UGS.volatile_reason          # the settings fixture stubs the module attribute

DOC = json.loads((HERE / "fixtures" / "loda-old-generation-rows.json").read_text())
ROWS = DOC["rows"]
CARD = next(r for r in ROWS if r["role"] == "card")
URL_A = "https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000043180265"   # 7 old + 1 new
URL_B = "https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000053100925"   # 1 old + 1 new
OLD_A = [r for r in ROWS if r["url"] == URL_A and r["role"] == "old"]
NEW_A = next(r for r in ROWS if r["url"] == URL_A and r["role"] == "new")
OLD_B = next(r for r in ROWS if r["url"] == URL_B and r["role"] == "old")
NEW_B = next(r for r in ROWS if r["url"] == URL_B and r["role"] == "new")
DENSE = {"dense": {"size": 2, "distance": "Cosine"}}
VEC = {"dense": [0.1, 0.2], "bm25": {"indices": [1], "values": [1.0]}}
WORDS25 = " ".join(f"mot{i}" for i in range(25))
GUARD_OK = {"docs_checked": 5, "docs_unchecked": 0, "docs_shortfall": 0,
            "threshold_declared": True, "min_coverage": 1.0, "lowest_coverage": 1.0}
NEW_ID = "11111111-2222-5333-8444-555555555551"      # a pack row: new text, so a new id
NEW_ID_2 = "11111111-2222-5333-8444-555555555552"


class Cluster:
    """Two in-memory collections, `http://twin` and `http://prod`, both named `aw_e8`."""

    def __init__(self, rows):
        self.colls: dict[str, dict[str, dict]] = {}
        for url in ("http://twin", "http://prod"):
            self.colls[url] = {r["id"]: {"payload": dict(r["payload"]), "vector": dict(VEC)}
                               for r in rows}
        self.calls: list[tuple[str, str]] = []
        self.break_vectors = False
        self.dense = {"http://twin": DENSE, "http://prod": DENSE}   # per side: the collection config
        self.prod_vector = None       # replaces the vectors prod answers with (a different shape)
        self.reads: list[tuple[str, int, bool]] = []      # (url, ids asked, with_vector)

    def api(self, url, path, body=None, method=None):
        coll = self.colls[url]
        if path == "/collections/aw_e8":
            return {"config": {"params": {"vectors": self.dense[url],
                                          "sparse_vectors": {"bm25": {}}}}}
        if url == "http://prod" and path.endswith("/points/scroll"):
            raise AssertionError("a scroll on prod")
        if path.endswith("/points/scroll"):
            vals = set(body["filter"]["must"][0]["match"]["any"])
            return {"points": [{"id": i, "payload": dict(r["payload"])} for i, r in coll.items()
                               if r["payload"].get("url") in vals], "next_page_offset": None}
        if "/points/delete" in path:
            self.calls.append((url, "delete"))
            for i in body["points"]:
                coll.pop(i, None)
            return {}
        if path.endswith("/points") and body and "ids" in body:
            keep, vec = body["with_payload"], body.get("with_vector")
            self.reads.append((url, len(body["ids"]), bool(vec)))
            out = []
            for i in body["ids"]:
                if i not in coll:
                    continue
                r = coll[i]
                row = {"id": i}
                if keep is True:
                    row["payload"] = dict(r["payload"])
                elif keep:
                    row["payload"] = {k: v for k, v in r["payload"].items() if k in keep}
                if vec:
                    v = dict(r["vector"])
                    if self.break_vectors:
                        v.pop("bm25")
                    if url == "http://prod" and self.prod_vector is not None:
                        v = dict(self.prod_vector)
                    row["vector"] = v
                out.append(row)
            return out
        raise AssertionError(f"unstubbed qdrant call {path}")

    def ids(self, url):
        return set(self.colls[url])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    export = tmp_path / "exports"
    export.mkdir()
    monkeypatch.setenv("AW_REMOVED_ROWS_EXPORT_DIR", str(export))
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AW_PROD_LOCK_HOST", "prod-test")
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    # tmp_path lies under /tmp, which the real durability check refuses (tested on its own)
    monkeypatch.setattr(UGS, "volatile_reason", lambda d: None)
    monkeypatch.setattr(UGS, "PROD_READ_SLEEP_S", 0)
    P.reset_settings_cache()
    yield export
    P.reset_settings_cache()


def make_cluster(monkeypatch, rows):
    c = Cluster(rows)
    monkeypatch.setattr(Q.PSP, "_api", c.api)
    monkeypatch.setattr(PD, "_api", c.api)
    return c


def pack_row(pid, url, *, prov="loda", snippet=WORDS25, **extra):
    payload = {"url": url, "provenance": prov, "source_type": "official", "tier": "A",
               "snippet": snippet, **extra}
    return pid, ("dense text", "sparse text", payload, {})


def joined_of(*items):
    return dict(items)


def served(*rows, snippets=None):
    """Fixture rows as the twin serves them; `snippets` = words per row (the fixture has none)."""
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        if snippets:
            r["payload"]["snippet"] = snippets
        out.append(r)
    return out


def plan(manifest, joined, cluster, tmp_path, *, write=True, sha="a" * 64, drop=frozenset(),
         contract=None, queue_dir=None, holds_skips=False):
    return UGS.plan_for_drain(manifest, manifest_sha=sha, joined=joined, drop_ids=drop,
                              twin_url="http://twin", artifact_root=tmp_path / "artifacts",
                              write=write,
                              contract_path=contract or tmp_path / "no-contract.json",
                              queue_dir=queue_dir, holds_skips=holds_skips)


_STAGE: dict = {}


@pytest.fixture(autouse=True)
def _stage_root(tmp_path):
    _STAGE["root"] = tmp_path / "stage"


def sealed(block, root=None):
    """A sealed stage dir as `family_stage` writes it: `PACK.json` + `chunks/shard-0000...`, the
    shard bound by `chunks_sha256`. `block` = the `pack_staged_text_guard` (None = a pack staged
    before the guard: no key; False = no PACK.json at all). Returns the `pack_chunks` path."""
    root = root or _STAGE["root"]
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    shard = root / "chunks" / "shard-0000.texts.jsonl.gz"
    shard.write_bytes(b"sealed rows")
    pj = root / "PACK.json"
    if block is False:
        pj.unlink(missing_ok=True)
    else:
        meta = {"schema": "aw.family-stage.v1", "pack_id": "loda-shard0",
                "chunks_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
        if block is not None:
            meta["pack_staged_text_guard"] = block
        pj.write_text(json.dumps(meta))
    return str(root / "chunks")


def manifest(docs=1, guard="default", **kw):
    """A loda_exact url-group manifest over a sealed stage whose guard block says `docs` docs were
    checked (one URL group by default)."""
    block = {**GUARD_OK, "docs_checked": docs} if guard == "default" else guard
    return {"pack_id": "loda-refresh", "family": "loda_exact", "collection": "aw_e8",
            "allow_supersede": "task 712 test", "supersede": "url-group",
            "pack_chunks": sealed(block), **kw}


# -------------------------------------------------------------- the declaration
def test_a_pack_without_the_field_is_not_a_supersede_pack():
    assert UGS.declared({"allow_supersede": "reason"}) is False
    assert UGS.declared({}) is False


def test_the_field_refuses_the_already_served_exclusion_beside_it():
    """Task 888 D1, the drain side: the excluded unchanged rows would be deleted unrewritten."""
    with pytest.raises(Q.QueueError, match="already_served_excluded_file"):
        UGS.declared({"supersede": "url-group", "allow_supersede": "r",
                      "already_served_excluded_file": "already-served.json"})


def test_the_field_needs_allow_supersede_and_a_known_value_and_a_sealed_pack():
    with pytest.raises(Q.QueueError, match="needs `allow_supersede`"):
        UGS.declared({"supersede": "url-group"})
    with pytest.raises(Q.QueueError, match="the only value"):
        UGS.declared({"supersede": "all", "allow_supersede": "r"})
    with pytest.raises(Q.QueueError, match="sealed packs"):
        UGS.declared({"supersede": "url-group", "allow_supersede": "r", "kind": "content_pack"})
    assert UGS.declared({"supersede": "url-group", "allow_supersede": "r"}) is True


# -------------------------------------------------------------------- the groups
def test_two_old_rows_and_the_cut_row_are_replaced_by_the_new_row_and_the_rest_survives(
        tmp_path, monkeypatch, settings):
    """URL A holds two old rows and the cut new-generation row (real ids). A card (the real card's
    payload, put at URL A), a row of another family and a row at URL B (not a pack url) are served
    too. The pack writes one new row at URL A."""
    a_rows = served(*OLD_A[:2], NEW_A)
    card = {"id": "00000000-0000-5000-8000-0000000000c1", "payload": {**CARD["payload"], "url": URL_A}}
    other = {"id": "00000000-0000-5000-8000-0000000000f1",
             "payload": {"url": URL_A, "provenance": "legi", "source_type": "official"}}
    b_row = served(OLD_B)[0]
    c = make_cluster(monkeypatch, a_rows + [card, other, b_row])
    got = plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path)
    assert got["ids"] == sorted(r["id"] for r in a_rows)
    assert card["id"] not in got["ids"] and other["id"] not in got["ids"]
    assert b_row["id"] not in got["ids"]
    s = got["summary"]
    assert (s["superseded_rows"], s["pack_urls"], s["kept_other_family"]) == (3, 1, 1)
    assert s["export"]["rows"] == 3 and s["export"]["sha256"]


def test_the_pack_own_ids_are_never_superseded_and_a_card_row_in_the_pack_opens_no_group(
        tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B, NEW_B))
    # the pack re-writes NEW_B under its own id (an unchanged row): only OLD_B is old
    got = plan(manifest(), joined_of(pack_row(NEW_B["id"], URL_B)), c, tmp_path)
    assert got["ids"] == [OLD_B["id"]]
    card_row = ("t", "t", {**CARD["payload"], "url": URL_B}, {})
    got2 = plan(manifest(docs=0), joined_of(("22222222-0000-5000-8000-000000000001", card_row)), c,
                tmp_path, sha="a" * 64)
    assert got2["ids"] == [] and got2["summary"]["pack_cards_skipped"] == 1


def test_a_served_row_without_provenance_refuses_the_pack_and_names_it(tmp_path, monkeypatch,
                                                                       settings):
    ghost = {"id": "00000000-0000-5000-8000-0000000000a1",
             "payload": {"url": URL_A, "source_type": "official"}}
    c = make_cluster(monkeypatch, served(*OLD_A[:1]) + [ghost])
    with pytest.raises(Q.QueueError, match="carry no `provenance`") as exc:
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path)
    assert ghost["id"] in str(exc.value)
    assert not list(settings.iterdir())          # nothing exported


def test_a_pack_row_without_provenance_refuses(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    pid, v = pack_row(NEW_ID, URL_B)
    v[2].pop("provenance")
    with pytest.raises(Q.QueueError, match="provenance '', the family is served as 'loda'"):
        plan(manifest(), joined_of((pid, v)), c, tmp_path)


def test_dropped_rows_open_no_group(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    got = plan(manifest(docs=0), joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path,
               drop=frozenset({NEW_ID}))
    assert got["ids"] == [] and got["summary"]["pack_urls"] == 0


# ------------------------------------------------------------ guards (b) and (c)
def test_a_replacement_that_loses_text_is_refused_and_names_the_url(tmp_path, monkeypatch,
                                                                    settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2], snippets=WORDS25))      # 50 old words
    with pytest.raises(Q.QueueError, match="would lose text") as exc:
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_A, snippet="trois mots seulement")),
             c, tmp_path)
    assert URL_A in str(exc.value)
    assert not list(settings.iterdir()) and not (tmp_path / "artifacts").exists()


def test_a_replacement_that_keeps_the_text_passes(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2], snippets=WORDS25))
    got = plan(manifest(), joined_of(pack_row(NEW_ID, URL_A), pack_row(NEW_ID_2, URL_A)), c,
               tmp_path)
    assert got["summary"]["word_coverage"]["violations"] == 0
    assert "snippet" in got["summary"]["word_coverage"]["sees"]


def test_the_stage_time_guard_is_required_and_a_shortfall_refuses(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    j = joined_of(pack_row(NEW_ID, URL_B))
    with pytest.raises(Q.QueueError, match="SEALED pack's own stage manifest"):
        plan(manifest(guard=False), j, c, tmp_path)               # no PACK.json
    with pytest.raises(Q.QueueError, match="carries no `pack_staged_text_guard`"):
        plan(manifest(guard=None), j, c, tmp_path)                # a pack staged before the guard
    short = manifest(guard={**GUARD_OK, "docs_checked": 1, "docs_shortfall": 2,
                            "lowest_coverage": 0.8})
    with pytest.raises(Q.QueueError, match="2 doc"):
        plan(short, j, c, tmp_path)
    unchecked = manifest(guard={**GUARD_OK, "docs_checked": 1, "docs_unchecked": 1})
    with pytest.raises(Q.QueueError, match="left unchecked"):
        plan(unchecked, j, c, tmp_path)
    assert not list(settings.iterdir())


def test_a_guard_block_typed_into_the_queue_manifest_does_not_pass(tmp_path, monkeypatch, settings):
    """Hand-typed: the sealed pack has NO block (staged before the guard), the queue manifest
    carries a perfect one. Refused. And a typed block that differs from the sealed one is refused."""
    c = make_cluster(monkeypatch, served(OLD_B))
    j = joined_of(pack_row(NEW_ID, URL_B))
    typed = manifest(guard=None, staged_text_guard={**GUARD_OK, "docs_checked": 1})
    with pytest.raises(Q.QueueError, match="carries no `pack_staged_text_guard`"):
        plan(typed, j, c, tmp_path)
    differs = manifest(staged_text_guard={**GUARD_OK, "docs_checked": 1, "min_coverage": 0.1})
    with pytest.raises(Q.QueueError, match="differs from the sealed pack's own block"):
        plan(differs, j, c, tmp_path)
    assert not list(settings.iterdir())


def test_the_guard_block_is_bound_to_the_pack_by_the_sha_of_its_chunks(tmp_path, monkeypatch,
                                                                       settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    j = joined_of(pack_row(NEW_ID, URL_B))
    man = manifest()
    (_STAGE["root"] / "chunks" / "shard-0000.texts.jsonl.gz").write_bytes(b"other rows")
    with pytest.raises(Q.QueueError, match="does not belong to this pack's rows"):
        plan(man, j, c, tmp_path)


def test_docs_checked_must_equal_the_url_groups_the_pack_writes(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    j = joined_of(pack_row(NEW_ID, URL_B))                     # one URL group
    with pytest.raises(Q.QueueError, match="checked 5 doc.*writes 1 URL group"):
        plan(manifest(docs=5), j, c, tmp_path)
    with pytest.raises(Q.QueueError, match="checked 0 doc.*writes 1 URL group"):
        plan(manifest(docs=0), j, c, tmp_path)
    assert not list(settings.iterdir()) and not (tmp_path / "artifacts").exists()


def test_a_family_with_no_threshold_is_blind_and_never_uses_url_group(tmp_path, monkeypatch,
                                                                      settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    blind = manifest(guard={**GUARD_OK, "docs_checked": 0, "docs_unchecked": 1,
                            "threshold_declared": False})
    with pytest.raises(Q.QueueError, match="BLIND"):
        plan(blind, joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path)
    declared_but_blind = manifest(guard={**GUARD_OK, "docs_checked": 0, "docs_unchecked": 1})
    with pytest.raises(Q.QueueError, match="left unchecked"):
        plan(declared_but_blind, joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path)


# ------------------------------------------------------------ the family allowlist
@pytest.mark.parametrize("family,prov", [("endorsement_discovery", "endorsement-discovery"),
                                         ("fedlex", None), ("wikipedia", None),
                                         (None, "loda")])
def test_a_family_not_on_the_allowlist_is_refused_before_any_read_or_write(
        tmp_path, monkeypatch, settings, family, prov):
    """CC rows carry the discovery-lane label, fedlex and Wikipedia rows carry none: `provenance`
    is not a family key for them. Refused with one line, before the twin is read."""
    c = make_cluster(monkeypatch, served(OLD_B))
    man = manifest()
    man["family"] = family
    with pytest.raises(Q.QueueError, match="not permitted for family") as exc:
        plan(man, joined_of(pack_row(NEW_ID, URL_B, prov=prov or "")), c, tmp_path)
    assert "loda_exact" in str(exc.value) and "\n" not in str(exc.value)
    assert c.reads == [] and c.calls == []
    assert not list(settings.iterdir()) and not (tmp_path / "artifacts").exists()


def test_the_allowlist_names_loda_and_the_four_pdf_families_with_their_served_provenance():
    """The four labels were read on the twin (`aw_e8`) on 2026-10-12: every `official` row at the
    200 door-covered urls carries exactly this label."""
    prov = {f: v for f, v in UGS.URL_GROUP_FAMILIES.items() if isinstance(v, str)}
    assert prov == {
        "loda_exact": "loda", "scratch_weko_docs_exact": "weko_docs",
        "scratch_bav_docs_exact": "bav_docs", "scratch_bazl_docs_exact": "bazl_docs",
        "scratch_elcom_docs_exact": "elcom_docs"}
    # task 1038: fedlex rows carry no `provenance`; the family is told by a url-scope rule
    assert UGS.URL_GROUP_FAMILIES["fedlex_exact"] is UGS.FEDLEX_SCOPE
    assert set(UGS.URL_GROUP_FAMILIES) - set(prov) == {"fedlex_exact"}
    for family, prov in UGS.URL_GROUP_FAMILIES.items():
        assert UGS.family_provenance({"family": family}) == prov


def test_a_pdf_family_pack_row_with_the_other_lane_label_refuses(tmp_path, monkeypatch, settings):
    """A weko pack row labelled with another family's served value is refused, not grouped."""
    c = make_cluster(monkeypatch, [])
    with pytest.raises(Q.QueueError, match="the family is served as 'weko_docs'"):
        plan(manifest(family="scratch_weko_docs_exact"),
             joined_of(pack_row(NEW_ID, URL_A, prov="bav_docs")), c, tmp_path)


def test_a_pack_row_whose_provenance_is_not_the_family_served_value_refuses(
        tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(OLD_B))
    with pytest.raises(Q.QueueError, match="the family is served as 'loda'"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_B, prov="endorsement-discovery")), c,
             tmp_path)


# ------------------------------------------------------------------ the export
def test_the_export_is_one_verified_file_with_every_vector_and_the_payload(tmp_path, monkeypatch,
                                                                           settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2], NEW_A))
    got = plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path)
    exp = got["export"]
    path = pathlib.Path(exp["path"])
    assert path.parent == settings and path.name.startswith("loda-refresh-aaaaaaaaaaaa")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == exp["sha256"]
    lines = [json.loads(x) for x in gzip.open(path, "rt")]
    assert sorted(x["id"] for x in lines) == got["ids"] and exp["rows"] == len(lines)
    assert lines[0]["vector"]["bm25"]["__sparse__"] is True and lines[0]["vector"]["dense"]
    assert lines[0]["payload"]["url"] == URL_A
    assert (tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").exists()


def test_a_dry_plan_writes_nothing_and_says_what_it_would_write(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    got = plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path, write=False)
    assert got["ids"] and got["summary"]["export"]["would_write"].startswith(str(settings))
    assert not list(settings.iterdir()) and not (tmp_path / "artifacts").exists()


def test_no_export_directory_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("AW_REMOVED_ROWS_EXPORT_DIR", raising=False)
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    P.reset_settings_cache()
    c = make_cluster(monkeypatch, served(OLD_B))
    with pytest.raises(Q.QueueError, match="durable export directory"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path)
    P.reset_settings_cache()


def test_an_export_directory_that_does_not_exist_refuses(tmp_path, monkeypatch, settings):
    monkeypatch.setenv("AW_REMOVED_ROWS_EXPORT_DIR", str(tmp_path / "nowhere"))
    P.reset_settings_cache()
    c = make_cluster(monkeypatch, served(OLD_B))
    with pytest.raises(Q.QueueError, match="not an existing directory"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path)


def test_an_incomplete_restore_row_refuses_and_leaves_no_state(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    c.break_vectors = True               # the twin answers rows without their sparse vector
    with pytest.raises(Q.QueueError, match="not a complete restore row"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_a_row_that_vanished_before_the_export_refuses(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    real = UGS.compute

    def compute_then_lose(*a, **k):
        out = real(*a, **k)
        c.colls["http://twin"].pop(out["ids"][0])
        return out
    monkeypatch.setattr(UGS, "compute", compute_then_lose)
    with pytest.raises(Q.QueueError, match="no longer on aw_e8"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_A)), c, tmp_path)


# ---------------------------------------------------- the write: twin first, then prod
class Apply:
    """Stands in for `apply_side` (its plan machinery needs a vectors dir): upserts the pack's
    rows into the side's collection and records the order of every write."""

    def __init__(self, cluster, rows, *, fail_prod=False):
        self.c, self.rows, self.fail_prod = cluster, rows, fail_prod

    def __call__(self, role, url, manifest, collection, *, batch, pause_s, ids_dir,
                 drop_ids=frozenset(), parallel=1, precomputed=None, write_attempt=None,
                 persisted_check=True, profile_collection=None):
        if role == "prod" and self.fail_prod:
            raise RuntimeError("prod went away")
        if write_attempt is not None:
            write_attempt.started = True
        self.c.calls.append((url, "upsert"))
        for pid, (_d, _s, payload, _r) in self.rows.items():
            self.c.colls[url][pid] = {"payload": dict(payload), "vector": dict(VEC)}
        n = len(self.c.colls[url])
        return ({"layout_before": {"points_count": 0}, "written": len(self.rows),
                 "write_seconds": 0.0, "layout_after": {"points_count": n}},
                pathlib.Path(ids_dir) / f"{role}-ids.json")


class FakeLock:
    def __init__(self, *_a, **_k):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


def wire(monkeypatch, cluster, rows, **kw):
    monkeypatch.setattr(Q, "apply_side", Apply(cluster, rows, **kw))
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", FakeLock)
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint",
                        lambda url, coll: {"points_count": len(cluster.colls[url])})

    class RB:
        from cytir.ops.write_lock import LOCKS
        assert_locks_free = staticmethod(lambda task: {})
        acquire = staticmethod(lambda *a, **k: None)
        release = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)


def run_live(cluster_plan, man, tmp_path, **kw):
    return Q.live_apply(pack_id=man["pack_id"], manifest=man, twin_url="http://twin",
                        prod_url="http://prod", task="712-test", batch=10, pause_s=0,
                        artifact_root=tmp_path / "artifacts", prod_collection="aw_e8",
                        supersede=cluster_plan, **kw)


def test_twin_first_new_rows_then_old_rows_deleted_then_prod_the_same(tmp_path, monkeypatch,
                                                                      settings):
    old = served(*OLD_A[:2], NEW_A)
    card = {"id": "00000000-0000-5000-8000-0000000000c1", "payload": {**CARD["payload"], "url": URL_A}}
    c = make_cluster(monkeypatch, old + [card])
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path)
    wire(monkeypatch, c, j)
    state = run_live(sp, man, tmp_path)
    for url in ("http://twin", "http://prod"):
        assert c.ids(url) == {NEW_ID, card["id"]}, url          # old gone, card and new there
    assert state["twin_receipt"]["superseded"] == 3 == state["prod_receipt"]["superseded"]
    order = [(u, w) for u, w in c.calls]
    assert order == [("http://twin", "upsert"), ("http://twin", "delete"),
                     ("http://prod", "upsert"), ("http://prod", "delete")]
    done = json.loads((tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").read_text())
    assert done["stage"] == "applied" and done["export"]["sha256"]


def test_no_supersede_argument_means_no_delete_at_all(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    j = joined_of(pack_row(NEW_ID, URL_A))
    wire(monkeypatch, c, j)
    state = run_live(None, manifest(), tmp_path)
    assert "superseded" not in state["twin_receipt"]
    assert not [x for x in c.calls if x[1] == "delete"]
    assert len(c.ids("http://twin")) == 3                       # both old rows still there


def test_a_tampered_export_stops_the_twin_before_any_delete(tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path)
    with open(sp["export"]["path"], "ab") as fh:
        fh.write(b"tamper")
    wire(monkeypatch, c, j)
    with pytest.raises(Q.QueueError, match="sha256"):
        run_live(sp, man, tmp_path)
    assert not [x for x in c.calls if x[1] == "delete"]


def test_resume_after_the_twin_leg_rereads_the_ids_and_finishes_prod(tmp_path, monkeypatch,
                                                                     settings):
    """Kill between the legs: the twin has lost its old rows, prod has not. A re-run cannot
    recompute the set from the twin; it reads the state, re-verifies the export, finishes prod."""
    c = make_cluster(monkeypatch, served(*OLD_A[:2], NEW_A))
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path)
    wire(monkeypatch, c, j, fail_prod=True)
    with pytest.raises(RuntimeError, match="prod went away"):
        run_live(sp, man, tmp_path)
    assert c.ids("http://twin") == {NEW_ID} and len(c.ids("http://prod")) == 3
    # the re-run
    again = plan(man, j, c, tmp_path)
    assert again["ids"] == sp["ids"] and again["summary"]["resumed_from_state"] == "deleting"
    wire(monkeypatch, c, j)
    run_live(again, man, tmp_path)
    assert c.ids("http://prod") == {NEW_ID}


def test_resume_with_a_tampered_export_is_refused_and_the_state_is_kept(tmp_path, monkeypatch,
                                                                        settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2], NEW_A))
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path)
    wire(monkeypatch, c, j, fail_prod=True)
    with pytest.raises(RuntimeError):
        run_live(sp, man, tmp_path)
    with open(sp["export"]["path"], "r+b") as fh:
        fh.seek(20)
        fh.write(b"\x00\x01\x02")
    with pytest.raises(Q.QueueError, match="sha256"):
        plan(man, j, c, tmp_path)
    assert (tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").exists()
    assert len(c.ids("http://prod")) == 3                       # prod untouched


def test_a_resealed_manifest_does_not_forget_a_supersede_that_began_deleting(tmp_path, monkeypatch,
                                                                              settings):
    c = make_cluster(monkeypatch, served(*OLD_A[:2]))
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path, sha="a" * 64)
    # planned, then refused before any write (a lock, a profile check): a reseal just replans
    again = plan(man, j, c, tmp_path, sha="b" * 64)
    assert again["ids"] == sp["ids"] and again["export"]["path"] != sp["export"]["path"]
    wire(monkeypatch, c, j, fail_prod=True)
    with pytest.raises(RuntimeError):
        run_live(again, man, tmp_path)            # the twin leg ran: state is now `deleting`
    with pytest.raises(Q.QueueError, match="resealed"):
        plan(man, j, c, tmp_path, sha="c" * 64)


# ------------------------------------------------------ change 2: content packs
def tomb(row, reason=Q.SUPERSEDED_GENERATION):
    return {"id": row["id"], "url": row["url"], "reason": reason}


def write_generation_export(tmp_path, rows):
    path = tmp_path / "gen-export.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "payload": dict(r["payload"]), "vector": VEC}) + "\n")
    return {"export": {"path": str(path), "rows": len(rows),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_reverify_generation_export_passes_an_intact_export_and_ignores_other_reasons(
        tmp_path, monkeypatch):
    c = make_cluster(monkeypatch, [])
    del c
    man = write_generation_export(tmp_path, OLD_A)
    out = UGS.reverify_generation_export(man, [tomb(r) for r in OLD_A], twin_url="http://twin",
                                         twin_collection="aw_e8")
    assert out["rows"] == len(OLD_A)
    assert UGS.reverify_generation_export({}, [tomb(OLD_A[0], "aged_out")], twin_url="http://twin",
                                          twin_collection="aw_e8") is None


def test_the_content_pack_resume_refuses_a_tampered_export_before_any_prod_write(
        tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, [])
    man = {**write_generation_export(tmp_path, OLD_A), "kind": "content_pack",
           "pack_id": "loda-old-gen", "collection": "aw_e8", "pack_points": "unused",
           "tombstones": str(tmp_path / "tombs.jsonl")}
    tombs = [tomb(r) for r in OLD_A]
    (tmp_path / "tombs.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tombs))
    with open(man["export"]["path"], "ab") as fh:
        fh.write(b"tamper")
    monkeypatch.setattr(Q, "load_content_pack_points", lambda path: [])
    wire(monkeypatch, c, {})
    monkeypatch.setattr(Q.PSP, "already_present", lambda url, coll, ids, **k:
                        [] if "twin" in url else list(ids))
    writes: list = []
    monkeypatch.setattr(Q, "apply_content_pack_side", lambda *a, **k: writes.append(1))
    state = {"stage": "twin-write-complete", "pack_ids": [], "tomb_ids": [t["id"] for t in tombs]}
    with pytest.raises(Q.QueueError, match="sha256"):
        Q.resume_content_pack_prod_only(
            pack_id="loda-old-gen", manifest=man, twin_url="http://twin", prod_url="http://prod",
            task="712-test", batch=10, pause_s=0, artifact_root=tmp_path / "artifacts",
            state_record=state, prod_collection="aw_e8")
    assert writes == []


def test_the_content_pack_replay_from_twin_write_started_refuses_a_tampered_export_on_the_prod_leg(
        tmp_path, monkeypatch, settings):
    c = make_cluster(monkeypatch, [])
    man = {**write_generation_export(tmp_path, OLD_A), "collection": "aw_e8"}
    tombs = [tomb(r) for r in OLD_A]
    with open(man["export"]["path"], "ab") as fh:
        fh.write(b"tamper")
    wire(monkeypatch, c, {})
    monkeypatch.setattr(Q.PSP, "already_present", lambda *a, **k: [])
    monkeypatch.setattr(Q, "apply_content_pack_side", lambda role, *a, **k: {
        "role": role, "written": 0, "tombstoned": 0, "write_seconds": 0,
        "layout_before": {"points_count": 1}, "layout_after": {"points_count": 1}})
    monkeypatch.setattr(Q, "verify_written_form", lambda *a, **k: {})
    with pytest.raises(Q.QueueError, match="sha256"):
        Q.live_apply_content_pack(
            pack_id="loda-old-gen", manifest=man, twin_url="http://twin", prod_url="http://prod",
            task="712-test", batch=10, pause_s=0, artifact_root=tmp_path / "artifacts",
            points=[], tombstones=tombs, prod_collection="aw_e8", manifest_sha="s" * 64)


# --------------------------------------------------------------- drain level
SEALED = HERE / "fixtures" / "promote" / "task629_sealed_pack"
needs_pack = pytest.mark.skipif(not list((SEALED / "vectors").glob("*.vectors.manifest.json")),
                                reason="the real sealed-pack fixture is not in this tree (the public cut)")


def drain_fixture(tmp_path, monkeypatch, settings, *, guard="default", **manifest_extra):
    """The real 3-row sealed pack, copied into a stage dir with a `PACK.json` (the guard block,
    one doc per URL group) and queued as family `loda_exact` with `allow_supersede`; the twin
    serves one old row of the pack's first url. `guard`: a block, None (no key) or "default"."""
    stub_plan_reads(monkeypatch)
    stage = tmp_path / "stage"
    shutil.copytree(SEALED / "chunks", stage / "chunks")
    shard = stage / "chunks" / "shard-0000.texts.jsonl.gz"
    groups = len(set(real_pack_urls(monkeypatch).values()))
    block = {**GUARD_OK, "docs_checked": groups} if guard == "default" else guard
    meta = {"schema": "aw.family-stage.v1", "chunks_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
    if block is not None:
        meta["pack_staged_text_guard"] = block
    (stage / "PACK.json").write_text(json.dumps(meta))
    man = {"schema": Q.MANIFEST_SCHEMA, "pack_id": "loda-refresh",
           "family": "loda_exact", "pack_chunks": str(stage / "chunks"),
           "vectors": str(SEALED / "vectors"), "mrl": 256, "collection": "aw_e8",
           "identity_check": "pack-declared", "identity_reason": "test: ids not reproducible",
           "family_payload": {"source_type": "docs", "trust_tier": 0.6},
           "title_from": ["source_metadata.title"], "allow_supersede": "task 712 test",
           **manifest_extra}
    queue = tmp_path / "queue"
    queue.mkdir()
    pack = queue / man["pack_id"]
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps(man, indent=1))
    (pack / "READY").write_text(hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest() + "\n")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110", "pr": 302,
                                    "profile": "mechanical"}))
    return queue, contract


def stub_plan_reads(monkeypatch):
    for name, val in (("target_width", lambda *a, **k: 256),
                      ("already_served_urls", lambda *a, **k: [])):
        monkeypatch.setattr(Q.PSP, name, val)
    monkeypatch.setattr(Q.PSP.PI, "assert_pages_persisted", lambda *a, **k: {"verdict": "PASS"})
    monkeypatch.setattr(Q.PSP.PC, "assert_scoreable", lambda *a, **k: {"verdict": "PASS"})


def real_pack_urls(monkeypatch):
    stub_plan_reads(monkeypatch)
    monkeypatch.setattr(Q.PSP, "already_present", lambda *a, **k: [])
    monkeypatch.setattr(Q.PSP.LAYOUT, "fingerprint", lambda *a, **k: {})
    r = Q.dry_plan({"pack_chunks": str(SEALED / "chunks"), "vectors": str(SEALED / "vectors"),
                    "mrl": 256, "collection": "aw_e8", "identity_check": "pack-declared",
                    "identity_reason": "t", "allow_supersede": "r",
                    "family_payload": {"source_type": "docs", "trust_tier": 0.6},
                    "title_from": ["source_metadata.title"]}, "http://x")
    return {pid: v[2]["url"] for pid, v in r["_joined"].items()}


def give_the_pack_a_family(monkeypatch):
    """The real Pool A rows carry NO `provenance` (a finding in itself: such a pack cannot
    declare `supersede: url-group`). The family is set on the planned rows here, as a family whose
    records carry one would."""
    real = Q.dry_plan

    def with_provenance(*a, **k):
        r = real(*a, **k)
        for v in r["_joined"].values():
            v[2]["provenance"] = "loda"
        return r
    monkeypatch.setattr(Q, "dry_plan", with_provenance)


def old_row_at(url, rid="00000000-0000-5000-8000-0000000000e1"):
    return {"id": rid, "role": "old", "payload": {"url": url, "provenance": "loda",
                                                   "source_type": "docs", "snippet": "old text"}}


def run_drain(queue, contract, tmp_path, **kw):
    return Q.drain(queue, twin_url="http://twin", prod_url="http://prod", task="712-test",
                   apply=kw.pop("apply", True), batch=10, pause_s=0,
                   artifact_root=tmp_path / "artifacts", contract_path=contract,
                   refresh_index=False)


@needs_pack
def test_drain_replaces_the_old_row_of_a_real_pack_url_and_receipts_it(tmp_path, monkeypatch,
                                                                      settings):
    urls = real_pack_urls(monkeypatch)
    give_the_pack_a_family(monkeypatch)
    first_url = sorted(urls.values())[0]
    old = old_row_at(first_url)
    c = make_cluster(monkeypatch, [old])
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    # the real apply_side runs its own plan(): give it the same stubs, and let it upsert into c
    monkeypatch.setattr(Q.PSP, "already_present", lambda *a, **k: [])
    monkeypatch.setattr(Q.PSP, "upsert", lambda url, coll, pts, batch, **k: [
        c.colls[url].__setitem__(p["id"], {"payload": p["payload"], "vector": dict(VEC)})
        for p in pts] and len(pts))
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", FakeLock)
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q.LAYOUT, "fingerprint",
                        lambda url, coll: {"points_count": len(c.colls[url])})

    class RB:
        from cytir.ops.write_lock import LOCKS
        assert_locks_free = staticmethod(lambda task: {})
        acquire = staticmethod(lambda *a, **k: None)
        release = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})
    res = run_drain(queue, contract, tmp_path)
    rec = res[0]
    assert rec["status"] == "LANDED", rec
    s = rec["url_group_supersede"]
    assert s["superseded_rows"] == 1 and s["export"]["sha256"]
    for url in ("http://twin", "http://prod"):
        assert old["id"] not in c.ids(url) and set(urls) <= c.ids(url)
    assert rec["twin"]["superseded"] == 1 and rec["prod"]["superseded"] == 1
    prod = s["prod"]
    assert (prod["requested"], prod["present"], prod["absent"]) == (1, 1, 0)
    assert pathlib.Path(prod["export"]["path"]).name.endswith("-prod-removed-rows.jsonl.gz")


@needs_pack
def test_drain_refuses_a_supersede_pack_without_the_stage_guard_and_writes_nothing(
        tmp_path, monkeypatch, settings):
    urls = real_pack_urls(monkeypatch)
    c = make_cluster(monkeypatch, [old_row_at(sorted(urls.values())[0])])
    give_the_pack_a_family(monkeypatch)
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group",
                                    guard=None)
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})
    monkeypatch.setattr(Q, "live_apply", lambda **k: pytest.fail("must not write"))
    res = run_drain(queue, contract, tmp_path)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "supersede_check"
    assert "pack_staged_text_guard" in res[0]["reason"]
    assert len(c.ids("http://twin")) == 1 and not list(settings.iterdir())
    # retried every drain: the stage is not one a refusal is cached under
    assert "supersede_check" not in Q._CACHEABLE_REFUSAL_STAGES


@needs_pack
def test_a_dry_drain_proves_the_supersede_and_writes_no_export(tmp_path, monkeypatch, settings):
    urls = real_pack_urls(monkeypatch)
    give_the_pack_a_family(monkeypatch)
    c = make_cluster(monkeypatch, [old_row_at(sorted(urls.values())[0])])
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})
    monkeypatch.setattr(Q, "live_apply", lambda **k: pytest.fail("must not write"))
    res = run_drain(queue, contract, tmp_path, apply=False)
    assert res[0]["status"] == "PROVED-DRY-RUN"
    assert res[0]["url_group_supersede"]["export"]["would_write"]
    assert not list(settings.iterdir()) and len(c.ids("http://twin")) == 1


@needs_pack
def test_a_pack_with_allow_supersede_and_no_field_keeps_its_old_meaning(tmp_path, monkeypatch,
                                                                      settings):
    urls = real_pack_urls(monkeypatch)
    make_cluster(monkeypatch, [old_row_at(sorted(urls.values())[0])])
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings)       # allow_supersede only
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})
    seen = {}
    monkeypatch.setattr(Q, "live_apply", lambda **k: seen.update(k) or {"twin_receipt": {}, "prod_receipt": {}})
    monkeypatch.setattr(UGS, "plan_for_drain", lambda *a, **k: pytest.fail("not reached"))
    res = run_drain(queue, contract, tmp_path)
    assert res[0]["status"] == "LANDED" and "supersede" not in seen
    assert "url_group_supersede" not in res[0]


# ===================================================== review fixes 2 and 3: prod presence + export
def live_with_prod(monkeypatch, tmp_path, rows, *, drop_on_prod=(), prod_payload=None):
    """The 3-row loda url A scenario (two old rows + the cut row), a plan, and wiring for a live
    apply. `drop_on_prod` = ids the twin holds and prod does not (a twin-only id)."""
    c = make_cluster(monkeypatch, rows)
    for i in drop_on_prod:
        c.colls["http://prod"].pop(i)
    for i, pl in (prod_payload or {}).items():
        c.colls["http://prod"][i]["payload"] = pl
    j = joined_of(pack_row(NEW_ID, URL_A))
    man = manifest()
    sp = plan(man, j, c, tmp_path)
    wire(monkeypatch, c, j)
    return c, j, man, sp


def test_a_twin_only_id_is_absent_on_prod_not_counted_as_deleted(tmp_path, monkeypatch, settings):
    """Prod holds two of the three superseded ids. The prod count bound is the MEASURED 2, the
    receipt says present 2 / absent 1, prod's own export holds 2 rows."""
    rows = served(*OLD_A[:2], NEW_A)
    lost = rows[0]["id"]
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows, drop_on_prod=[lost])
    state = run_live(sp, man, tmp_path)
    assert state["twin_receipt"]["superseded"] == 3
    assert state["prod_receipt"]["superseded"] == 2            # measured, not the requested 3
    prod = sp["summary"]["prod"]
    assert (prod["requested"], prod["present"], prod["absent"]) == (3, 2, 1)
    assert prod["export"]["rows"] == 2 and prod["export"]["sha256"]
    assert c.ids("http://prod") == {NEW_ID}
    saved = json.loads((tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").read_text())
    assert saved["prod"]["present"] == 2 and saved["prod"]["absent"] == 1
    assert saved["summary"]["prod"]["export"]["path"] == prod["export"]["path"]
    assert saved["stage"] == "applied"


def test_the_prod_bound_is_the_measured_number(tmp_path, monkeypatch, settings):
    """The old bound allowed the twin's count on prod. With one id absent on prod (measured 2), a
    prod that falls by 3 is refused."""
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows, drop_on_prod=[rows[0]["id"]])
    filler = [f"00000000-0000-5000-8000-00000000ff{n:02d}" for n in range(2)]
    for i in filler:                               # unrelated rows only prod holds
        c.colls["http://prod"][i] = {"payload": {"url": "https://x.example/" + i}, "vector": VEC}
    real_delete = PD.delete_points_verified

    def delete_and_lose_one_more(url, coll, ids, **k):
        n = real_delete(url, coll, ids, **k)
        if url == "http://prod":
            for i in filler:                       # two unrelated rows vanish inside the transaction
                c.colls[url].pop(i, None)
        return n
    monkeypatch.setattr(PD, "delete_points_verified", delete_and_lose_one_more)
    with pytest.raises(Q.QueueError, match="points_count FELL"):
        run_live(sp, man, tmp_path)


def test_the_prod_export_holds_prods_own_rows_not_the_twins(tmp_path, monkeypatch, settings):
    rows = served(*OLD_A[:2], NEW_A)
    pid = rows[0]["id"]
    normalized = {"url": rows[0]["payload"]["url"], "provenance": "loda", "tier": "B",
                  "prod_only_marker": True}
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows, prod_payload={pid: normalized})
    c.colls["http://prod"][pid]["vector"] = {"dense": [0.5, 0.6],
                                              "bm25": {"indices": [9, 8], "values": [2.0, 3.0]}}
    run_live(sp, man, tmp_path)
    twin_x = {x["id"]: x for x in (json.loads(l) for l in gzip.open(sp["export"]["path"], "rt"))}
    prod_x = {x["id"]: x for x in (json.loads(l)
                                   for l in gzip.open(sp["prod"]["export"]["path"], "rt"))}
    assert twin_x[pid]["payload"] != prod_x[pid]["payload"]
    assert prod_x[pid]["payload"] == normalized
    assert prod_x[pid]["vector"]["bm25"]["indices"] == [9, 8]
    assert pathlib.Path(sp["prod"]["export"]["path"]).parent == pathlib.Path(sp["export"]["path"]).parent


def test_prod_is_read_by_id_in_small_batches_never_by_scroll(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(UGS, "PROD_EXPORT_BATCH", 2)
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows)
    run_live(sp, man, tmp_path)               # the Cluster raises on a scroll against prod
    prod_reads = [r for r in c.reads if r[0] == "http://prod"]
    assert any(n == 2 and vec for _u, n, vec in prod_reads)       # the export read, in batches of 2
    assert all(n <= 500 for _u, n, _v in prod_reads)
    assert UGS.PROD_PRESENCE_BATCH <= 500 and 0 < UGS.PROD_READ_SLEEP_S + 1


def test_a_prod_export_that_does_not_verify_stops_before_any_prod_delete_and_keeps_the_state(
        tmp_path, monkeypatch, settings):
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows)
    c.prod_vector = {"dense": [0.1], "bm25": {"indices": [1], "values": [1.0]}}   # 1 dim, config says 2
    with pytest.raises(Q.QueueError, match="not a complete restore row"):
        run_live(sp, man, tmp_path)
    assert len(c.ids("http://prod")) == 4                       # NEW_ID upserted, none deleted
    assert OLD_A[0]["id"] in c.ids("http://prod")
    assert ("http://prod", "delete") not in c.calls
    saved = json.loads((tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").read_text())
    assert saved["stage"] == "deleting" and saved["prod"] is None     # the twin leg ran; state kept


def die_on_prod_delete(monkeypatch):
    real_delete = PD.delete_points_verified

    def die(url, coll, ids, **k):
        if url == "http://prod":
            raise RuntimeError("prod went away")
        return real_delete(url, coll, ids, **k)
    monkeypatch.setattr(PD, "delete_points_verified", die)
    return real_delete


def test_resume_reverifies_the_prod_export_before_the_prod_leg(tmp_path, monkeypatch, settings):
    """Prod export written, then the prod delete dies. The export is tampered with. The re-run
    stops on its sha256 before it deletes anything on prod and the state is kept."""
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows)
    real_delete = die_on_prod_delete(monkeypatch)
    with pytest.raises(RuntimeError, match="prod went away"):
        run_live(sp, man, tmp_path)
    saved = json.loads((tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").read_text())
    assert saved["prod"]["export"]["sha256"] and saved["prod"]["present"] == 3
    with open(saved["prod"]["export"]["path"], "ab") as fh:
        fh.write(b"tamper")
    monkeypatch.setattr(PD, "delete_points_verified", real_delete)
    again = plan(man, j, c, tmp_path)
    assert again["summary"]["resumed_from_state"] == "deleting" and again["prod"]["present"] == 3
    wire(monkeypatch, c, j)
    with pytest.raises(Q.QueueError, match="sha256"):
        run_live(again, man, tmp_path)
    assert len(c.ids("http://prod")) == 4 and OLD_A[0]["id"] in c.ids("http://prod")
    assert (tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json").exists()


def test_resume_with_an_intact_prod_export_finishes_prod(tmp_path, monkeypatch, settings):
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows)
    real_delete = die_on_prod_delete(monkeypatch)
    with pytest.raises(RuntimeError):
        run_live(sp, man, tmp_path)
    monkeypatch.setattr(PD, "delete_points_verified", real_delete)
    again = plan(man, j, c, tmp_path)
    wire(monkeypatch, c, j)
    state = run_live(again, man, tmp_path)
    assert c.ids("http://prod") == {NEW_ID} and state["prod_receipt"]["superseded"] == 3


# ============================================ review fix 6: a durable export dir, the real queue dir
def test_an_export_dir_under_tmp_on_tmpfs_or_in_the_repo_is_not_durable(tmp_path, monkeypatch):
    assert REAL_VOLATILE(pathlib.Path("/tmp")) == "lies under /tmp"
    assert REAL_VOLATILE(tmp_path) == "lies under /tmp"
    monkeypatch.setattr(UGS, "_fstype_of", lambda d: "tmpfs")
    assert "tmpfs" in REAL_VOLATILE(pathlib.Path("/home"))
    monkeypatch.setattr(UGS, "_fstype_of", lambda d: "ext4")
    # /tmp is checked first, so a checkout under /tmp (the cut) reports "lies under /tmp"
    tmp_root = pathlib.Path("/tmp").resolve()
    here_under_tmp = HERE.resolve() == tmp_root or tmp_root in HERE.resolve().parents
    assert REAL_VOLATILE(HERE) == ("lies under /tmp" if here_under_tmp else "lies inside the repo checkout")
    assert REAL_VOLATILE(pathlib.Path("/home")) is None


def test_the_real_check_refuses_a_volatile_export_dir(tmp_path, monkeypatch):
    d = tmp_path / "exports"
    d.mkdir()
    monkeypatch.setenv("AW_REMOVED_ROWS_EXPORT_DIR", str(d))
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    P.reset_settings_cache()
    with pytest.raises(Q.QueueError, match="lies under /tmp"):
        UGS.export_dir()
    P.reset_settings_cache()


def test_the_export_dir_is_checked_against_the_queue_this_drain_reads(tmp_path, monkeypatch,
                                                                     settings):
    """`--queue-dir` is not the configured queue: the setting alone would let this one through."""
    queue = tmp_path / "other-queue"
    (queue / "exports").mkdir(parents=True)
    monkeypatch.setenv("AW_REMOVED_ROWS_EXPORT_DIR", str(queue / "exports"))
    P.reset_settings_cache()
    assert UGS.export_dir() == queue / "exports"                      # the setting alone passes
    with pytest.raises(Q.QueueError, match="inside the promotion queue"):
        UGS.export_dir(queue)
    c = make_cluster(monkeypatch, served(OLD_B))
    with pytest.raises(Q.QueueError, match="inside the promotion queue"):
        plan(manifest(), joined_of(pack_row(NEW_ID, URL_B)), c, tmp_path, queue_dir=queue)


def test_the_preservation_scope_no_longer_says_net_new_append_for_a_supersede(tmp_path,
                                                                             monkeypatch, settings):
    rows = served(*OLD_A[:2], NEW_A)
    c, j, man, sp = live_with_prod(monkeypatch, tmp_path, rows)
    seen = {}
    real = Q.TFA.twin_first

    def spy(**k):
        seen["scope"] = k["preservation_verifier"]()["scope"]
        return real(**k)
    monkeypatch.setattr(Q.TFA, "twin_first", spy)
    run_live(sp, man, tmp_path)
    assert "net-new" not in seen["scope"] and "url-group" in seen["scope"]
    seen.clear()
    run_live(None, man, tmp_path)
    assert "net-new append" in seen["scope"]


# ==================================== review fix 6: a crashed url-group pack resumes through drain
def wire_real_drain(monkeypatch, c, *, prod_upsert_fails: list):
    """The wiring of the drain-level tests (real `apply_side`, `plan`; stubbed qdrant writes).
    `prod_upsert_fails`: a one-item list; while it holds a truthy item the prod upsert dies."""
    monkeypatch.setattr(Q.PSP, "already_present", lambda *a, **k: [])

    def upsert(url, coll, pts, batch, **k):
        if url == "http://prod" and prod_upsert_fails and prod_upsert_fails[0]:
            raise RuntimeError("prod went away")
        for p in pts:
            c.colls[url][p["id"]] = {"payload": p["payload"], "vector": dict(VEC)}
        return len(pts)
    monkeypatch.setattr(Q.PSP, "upsert", upsert)
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", FakeLock)
    monkeypatch.setattr(Q, "_revalidate_prod_collection", lambda *a, **k: None)
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *a, **k: "aw_e8")
    monkeypatch.setattr(Q, "payload_profile_preflight_both", lambda *a, **k: {})
    monkeypatch.setattr(Q.LAYOUT, "fingerprint",
                        lambda url, coll: {"points_count": len(c.colls[url])})

    class RB:
        from cytir.ops.write_lock import LOCKS
        assert_locks_free = staticmethod(lambda task: {})
        acquire = staticmethod(lambda *a, **k: None)
        release = staticmethod(lambda *a, **k: None)
    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q.MCC, "evaluate", lambda rows, **k: {
        "verdict": "PASS", "problems": {}, "other_failing_share": 0, "fail_share_max": 1,
        "word_floor_share": 0, "word_floor_fail_share_max": 1, "rows_scored": len(rows)})


@needs_pack
def test_a_crashed_url_group_pack_resumes_through_drain_itself(tmp_path, monkeypatch, settings):
    """First drain: the twin leg lands (new rows in, old row deleted, state `deleting`), the prod
    leg dies. Second drain, same queue: the state is read, prod gets its export and the delete,
    the pack ends LANDED and DONE."""
    urls = real_pack_urls(monkeypatch)
    give_the_pack_a_family(monkeypatch)
    old = old_row_at(sorted(urls.values())[0])
    c = make_cluster(monkeypatch, [old])
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    fails = [True]
    wire_real_drain(monkeypatch, c, prod_upsert_fails=fails)
    with pytest.raises(RuntimeError, match="prod went away"):
        run_drain(queue, contract, tmp_path)
    assert old["id"] not in c.ids("http://twin") and old["id"] in c.ids("http://prod")
    state_file = tmp_path / "artifacts" / "loda-refresh-url-group-supersede.json"
    assert json.loads(state_file.read_text())["stage"] == "deleting"
    assert not (queue / "loda-refresh" / "DONE").exists()
    fails[0] = False
    res = run_drain(queue, contract, tmp_path)
    rec = res[0]
    assert rec["status"] == "LANDED", rec
    assert rec["url_group_supersede"]["resumed_from_state"] == "deleting"
    assert old["id"] not in c.ids("http://prod") and set(urls) <= c.ids("http://prod")
    assert rec["url_group_supersede"]["prod"]["present"] == 1
    assert (queue / "loda-refresh" / "DONE").exists()
    assert json.loads(state_file.read_text())["stage"] == "applied"


def test_the_sealed_stage_schema_is_the_one_family_stage_writes():
    from cytir.sources import family_stage as FS
    assert UGS.SEALED_STAGE_SCHEMA == FS.STAGE_SCHEMA


# ------------------------------------------------- guard (b): skip per URL (task 876, ADR-114 am.2)
PDF_FAMILY = "scratch_weko_docs_exact"
SWISSCOM = "https://www.weko.admin.ch/dam/de/sd-web/swisscom/verfuegung.pdf"
SKIP_PAGE = "https://www.weko.admin.ch/dam/de/sd-web/%s.pdf"


def _uuid(n):
    return f"00000000-0000-5000-8000-{n:012d}"


def pdf_world(monkeypatch, *, small=(SWISSCOM,), clean=9):
    """A weko pack over `len(small) + clean` PDF urls. A clean url: 4 old garbage rows of 25
    snippet words, 4 new rows of 25 (ratio 1.0). A `small` url (the Swisscom decision): 10 old
    garbage rows, 2 new large rows (ratio 50 / 250 = 0.2). Returns `(cluster, joined, urls)`."""
    old, items, urls = [], [], list(small) + [SKIP_PAGE % f"doc{i}" for i in range(clean)]
    n = 1
    for u in urls:
        n_old, n_new = (10, 2) if u in small else (4, 4)
        for _ in range(n_old):
            old.append({"id": _uuid(n), "payload": {
                "url": u, "provenance": "weko_docs", "source_type": "official",
                "snippet": WORDS25}})
            n += 1
        for _ in range(n_new):
            items.append(pack_row(_uuid(10_000 + n), u, prov="weko_docs"))
            n += 1
    return make_cluster(monkeypatch, old), joined_of(*items), urls


def test_a_url_below_the_floor_is_skipped_counted_and_the_rest_of_the_pack_lands(
        tmp_path, monkeypatch, settings):
    c, joined, urls = pdf_world(monkeypatch)
    got = plan(manifest(docs=len(urls), family=PDF_FAMILY), joined, c, tmp_path, holds_skips=True)
    s = got["summary"]
    assert s["word_coverage_skipped_urls"] == 1
    sk = s["word_coverage"]["skipped"]
    assert [x["url"] for x in sk] == [SWISSCOM]
    assert sk[0]["ratio"] == 0.2 and (sk[0]["new_words"], sk[0]["old_words"]) == (50, 250)
    assert s["word_coverage"]["skip_share"] == 0.1 and s["word_coverage_skip_ceiling"] == 0.2
    # the skipped url: its 10 old rows stay (not in the plan), its 2 pack rows are named to be held
    assert s["word_coverage_skipped_old_rows"] == 10 and s["word_coverage_skipped_pack_rows"] == 2
    assert len(got["skipped_pack_ids"]) == 2
    swiss_old = {i for i, r in c.colls["http://twin"].items() if r["payload"]["url"] == SWISSCOM}
    assert len(swiss_old) == 10 and not swiss_old & set(got["ids"])
    assert SWISSCOM not in got["urls"].values()
    # the 9 clean urls land: 36 old rows superseded
    assert len(got["ids"]) == 36 and s["superseded_rows"] == 36 and s["superseded_urls"] == 9
    assert s["export"]["rows"] == 36


def test_a_clean_pack_skips_nothing_and_says_so(tmp_path, monkeypatch, settings):
    c, joined, urls = pdf_world(monkeypatch, small=(), clean=5)
    got = plan(manifest(docs=len(urls), family=PDF_FAMILY), joined, c, tmp_path, holds_skips=True)
    s = got["summary"]
    assert s["word_coverage_skipped_urls"] == 0 and s["word_coverage"]["skipped"] == []
    assert s["word_coverage"]["violations"] == 0 and got["skipped_pack_ids"] == []
    assert len(got["ids"]) == 20 and s["word_coverage_skip_ceiling"] == UGS.SKIP_SHARE_MAX


def test_a_skip_is_refused_by_a_write_path_that_does_not_hold_the_rows_out(
        tmp_path, monkeypatch, settings):
    """The default (`holds_skips` False, what the drain passes today): writing the skipped url's
    pack rows beside its old rows would double the url, so the pack is refused, before any export."""
    c, joined, urls = pdf_world(monkeypatch)
    with pytest.raises(Q.QueueError, match="Rebuild the pack without these url") as exc:
        plan(manifest(docs=len(urls), family=PDF_FAMILY), joined, c, tmp_path)
    assert SWISSCOM in str(exc.value) and "ratio 0.2" in str(exc.value)
    assert not list(settings.iterdir()) and not (tmp_path / "artifacts").exists()


def test_more_than_a_fifth_skipped_refuses_the_whole_pack_and_names_the_urls(
        tmp_path, monkeypatch, settings):
    small = (SWISSCOM, SKIP_PAGE % "small1", SKIP_PAGE % "small2")          # 3 of 10 = 30 %
    c, joined, urls = pdf_world(monkeypatch, small=small, clean=7)
    with pytest.raises(Q.QueueError, match="skip ceiling") as exc:
        plan(manifest(docs=len(urls), family=PDF_FAMILY), joined, c, tmp_path, holds_skips=True)
    msg = str(exc.value)
    assert "3 of 10" in msg and "30.0%" in msg and "would lose text" in msg
    assert all(u in msg for u in small)
    assert not list(settings.iterdir())


def test_exactly_a_fifth_skipped_still_lands_the_rest(tmp_path, monkeypatch, settings):
    small = (SWISSCOM, SKIP_PAGE % "small1")                                 # 2 of 10 = 20 %
    c, joined, urls = pdf_world(monkeypatch, small=small, clean=8)
    got = plan(manifest(docs=len(urls), family=PDF_FAMILY), joined, c, tmp_path, holds_skips=True)
    assert got["summary"]["word_coverage_skipped_urls"] == 2 and len(got["ids"]) == 32


def test_the_skip_ceiling_is_the_content_pack_ceiling():
    assert UGS.SKIP_SHARE_MAX == Q.COVERAGE_VIOLATION_SHARE_MAX == 0.20


def test_a_resumed_run_that_would_write_a_skipped_urls_rows_is_refused(
        tmp_path, monkeypatch, settings):
    c, joined, urls = pdf_world(monkeypatch)
    m = manifest(docs=len(urls), family=PDF_FAMILY)
    got = plan(m, joined, c, tmp_path, holds_skips=True)
    UGS.write_state(tmp_path / "artifacts", got, stage="deleting")
    again = plan(m, joined, c, tmp_path, holds_skips=True)
    assert again["skipped_pack_ids"] == got["skipped_pack_ids"] and again["ids"] == got["ids"]
    with pytest.raises(Q.QueueError, match="resume: 2 pack row"):
        plan(m, joined, c, tmp_path)


@needs_pack
def test_draining_the_same_landed_pack_twice_is_a_no_op_the_second_time(
        tmp_path, monkeypatch, settings):
    """Task 888: after LANDED + DONE a second drain of the same queue writes and deletes nothing."""
    urls = real_pack_urls(monkeypatch)
    give_the_pack_a_family(monkeypatch)
    old = old_row_at(sorted(urls.values())[0])
    c = make_cluster(monkeypatch, [old])
    queue, contract = drain_fixture(tmp_path, monkeypatch, settings, supersede="url-group")
    wire_real_drain(monkeypatch, c, prod_upsert_fails=[False])
    first = run_drain(queue, contract, tmp_path)
    assert first[0]["status"] == "LANDED" and (queue / "loda-refresh" / "DONE").exists()
    after = {u: dict(rows) for u, rows in c.colls.items()}
    calls = len(c.calls)
    second = run_drain(queue, contract, tmp_path)
    assert not any(r.get("status") == "LANDED" for r in second), second
    assert {u: dict(rows) for u, rows in c.colls.items()} == after
    assert len(c.calls) == calls
