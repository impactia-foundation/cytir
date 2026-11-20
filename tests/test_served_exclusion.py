"""Ruling 23.5 / ADR-116 (task 702): the compose-time already-served exclusion, on REAL rows
(the two task-656 ERIC chunk records `test_aw_index_compose` uses), round 2 (review of #572)."""
from __future__ import annotations

import json
import pathlib

import pytest

import cytir.index.compose as compose
from cytir.index import exclude_file as XF
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as PSP
from cytir.index import served_exclusion as SE
import cytir.ops.growth_pending_compose as GPC
from tests.helpers.repo_files import repo_file_or_skip
from tests.helpers.sealed_evidence import require_sealed_probe
from test_aw_index_compose import (_real_rows, _write_chunk_shard, _write_vector_manifest,
                                   _trimmed_ids_file)

require_sealed_probe()  # real ERIC rows need the sealed consent probe, which the public cut lacks

TARGETS = {"twin": ("http://twin", "aw_e8"), "prod": ("http://prod", "aw_prod")}
FAMILY_KW = dict(family_license="ADR-111-opt-out-minimum-record",
                 family_payload={"source_type": "docs", "trust_tier": 0.75},
                 title_from=["section_heading"])


@pytest.fixture
def pack(tmp_path):
    rows = _real_rows()
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    _write_chunk_shard(chunks, rows)
    _write_vector_manifest(vectors, n=len(rows), dim=1024)
    return {"rows": rows, "ids": [r["point_id"] for r in rows], "chunks": chunks,
            "vectors": vectors, "queue": tmp_path / "queue", "tmp": tmp_path}


def _own_payload(pack, ids):
    """What the drain would write for `ids`, restricted to the compared fields: the stored row of
    an id that is the SAME row on the twin."""
    got = compose._pack_payloads(str(pack["chunks"]), list(ids), {
        "identity_check": "pack-declared", "identity_reason": "t", **FAMILY_KW})
    return {i: {f: p.get(f) for f in SE.PAYLOAD_COMPARE_FIELDS} for i, p in got.items()}


def _holds(monkeypatch, pack, twin=(), prod=(), twin_payload=None):
    """Stand in for the two targets. `calls` records every request, per target."""
    calls = {"twin": [], "prod": []}
    stored = _own_payload(pack, twin)
    stored.update(twin_payload or {})

    def retrieve(url, coll, ids, fields, batch=500, workers=1):
        calls["twin"].append({"n": len(ids), "workers": workers, "fields": tuple(fields)})
        return {i: dict(stored[i]) for i in ids if i in stored}

    def present(url, coll, ids, batch=500, workers=1):
        calls["prod"].append({"n": len(ids), "workers": workers, "url": url})
        return [i for i in ids if i in set(prod)]

    monkeypatch.setattr(compose.PSP, "retrieve_payloads", retrieve)
    monkeypatch.setattr(compose.PSP, "already_present", present)
    return calls


def _compose(pack, **kw):
    return compose.compose_pack(
        pack_id="p1", family="eric_official_pdf", chunks_dir=str(pack["chunks"]),
        vectors_dir=str(pack["vectors"]), collection="aw_e8", mrl=256,
        queue_dir=str(pack["queue"]), served_targets=TARGETS, **FAMILY_KW, **kw)


def test_ids_on_both_targets_are_excluded_with_adr116_reason_per_id(pack, monkeypatch):
    served = pack["ids"][:1]
    _holds(monkeypatch, pack, twin=served, prod=served)
    res = _compose(pack)
    m = res["manifest"]
    assert m["already_served_excluded"] == 1
    f = pathlib.Path(res["pack_dir"]) / m["already_served_excluded_file"]
    assert m["already_served_excluded_sha256"] == XF.sha256_file(f)
    # the drain reads the SAME keys a --exclude-ids-file reseal writes
    assert m["exclude_ids_sha256"] == m["already_served_excluded_sha256"]
    got = XF.load_exclude_ids(m["exclude_ids_file"], sha256=m["exclude_ids_sha256"],
                              reason=m["exclude_ids_reason"])
    assert got == {served[0]: "already_on_twin_and_prod"}      # ADR-116 section 2's reason code
    assert m["already_served_checked_against"]["twin"]["ids_present"] == 1


def test_manifest_rows_are_the_rows_that_will_be_written(pack, monkeypatch):
    """L7: the queue's row cap counts what the drain writes, not the excluded ids."""
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    res = _compose(pack)
    m = res["manifest"]
    assert (m["rows"], m["rows_before_exclusion"], m["already_served_excluded"]) == (1, 2, 1)
    assert (m["rows_staged"], m["rows_to_write_max"]) == (2, 1)      # the same two numbers, named
    assert m["excluded_ids_rechecked_at_drain"] is False         # L6: stated, not implied
    assert GPC.queue_depth_rows(str(pack["queue"])) == 1


def test_a_twin_only_id_is_a_resume_not_an_overlap(pack, monkeypatch):
    """Half-applied by this drain: twin written, prod not. Nothing may be excluded."""
    calls = _holds(monkeypatch, pack, twin=pack["ids"], prod=())
    m = _compose(pack)["manifest"]
    assert m["already_served_excluded"] == 0 and m["rows"] == 2
    assert "exclude_ids_file" not in m
    assert m["already_served_checked_against"]["twin_only_not_excluded"] == 2
    assert calls["prod"][0]["n"] == 2          # prod was asked about the twin's hits only


def test_a_prod_only_id_is_not_excluded_and_prod_is_not_even_asked(pack, monkeypatch):
    """Prod holds an id the twin does not: a twin/prod divergence. Compose does not touch it, so
    the drain's own refusal for it stays (ADR-116 'Not decided here')."""
    calls = _holds(monkeypatch, pack, twin=(), prod=pack["ids"])
    m = _compose(pack)["manifest"]
    assert m["already_served_excluded"] == 0 and m["rows"] == 2
    assert "exclude_ids_file" not in m and "already_served_excluded_file" not in m
    assert calls["prod"] == []                 # net-new on the twin: 0 prod requests


def test_prod_is_asked_second_only_about_twin_hits_at_the_drains_width(pack, monkeypatch):
    calls = _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    _compose(pack)
    assert [c["n"] for c in calls["twin"]] == [2]
    assert [c["n"] for c in calls["prod"]] == [1]
    assert calls["prod"][0]["workers"] == PQ.LOOKUP_WORKERS["prod"] == 2   # incident 2026-10-07: prod 2-wide
    assert calls["twin"][0]["workers"] == PQ.LOOKUP_WORKERS["twin"]
    assert calls["twin"][0]["fields"] == SE.PAYLOAD_COMPARE_FIELDS


@pytest.mark.parametrize("which", ["twin", "prod"])
def test_an_unanswered_check_refuses_and_writes_nothing(pack, monkeypatch, which):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])

    def boom(*a, **kw):
        raise OSError("tunnel down")
    monkeypatch.setattr(compose.PSP, "retrieve_payloads" if which == "twin" else "already_present",
                        boom)
    with pytest.raises(SE.ServedExclusionError, match="tunnel down"):
        _compose(pack)
    assert not (pack["queue"] / "p1").exists()
    assert not (pack["queue"] / SE.SETTLED_DIR).exists()


# ------------------------------------------------------------ the resume guard (H3)
def test_a_sealed_packs_twin_written_artifact_stops_the_check(pack, monkeypatch):
    """The artifact a SEALED pack's drain leaves, named by the drain's own helper."""
    calls = _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])   # would be ALL served
    art = pack["queue"] / "_artifacts"
    art.mkdir(parents=True)
    PQ.twin_written_ids_path(art, "p1").write_text(json.dumps(pack["ids"]))
    assert PQ.twin_written_ids_path(art, "p1").name == "p1-twin-written-ids.json"
    res = _compose(pack)
    assert res["all_served"] is False
    assert res["manifest"]["already_served_excluded"] == 0 and res["manifest"]["rows"] == 2
    assert "resume guard" in res["manifest"]["already_served_check"]
    assert calls == {"twin": [], "prod": []}      # nothing was even asked


def test_a_content_packs_twin_state_also_stops_the_check(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    art = pack["queue"] / "_artifacts"
    art.mkdir(parents=True)
    PQ.write_content_pack_twin_state(art, "p1", stage="twin-write-complete", manifest_sha="x")
    assert _compose(pack)["manifest"]["already_served_excluded"] == 0


def test_without_an_artifact_the_same_state_is_excluded(pack, monkeypatch):
    """The control for the two tests above: the artifact is what changes the answer."""
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    assert _compose(pack)["manifest"]["already_served_excluded"] == 1


def test_a_recompose_keeps_the_first_declaration_byte_for_byte(pack, monkeypatch):
    # The manifest also carries the consent stamp's `as_of` (second resolution, `stamp.utc_now()`).
    # Two composes more than a second apart wrote different bytes, so this test passed or failed by
    # the clock (task 826: it failed on a slow shard and standalone on origin/main). The clock is
    # pinned; the test is about the already-served declaration.
    import cytir.consent.stamp as ST
    monkeypatch.setattr(ST, "utc_now", lambda: "2026-11-09T00:00:00Z")
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    first = _compose(pack)
    before = (pathlib.Path(first["pack_dir"]) / "manifest.json").read_bytes()
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])   # the drain has since landed more
    again = _compose(pack)
    assert (pathlib.Path(again["pack_dir"]) / "manifest.json").read_bytes() == before


# ------------------------------------------------------- composes to nothing (H2)
def _drain(queue):
    return PQ.drain(queue, twin_url="http://twin", prod_url="http://prod", task="t", apply=False,
                    batch=500, pause_s=0, artifact_root=queue / "_artifacts",
                    refresh_index=False)


def test_a_pack_with_every_id_served_never_enters_the_queue_as_a_pack(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    res = _compose(pack)
    assert res["all_served"] is True and res["ready_sha"] is None and res["manifest"] is None
    assert not (pack["queue"] / "p1").exists()             # no pack directory, no manifest, no READY
    rec = json.loads(SE.settled_path(pack["queue"], "p1").read_text())
    assert rec["state"] == "composed-to-nothing" and rec["reason"] == "already_on_twin_and_prod"
    assert rec["rows"] == 2 and rec["already_served_excluded"] == 2
    d = pack["queue"] / SE.SETTLED_DIR
    assert rec["excluded_ids_sha256"] == XF.sha256_file(d / rec["excluded_ids_file"])
    assert {json.loads(l)["point_id"] for l in (d / rec["excluded_ids_file"]).read_text().splitlines()
            } == set(pack["ids"])
    assert GPC.already_enqueued(str(pack["queue"]), "p1")      # settled: the timer skips it


def test_the_drain_has_nothing_to_do_after_an_all_served_compose(pack, monkeypatch, capsys):
    """The review's failure: a dir with ALL-SERVED.json and no manifest was REFUSED at intake every
    hour for ever, with a new receipt each time. Drive the real drain, twice."""
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    _compose(pack)
    for _ in range(2):
        assert _drain(pack["queue"]) == []
    assert "empty" in capsys.readouterr().out
    assert sorted(p.name for p in pack["queue"].iterdir()) == [SE.SETTLED_DIR]
    assert not list(pack["queue"].rglob("receipt*"))


def test_the_timer_does_not_loop_on_an_all_served_stage(pack, monkeypatch, tmp_path):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    _compose(pack)
    assert GPC.already_enqueued(str(pack["queue"]), "p1")
    assert not GPC.already_enqueued(str(pack["queue"]), "p2")


# --------------------------------------------- explicit file keeps each id's own reason (M5)
def test_an_explicit_exclusion_file_is_merged_and_keeps_its_own_reason(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    explicit = _trimmed_ids_file(pack["tmp"], ["a-floor-trimmed-id"])
    res = _compose(pack, exclude_ids_file=str(explicit), exclude_ids_reason="floor")
    m = res["manifest"]
    got = XF.load_exclude_ids(m["exclude_ids_file"], sha256=m["exclude_ids_sha256"],
                              reason=m["exclude_ids_reason"])
    assert got == {pack["ids"][0]: "already_on_twin_and_prod", "a-floor-trimmed-id": "floor"}
    assert m["already_served_excluded"] == 1 and m["exclude_ids_reason"]
    assert m["rows"] == 1            # the floor id is not in this pack: only the served one is dropped


def test_a_callers_own_exclusion_also_leaves_the_row_count(pack, monkeypatch):
    """L7 for the explicit file alone (nothing served): `rows` is what will be written."""
    _holds(monkeypatch, pack)
    explicit = _trimmed_ids_file(pack["tmp"], pack["ids"][:1])
    m = _compose(pack, exclude_ids_file=str(explicit), exclude_ids_reason="floor")["manifest"]
    assert (m["rows"], m["rows_before_exclusion"], m["already_served_excluded"]) == (1, 2, 0)
    assert (m["rows_staged"], m["rows_to_write_max"]) == (2, 1)
    assert m["exclude_ids_reason"] == "floor"            # untouched: no served ids to merge


def test_a_row_without_its_own_reason_gets_the_files_reason(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps({"point_id": "a"}) + "\n" + json.dumps({"point_id": "b", "reason": "r2"}) + "\n")
    got = XF.load_exclude_ids(str(f), sha256=XF.sha256_file(f), reason="r1")
    assert got == {"a": "r1", "b": "r2"}


# ------------------------------------- "already served" is not "same content" (H1, measured)
def test_payload_that_differs_is_measured_recorded_and_still_excluded(pack, monkeypatch):
    changed = pack["ids"][0]
    stored = {changed: {"title": "Old title - Github", "source_type": "reference",
                        "trust_tier": 0.6, "tier": "B", "lang": None,
                        "license": "ADR-111-opt-out-minimum-record", "published": None}}
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"], twin_payload=stored)
    res = _compose(pack)
    assert res["all_served"] is True                       # what is excluded does not change
    rec = json.loads(SE.settled_path(pack["queue"], "p1").read_text())
    assert rec["already_served_payload_differs"] == 1
    assert rec["already_served_payload_differs_fields"] == {
        "source_type": 1, "title": 1, "trust_tier": 1}
    d = pack["queue"] / SE.SETTLED_DIR
    assert [json.loads(l) for l in (d / rec["payload_differs_file"]).read_text().splitlines()] == [
        {"fields": ["title", "source_type", "trust_tier"], "point_id": changed}]
    assert rec["payload_differs_sha256"] == XF.sha256_file(d / rec["payload_differs_file"])


def test_the_sidecar_of_a_pack_that_still_composes_is_pinned_in_the_manifest(pack, monkeypatch):
    changed = pack["ids"][0]
    stored = {changed: {"title": "Old", "source_type": "reference", "trust_tier": 0.6, "tier": "B",
                        "lang": None, "license": "ADR-111-opt-out-minimum-record", "published": None}}
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1], twin_payload=stored)
    res = _compose(pack)
    m = res["manifest"]
    side = pathlib.Path(res["pack_dir"]) / m["already_served_payload_differs_file"]
    assert m["already_served_payload_differs"] == 1
    assert m["already_served_payload_differs_sha256"] == XF.sha256_file(side)
    assert m["already_served_payload_compared_fields"] == list(SE.PAYLOAD_COMPARE_FIELDS)
    assert m["already_served_excluded"] == 1 and m["exclude_ids_file"]       # still excluded


def test_the_same_payload_records_zero_differences(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])
    m = _compose(pack)["manifest"]
    assert m["already_served_payload_differs"] == 0
    assert "already_served_payload_differs_file" not in m


def test_content_digest_is_not_an_equality_key():
    assert "content_digest" not in SE.PAYLOAD_COMPARE_FIELDS
    assert SE.PAYLOAD_COMPARE_FIELDS == ("title", "source_type", "trust_tier", "tier", "lang",
                                         "license", "published")
    assert SE.compare_payloads({"title": "t", "content_digest": "a"},
                               {"title": "t", "content_digest": "b"}) == []
    assert SE.compare_payloads({"trust_tier": 0.75}, {"trust_tier": 0.6}) == ["trust_tier"]
    assert SE.compare_payloads({"lang": None}, {"lang": ""}) == []      # absent reads as absent


def test_retrieve_payloads_is_an_id_retrieve_with_a_field_projection(monkeypatch):
    seen = []

    def fake_api(url, path, body=None, method=None):
        seen.append((url, path, body))
        return [{"id": i, "payload": {"title": "T-" + i}} for i in body["ids"] if i != "gone"]

    monkeypatch.setattr(PSP, "_api", fake_api)
    got = PSP.retrieve_payloads("http://t", "c", ["a", "gone", "b"], ("title",), batch=2, workers=2)
    assert got == {"a": {"title": "T-a"}, "b": {"title": "T-b"}}
    assert [(p, b["with_payload"], b["with_vector"]) for _u, p, b in seen] == [
        ("/collections/c/points", ["title"], False)] * 2


# ------------------------------------------------------------------ unchanged behaviour
def test_without_targets_nothing_is_asked_and_no_fields_appear(pack):
    res = compose.compose_pack(
        pack_id="p1", family="eric_official_pdf", chunks_dir=str(pack["chunks"]),
        vectors_dir=str(pack["vectors"]), collection="aw_e8", mrl=256,
        queue_dir=str(pack["queue"]))
    assert "already_served_excluded" not in res["manifest"] and res["all_served"] is False
    assert res["manifest"]["rows"] == 2


def test_prod_url_without_a_twin_url_refuses():
    assert compose.served_targets_from_args(None, "c", None, "aw_prod") is None
    with pytest.raises(compose.ComposeError, match="--qdrant-url"):
        compose.served_targets_from_args(None, "c", "http://prod", "aw_prod")
    assert compose.served_targets_from_args("http://t", "c", "http://p", "aw_prod") == {
        "twin": ("http://t", "c"), "prod": ("http://p", "aw_prod")}


@pytest.mark.parametrize("path", [
    "cytir/news/compose.py", "cytir/sources/refresh_direct/direct_compose.py",
    "cytir/ops/spam_tombstones.py", "cytir/index/reseal.py",
    "cytir/sources/run_family.py"])
def test_the_producers_that_re_emit_an_id_on_purpose_never_use_the_exclusion(path):
    """Review 'Composers': news, the refresh lane, L1 backfill, spam tombstones, reseal and the
    family runner re-emit existing ids by design. Excluding them there would drop the update."""
    text = repo_file_or_skip(path).read_text()
    assert "served_targets" not in text and "served_exclusion" not in text


def test_tranche_6_real_rows_203_excluded_203_hold_a_different_payload(tmp_path):
    """The 203 ids of the hand reseal (task 696), read from the twin by id (all `aw_b1` rows),
    against the payload the tranche-6 pack would write (the drain's projection of its records).
    Every one differs in title, source_type and trust_tier; lang, license, tier, published agree."""
    real = json.loads((pathlib.Path(__file__).parent / "testdata"
                       / "task702_tranche6_served_203.json").read_text())
    ids = sorted(real)
    asked = {"prod": []}

    def prod(i):
        asked["prod"].append(len(i))
        return list(i)

    out = SE.declare_already_served(
        pack_dir=tmp_path / "t6", pack_id="cc-authority-frontier-w7-tranche6",
        ids=ids + ["net-new-%d" % n for n in range(50)], targets=TARGETS,
        twin_rows=lambda i: {x: real[x]["stored"] for x in i if x in real},
        prod_present=prod, pack_payloads=lambda i: {x: real[x]["pack"] for x in i})
    f = out["fields"]
    assert (f["already_served_excluded"], f["already_served_payload_differs"]) == (203, 203)
    assert f["already_served_payload_differs_fields"] == {
        "source_type": 203, "title": 203, "trust_tier": 203}
    assert asked["prod"] == [203]                       # ONE prod question, of the twin's 203 hits
    assert f["already_served_checked_against"]["twin"]["ids_asked"] == 253
    side = [json.loads(l) for l in (tmp_path / "t6" / SE.DIFFERS_FILE).read_text().splitlines()]
    assert len(side) == 203 and {tuple(s["fields"]) for s in side} == {
        ("title", "source_type", "trust_tier")}
    assert f["already_served_payload_differs_sha256"] == XF.sha256_file(tmp_path / "t6" / SE.DIFFERS_FILE)


def test_a_row_that_cannot_be_projected_is_unchecked_never_equal(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"][:1], prod=pack["ids"][:1])

    def refuse(*a, **kw):
        raise PSP.PromotionError("unresolved license")
    monkeypatch.setattr(compose.PSP, "project_rows", refuse)
    m = _compose(pack)["manifest"]
    assert m["already_served_excluded"] == 1                  # still excluded (ADR-116 as merged)
    assert m["already_served_payload_unchecked"] == 1 and m["already_served_payload_differs"] == 0


# ---------------------------------------------------------------- round 3 (review of #572)
def test_the_settled_record_carries_what_an_audit_needs(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    explicit = _trimmed_ids_file(pack["tmp"], ["a-floor-trimmed-id"])
    _compose(pack, exclude_ids_file=str(explicit), exclude_ids_reason="floor")
    rec = json.loads(SE.settled_path(pack["queue"], "p1").read_text())
    # `pack_chunks` is what a manifest of this pack would have named: the consent-stamped copy
    assert (rec["family"], rec["vectors"]) == ("eric_official_pdf", str(pack["vectors"]))
    assert rec["pack_chunks"].startswith(str(pack["chunks"]))
    assert rec["composed_at"].endswith("Z") and rec["composed_at"][:2] == "20"
    assert rec["explicit_exclude_file"] == str(explicit)
    assert rec["explicit_exclude_sha256"] == XF.sha256_file(explicit)


def test_the_settled_record_has_no_explicit_keys_without_a_caller_file(pack, monkeypatch):
    _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    _compose(pack)
    rec = json.loads(SE.settled_path(pack["queue"], "p1").read_text())
    assert "explicit_exclude_file" not in rec and rec["family"] == "eric_official_pdf"


def test_a_resume_artifact_still_takes_the_callers_own_file_off_rows(pack, monkeypatch):
    """The early return of the resume guard: nothing served is excluded, the caller's file still is."""
    calls = _holds(monkeypatch, pack, twin=pack["ids"], prod=pack["ids"])
    art = pack["queue"] / "_artifacts"
    art.mkdir(parents=True)
    PQ.twin_written_ids_path(art, "p1").write_text(json.dumps(pack["ids"]))
    explicit = _trimmed_ids_file(pack["tmp"], pack["ids"][:1])
    m = _compose(pack, exclude_ids_file=str(explicit), exclude_ids_reason="floor")["manifest"]
    assert calls == {"twin": [], "prod": []}
    assert (m["rows"], m["rows_before_exclusion"], m["already_served_excluded"]) == (1, 2, 0)


def test_rows_without_the_served_check_stays_the_stage_count(pack):
    """Fix 5 is NOT done (follow-up): a plain `--exclude-ids-file` compose keeps `rows` whole, as
    task 680 pinned and as `reseal` writes, so the family row cap's arithmetic for queued packs
    does not move. This test records the two meanings so the follow-up changes both on purpose."""
    explicit = _trimmed_ids_file(pack["tmp"], pack["ids"][:1])
    m = compose.compose_pack(
        pack_id="p1", family="eric_official_pdf", chunks_dir=str(pack["chunks"]),
        vectors_dir=str(pack["vectors"]), collection="aw_e8", mrl=256, queue_dir=str(pack["queue"]),
        exclude_ids_file=str(explicit), exclude_ids_reason="floor", **FAMILY_KW)["manifest"]
    assert m["rows"] == 2 and "rows_before_exclusion" not in m
    # the alias keeps its old meaning (the stage count); the new names say both numbers
    assert (m["rows_staged"], m["rows_to_write_max"]) == (2, 1)
