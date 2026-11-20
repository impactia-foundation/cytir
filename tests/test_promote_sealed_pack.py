#!/usr/bin/env python3
"""Real-record guards for `promote_sealed_pack` (task 629, precondition P3).

The fixture under `tests/fixtures/promote/task629_sealed_pack/` is THREE ACTUAL ROWS
of the `technical-manufacturer-admitted-products-v2` pack — the W1 pathfinder
family — with their actual sealed 1024-d fp16 vectors, in the `row_ids.jsonl`
order the manifest's `ordering` field asserts. Not a mock: the whole point of the
reader is that it handles the artefact this project really produces, and a
hand-built fixture would agree with a hand-built reader while both were wrong.

Every test here is a refusal this module is supposed to make, because the failure
mode it guards against is the one that is silent. `points_count` cannot see an
id that addresses nothing (task 546: context7 shipped 53,822 of 84,961 staged
rows addressing nothing), a partial join, or a vector stored beside the wrong
row's payload.
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

_ADDED = [n for n in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT")
          if n not in os.environ]
os.environ.setdefault("AW_OPERATOR_URL", "https://websear.ch")
os.environ.setdefault("AW_OPERATOR_CONTACT", "bender@impactia.org")
try:
    import cytir.core.license as LIC  # noqa: E402
    from cytir.index import promote_sealed_pack as P  # noqa: E402
finally:
    for _n in _ADDED:
        os.environ.pop(_n, None)

FIXTURE = HERE / "fixtures" / "promote" / "task629_sealed_pack"
CHUNKS = str(FIXTURE / "chunks")
VECTORS = str(FIXTURE / "vectors")

#: A second REAL fixture: three rows of news-current-articles-v5b, chosen for
#: their `publication_time_status` -- two `source_declared`, and one genuinely
#: `not_declared_by_publisher` (https://www.rts.ch/archives, a real row). The
#: POLICY 4.7 rule under test is "exclude, never backfill", and a fixture with
#: only good rows could not exercise it.
NEWS = HERE / "fixtures" / "promote" / "task629_news_pack"
NEWS_CHUNKS = str(NEWS / "chunks")
NEWS_VECTORS = str(NEWS / "vectors")

#: What the news family's contract declares. `source_type` and `trust_tier` are
#: POLICY facts, which is why the module refuses to guess them.
NEWS_PAYLOAD = {"source_type": "news", "trust_tier": 0.70}
NEWS_TITLE_FROM = ("source_metadata.title", "source_metadata.headline",
                   "section_heading")
NEWS_DATE = dict(published_from="source_metadata.publication_time",
                 published_status_from="source_metadata.publication_time_status",
                 published_status_requires="source_declared",
                 require_published=True)


def news_records() -> list[dict]:
    return list(P.read_records(NEWS_CHUNKS))


def promote_news(rows=None, **kw):
    opts = dict(identity_check="remint-b1", family_payload=NEWS_PAYLOAD,
                title_from=NEWS_TITLE_FROM, **NEWS_DATE)
    opts.update(kw)
    return P.project_rows(iter(rows if rows is not None else news_records()),
                          consent=None, attribution=None, license_version=None,
                          **opts)

#: A licence the pack itself does not carry. `resolve_license` trusts a truthy
#: value already on the record, so this exercises the module's licence PATH
#: without asserting anything about what sunfounder.com's licence should be —
#: that is P8's question and belongs in P8's artefact, not in a unit test.
TEST_LICENCE = "test-only-not-a-licence-finding"


def records() -> list[dict]:
    return list(P.read_records(CHUNKS))


def licensed_records() -> list[dict]:
    return [{**r, "license": TEST_LICENCE} for r in records()]


#: Pool A's ids are not reproducible from their rows (252 combinations tried,
#: 2026-09-22 — see the module docstring), so every test that needs rows PAST the
#: identity gate must name that and say why, exactly as a real run has to.
PACK_DECLARED = dict(
    identity_check="pack-declared",
    identity_reason="test: Pool A ids are not reproducible; module docstring")


def projected(rows, **kw):
    """(joined, skipped, modes, vector_key) -- pack-declared unless overridden."""
    opts = {**PACK_DECLARED, **kw}
    return P.project_rows(iter(rows), consent=None, attribution=None,
                          license_version=None, **opts)


def reminted(rows):
    return P.project_rows(iter(rows), consent=None, attribution=None,
                          license_version=None, identity_check="remint-b1")


# ------------------------------------------------------------------ the reader
def test_the_fixture_is_the_real_artefact_shape():
    man = json.loads((FIXTURE / "vectors"
                      / "shard-0000.vectors.manifest.json").read_text())
    assert man["dim"] == P.SEALED_DIM == 1024
    assert man["dtype"] == "float16"
    assert man["n"] == 3
    raw = (FIXTURE / "vectors" / "shard-0000.vectors.f16").stat().st_size
    assert raw == 3 * 1024 * 2, "the sealed packs are dense-only, rows*1024*2"


def test_read_vectors_truncates_and_renormalises():
    matrix, ids = P.read_vectors(VECTORS, 256)
    assert matrix.shape == (3, 256)
    assert len(ids) == 3
    norms = np.linalg.norm(matrix, axis=1)
    # Cosine: the renormalise is not optional (serving_config.mrl_trunc).
    assert np.allclose(norms, 1.0, atol=1e-5), norms


def test_read_vectors_is_a_noop_width_check_not_a_reshape(tmp_path):
    """A shard whose manifest disagrees with its bytes must refuse, not reshape.

    Reshaping to the manifest's `n` when the bytes say otherwise is how a whole
    shard silently shifts by one row, which mis-pairs every vector after it.
    """
    dst = tmp_path / "vectors"
    shutil.copytree(VECTORS, dst)
    man = json.loads((dst / "shard-0000.vectors.manifest.json").read_text())
    man["n"] = 4
    (dst / "shard-0000.vectors.manifest.json").write_text(json.dumps(man))
    with pytest.raises(P.PromotionError, match="is not n\\*1024"):
        P.read_vectors(str(dst), 256)


def test_row_ids_count_must_match_the_vector_count(tmp_path):
    dst = tmp_path / "vectors"
    shutil.copytree(VECTORS, dst)
    rows = (dst / "shard-0000.row_ids.jsonl").read_text().splitlines()
    (dst / "shard-0000.row_ids.jsonl").write_text("\n".join(rows[:2]) + "\n")
    with pytest.raises(P.PromotionError, match="ids for 3 vectors"):
        P.read_vectors(str(dst), 256)


def test_an_empty_chunks_dir_names_the_openalex_partition_trap(tmp_path):
    """The message has to carry the finding: two OpenAlex tranches keep their
    chunk shards under a worker partition (and a resume-v2 checkpoint), and a
    bare "no shards" error sent someone looking in the wrong place once."""
    with pytest.raises(P.PromotionError, match="resume-v2"):
        list(P.read_records(str(tmp_path)))


# --------------------------------------------------------------- identity gate
def test_reproduce_mode_refuses_pool_a_and_that_is_the_measured_state():
    """The default mode refuses these real rows, and that is not a bug.

    252 combinations of minting helper x text field x url form x anchor x
    source_type were tried against a real row and none reproduced the pack's id.
    Asserting the REFUSAL is what stops the default quietly becoming permissive:
    if someone later makes Pool A's ids reproducible, this test fails and they
    will see that the flag is no longer needed.
    """
    with pytest.raises(P.PromotionError, match="identity drift"):
        P.project_rows(iter(licensed_records()), consent=None, attribution=None,
                       license_version=None)


def test_pack_declared_mode_admits_them_and_counts_them_as_such():
    joined, skipped, modes, _vk, _u = projected(licensed_records())
    assert skipped == 0
    assert len(joined) == 3
    # The count is the point: a verdict must not read as the strong check passing.
    assert modes == {"full": 0, "point_id": 0, "pack_declared": 3, "reminted": 0}


def test_pack_declared_mode_requires_a_reason():
    """An unexplained acceptance is how a trust decision becomes a default."""
    with pytest.raises(P.PromotionError, match="requires a reason"):
        P.project_rows(iter(licensed_records()), consent=None, attribution=None,
                       license_version=None, identity_check="pack-declared")


def test_an_unknown_identity_mode_is_refused():
    with pytest.raises(P.PromotionError, match="identity-check must be one of"):
        P.project_rows(iter(licensed_records()), consent=None, attribution=None,
                       license_version=None, identity_check="trust-me",
                       identity_reason="no")


def test_the_admitted_id_is_the_packs_id_because_that_addresses_the_vector():
    joined, _s, _m, vector_key, _u = projected(licensed_records())
    assert set(joined) == {r["point_id"] for r in records()}
    # In pack-declared mode the written id IS the shard address.
    assert all(k == v for k, v in vector_key.items())


# ------------------------------------------------- remint-b1 (ruled for W1)
def test_remint_writes_the_id_a_rebuild_will_mint_not_the_packs():
    """The ruling: cold-start reproducibility. A rebuild mints under
    aw_b1_identity, so that is the id that gets stored."""
    joined, skipped, modes, vector_key, _u = reminted(licensed_records())
    assert skipped == 0
    assert modes == {"full": 0, "point_id": 0, "pack_declared": 0, "reminted": 3}
    pack_ids = {r["point_id"] for r in records()}
    assert set(joined).isdisjoint(pack_ids), (
        "re-minted ids must differ from the pack's -- if they matched, the pack "
        "would have been reproducible and this mode unnecessary")


def test_remint_keeps_the_pack_id_as_the_vector_address():
    """Cost (b) of the ruling, enforced: the sealed pack's lineage stays
    traceable from a served point, so its sha-verified vectors are provably the
    ones serving."""
    joined, _s, _m, vector_key, _u = reminted(licensed_records())
    assert set(vector_key) == set(joined)
    assert set(vector_key.values()) == {r["point_id"] for r in records()}
    # and the map is injective, or two written rows would claim one embedding
    assert len(set(vector_key.values())) == len(vector_key)


def test_remint_needs_no_reason_because_it_asserts_rather_than_trusts():
    """Unlike pack-declared, remint-b1 does not accept an unverifiable id -- it
    computes the id itself -- so it carries no reason requirement."""
    joined, _s, _m, _vk, _u = P.project_rows(
        iter(licensed_records()), consent=None, attribution=None,
        license_version=None, identity_check="remint-b1")
    assert len(joined) == 3


def test_remint_vectors_still_resolve_through_the_shard_index():
    matrix, ids = P.read_vectors(VECTORS, 256)
    joined, _s, _m, vector_key, _u = reminted(licensed_records())
    shard = {pid: i for i, pid in enumerate(ids)}
    index = {pid: shard[vector_key[pid]] for pid in joined}
    points = P.build_points(joined, matrix, index, 256)
    assert len(points) == 3
    for point in points:
        assert len(point["vector"][P.DENSE_VECTOR]) == 256
        assert point["vector"][P.SPARSE_VECTOR]["indices"]


def test_a_row_with_no_point_id_is_refused_in_every_mode():
    """It must be the FIRST row: in `reproduce` mode an intact Pool A row refuses
    on identity drift first, which would make this test pass for the wrong
    reason. remint-b1 must refuse too -- it needs the pack id to address the
    vector even though it writes a different one."""
    rows = licensed_records()
    rows[0].pop("point_id")
    for call in (lambda: projected(rows), lambda: reminted(rows),
                 lambda: P.project_rows(iter(rows), consent=None,
                                        attribution=None,
                                        license_version=None)):
        with pytest.raises(P.PromotionError, match="no `point_id`"):
            call()


def test_a_duplicate_point_id_within_the_pack_is_refused():
    rows = licensed_records()
    with pytest.raises(P.PromotionError, match="duplicate point id"):
        projected(rows + [rows[0]])
    with pytest.raises(P.PromotionError, match="duplicate point id"):
        reminted(rows + [rows[0]])


# ---------------------------------------------------------------- licence gate
def test_the_row_now_resolves_a_licence_AND_it_came_from_an_INSTRUMENT():
    """The predecessor of this test asserted a REFUSAL, and said why it would
    change: *"When P8 lands its instrument reading, this test keeps the bar: it
    asserts the REFUSAL, so a licence appearing by inference rather than by
    instrument would not make it pass."*

    P8 landed. `aw_b1_license` now carries sunfounder.com, so these rows resolve.
    The bar the old test was keeping is therefore re-stated here rather than
    deleted: it is not enough that a licence appears — it must come from the
    host's own instrument, with the document and the fetch date cited in the
    entry's evidence. An entry inferred from where the content is hosted would
    satisfy "a licence resolved" and fail this (memory
    `licence-basis-from-the-instrument`).

    Two things also had to be true for the row to resolve at all, and the second
    is what this file caught before B01 ran: the licence table must hold the host,
    and the RECORD must carry `domain`, because every host rule is keyed on it and
    `aw_b1_projection` never derives it.
    """
    joined, _skipped, _modes, _vk, _u = projected(records())
    assert joined, "the real rows no longer project at all"
    licences = {payload.get("license")
                for _dense, _sparse, payload, _rec in joined.values()}
    assert licences and all(licences), f"a row resolved no licence: {licences}"

    citation = LIC._HOST_LICENSE["sunfounder.com"].citation
    assert "sunfounder.com/pages/terms-of-service" in citation, (
        "the licence must cite the host's OWN instrument, not where the content "
        f"is hosted: {citation[:200]}")
    assert "fetched" in citation, "an instrument reading names when it was read"


def test_a_host_with_no_instrument_entry_is_still_refused():
    """The refusal the predecessor asserted, kept alive on a host that has none.

    Deleting the refusal along with its example would have left nothing asserting
    that an unlicensed row cannot be admitted -- ADR-039's amendment admits no new
    unlicensed content, and that bar is independent of which hosts happen to be
    in the table today.
    """
    from cytir.core import projection as aw_b1_projection
    rows = [dict(row, url="https://no-such-instrument.invalid/page")
            for row in records()]
    with pytest.raises(aw_b1_projection.ProjectionError,
                       match="unresolved license"):
        projected(rows)


# ------------------------------------------------------------------- the join
def test_the_fixture_join_is_total_in_both_directions():
    """A partial join would store some rows without their embedding and embed
    others without their payload. Both are silent, which is why `plan()` refuses
    on either direction; this asserts the real artefact starts total, so a future
    failure means the pack moved, not the reader."""
    _matrix, ids = P.read_vectors(VECTORS, 256)
    joined, _skipped, _modes, _vk, _u = projected(licensed_records())
    assert set(ids) == set(joined)
    assert len(ids) == len(set(ids)), "row_ids must not repeat an id"


def test_build_points_carries_a_sparse_posting_and_the_truncated_vector():
    matrix, ids = P.read_vectors(VECTORS, 256)
    joined, _s, _m, _vk, _u = projected(licensed_records())
    index = {pid: i for i, pid in enumerate(ids)}
    points = P.build_points(joined, matrix, index, 256)
    assert len(points) == 3
    for point in points:
        assert len(point["vector"][P.DENSE_VECTOR]) == 256
        # The packs are dense-only, so a posting here proves the sparse leg was
        # REGENERATED through the project's own tokeniser rather than carried.
        sparse = point["vector"].get(P.SPARSE_VECTOR)
        assert sparse and sparse["indices"], point["id"]
        assert len(sparse["indices"]) == len(sparse["values"])
        assert point["payload"].get("license") == TEST_LICENCE


def test_build_points_refuses_a_vector_of_the_wrong_width():
    matrix, ids = P.read_vectors(VECTORS, 128)
    joined, _s, _m, _vk, _u = projected(licensed_records())
    index = {pid: i for i, pid in enumerate(ids)}
    with pytest.raises(P.PromotionError, match="expected 256"):
        P.build_points(joined, matrix, index, 256)


def test_the_module_leaves_no_operator_identity_behind():
    for name in _ADDED:
        assert name not in os.environ


# ------------------------------- the family's payload fields (POLICY-declared)
def test_pool_a_rows_carry_only_url_and_lang_of_the_six_floored_fields():
    """The measured premise of the whole payload mechanism. If this ever changes,
    the derivation below is no longer needed and someone should know."""
    from cytir.index import payload_carry as PC
    floored = set(PC.RECORD_CONTRACT)
    assert floored == {"url", "source_type", "title", "trust_tier", "lang",
                       "domain"}
    for rec in news_records() + records():
        carried = {f for f in floored if rec.get(f)}
        assert carried == {"url", "lang"}, (rec.get("url"), carried)


def test_domain_is_derived_from_the_url_which_is_what_unblocks_the_licence():
    """aw_b1_projection reads record["domain"] and never derives it, so a row
    without one resolves NO licence -- every host rule is keyed on it. That is
    why the news family read `unresolved license` on a host that IS in the
    table."""
    assert P.derive_domain("https://www.rsi.ch/archivi/x") == "rsi.ch"
    assert P.derive_domain("https://sunfounder.com/products/y") == "sunfounder.com"
    assert P.derive_domain("") is None
    joined, _s, _m, _vk, _u = promote_news()
    for _pid, (_d, _sp, payload, _r) in joined.items():
        assert payload["domain"] == "rsi.ch"
        # and the licence now resolves, through the host rule
        assert payload["license"] == "copyright-fairuse-snippet-pointer"


def test_the_module_refuses_to_guess_source_type_or_trust_tier():
    """They are POLICY facts -- which §4 layer, and which trust class it grants."""
    with pytest.raises(P.PromotionError, match="trust_tier"):
        promote_news(family_payload={"source_type": "news"})
    with pytest.raises(P.PromotionError, match="source_type"):
        promote_news(family_payload={"trust_tier": 0.70})


def test_a_declared_value_never_overwrites_one_the_row_carries():
    """The pack is the authority on its own rows; the contract fills absences."""
    rows = [{**r, "trust_tier": 0.42} for r in news_records()]
    joined, _s, _m, _vk, _u = promote_news(rows)
    assert joined
    for _pid, (_d, _sp, payload, _r) in joined.items():
        assert payload["trust_tier"] == 0.42


def test_the_projected_payload_meets_the_carry_contract():
    """The floors the apply-time gate applies, met before the GPU or the write."""
    from cytir.index import payload_carry as PC
    joined, _s, _m, _vk, _u = promote_news()
    verdict = PC.assert_scoreable(
        (payload for _p, (_d, _s2, payload, _r) in joined.items()),
        label="task629 news fixture")
    assert verdict["rows_checked"] == 2
    assert all(f["ok"] for f in verdict["fields"])


# ------------------------------------------- POLICY 4.7 publication date
def test_the_date_is_lifted_from_where_the_row_actually_keeps_it():
    joined, _s, _m, _vk, undated = promote_news()
    assert len(joined) == 2
    assert {p["published"] for _i, (_d, _s2, p, _r) in joined.items()} == {
        "2026-07-17T06:00:00.000Z", "2026-07-12T06:00:00.000Z"}
    assert len(undated) == 1


def test_an_undated_row_is_excluded_and_named_not_backfilled():
    """POLICY 4.7: a news record without a date is inadmissible because it cannot
    age out. The excluded row is REPORTED (ADR-086: every row reaches one
    terminal), never silently dropped and never given a date we invented."""
    _j, _s, modes, _vk, undated = promote_news()
    assert modes["excluded_undated"] == 1
    row = undated[0]
    assert row["url"] == "https://www.rts.ch/archives"
    assert "no value at source_metadata.publication_time" in row["reason"]


def test_a_date_the_publisher_did_not_declare_is_not_a_date():
    """A value whose status is not `source_declared` is OUR inference, and
    admitting it would make the rolling window turn on a date nobody published."""
    rows = news_records()
    dated = [r for r in rows
             if (r.get("source_metadata") or {}).get("publication_time")]
    assert dated, "fixture must contain a dated row"
    mutated = []
    for rec in rows:
        meta = dict(rec.get("source_metadata") or {})
        if meta.get("publication_time"):
            meta["publication_time_status"] = "aw_inferred_from_url"
        mutated.append({**rec, "source_metadata": meta})
    joined, _s, modes, _vk, undated = promote_news(mutated)
    assert joined == {}
    assert modes["excluded_undated"] == len(rows)
    assert any("not the publisher's statement" in r["reason"] for r in undated)


def test_without_require_published_nothing_is_excluded():
    """The rule is opt-in per family, because only a dated family has it."""
    joined, _s, modes, _vk, undated = promote_news(require_published=False)
    assert len(joined) == 3
    assert undated == []
    assert "excluded_undated" not in modes


def test_dotted_reads_a_nested_path_and_survives_a_missing_hop():
    assert P.dotted({"a": {"b": {"c": 1}}}, "a.b.c") == 1
    assert P.dotted({"a": {}}, "a.b.c") is None
    assert P.dotted({"a": "scalar"}, "a.b") is None


# ---------------------------------------------------------------------------
# domain derivation -- licence-critical, and it used to be optional by accident
# ---------------------------------------------------------------------------
def test_domain_is_derived_without_a_family_payload():
    """The coupling this pins cost nothing only because it was found in a dry run.

    `aw_b1_projection` reads `record["domain"]` and never derives it, and EVERY
    licence host rule is keyed on it -- so a row without a domain resolves no
    licence and is refused, on a host that is in the table. The derivation used to
    live inside `apply_family_payload`, which runs only when a caller passes
    `--family-payload`. A pack whose rows already carry every declared payload
    field but no `domain` therefore failed with "unresolved license", and the
    operator's fix was to pass a family payload they did not otherwise need.

    Found dry-running B01's own fixture (task 629, 2026-09-22) -- before the
    batch, rather than inside a 75-minute locked hold.
    """
    joined, _skipped, _modes, _vk, _u = P.project_rows(
        iter(records()), consent=None, attribution=None, license_version=None,
        identity_check="remint-b1")          # NO family_payload
    assert joined
    for _dense, _sparse, payload, _rec in joined.values():
        assert payload.get("domain") == "sunfounder.com", payload.get("domain")
        assert payload.get("license"), "a domain-less row resolves no licence"


def test_a_domain_the_row_carries_is_never_overwritten():
    """The pack is the authority on its own rows; this only fills absences."""
    rows = [dict(row, domain="declared-by-the-pack.example") for row in records()]
    out = P.with_domain(rows[0])
    assert out["domain"] == "declared-by-the-pack.example"
    assert out is rows[0], "an untouched record need not be copied"


def test_with_domain_leaves_a_row_with_no_usable_url_alone():
    """No url means no derivation, and no invented domain either."""
    assert P.with_domain({"url": ""}) == {"url": ""}
    assert P.with_domain({}) == {}


def test_the_family_payload_path_uses_the_same_derivation():
    """One home. Two copies of this drift, and the second one is the silent one."""
    import inspect
    source = inspect.getsource(P.apply_family_payload)
    assert "with_domain(record)" in source
    assert "derive_domain(" not in source, (
        "apply_family_payload re-derives the domain instead of calling "
        "with_domain -- that is the second copy")


# ------------------------------------------------ task 629, 2026-09-23 fixes
# Two guards, both written after a measurement rather than after an incident --
# which is the only reason B02 and B03 did not enter prod as net-new.

SERVED = json.loads(
    (HERE / "fixtures" / "promote" / "task629_served_urls.json").read_text(encoding="utf-8"))


def test_the_domain_fold_is_aw_commons_and_www_n_hosts_fold_too():
    """`www150.statcan.gc.ca` -> `statcan.gc.ca`, which is what the index holds.

    This module used to strip a literal `www.` of its own. Under that rule B03
    would have entered carrying `domain = www150.statcan.gc.ca` while all
    13,861 rows of its own family already serving carry `statcan.gc.ca` -- the
    licence would still have resolved (longest-suffix) and nothing would have
    failed; the family would simply have been split in two for every `domain`
    filter and every site-search.
    """
    import cytir.core.common as AC
    assert P.derive_domain(
        "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1010000301"
    ) == "statcan.gc.ca"
    assert P.derive_domain(
        "https://www23.statcan.gc.ca/imdb/p2SV.pl?Id=8548") == "statcan.gc.ca"
    # and every pre-existing case is unchanged
    assert P.derive_domain("https://www.rfc-editor.org/rfc/rfc2616.txt") == "rfc-editor.org"
    assert P.derive_domain("https://www.rsi.ch/archivi/x") == "rsi.ch"
    assert P.derive_domain("https://sunfounder.com/products/y") == "sunfounder.com"
    assert P.derive_domain("") is None
    # one home, not a copy: the fold IS aw_common's
    assert AC.fold_www("www150.statcan.gc.ca") == "statcan.gc.ca"
    src = (ROOT / "cytir" / "index" / "promote_sealed_pack.py").read_text(encoding="utf-8")
    assert "startswith(\"www.\")" not in src, (
        "a second www fold has grown back in this module")


def _stub_qdrant(monkeypatch, served_urls):
    """A fake `_api` that answers only what `already_served_urls` asks, and
    RECORDS the calls -- the prod-safety property under test is that a net-new
    pack triggers no scroll at all."""
    calls = []

    def fake(url, path, body=None, method=None):
        calls.append((path, body))
        asked = []
        if body and body.get("filter"):
            asked = body["filter"]["must"][0]["match"]["any"]
        hits = [u for u in asked if u in served_urls]
        if path.endswith("/points/count"):
            return {"count": sum(served_urls[u] for u in hits)}
        if path.endswith("/points/scroll"):
            return {"points": [{"payload": {"url": u}} for u in hits],
                    "next_page_offset": None}
        raise AssertionError("unexpected call " + path)

    monkeypatch.setattr(P, "_api", fake)
    return calls


def test_already_served_urls_sees_what_already_present_cannot(monkeypatch):
    """The real B02 urls, with the twin's real row counts, are found."""
    served = {r["url"]: r["twin_rows"] for r in SERVED["b02_service_public_fiche"]}
    assert all(v > 0 for v in served.values()), "fixture must be served rows"
    calls = _stub_qdrant(monkeypatch, served)
    got = P.already_served_urls("http://x", "aw_e8", sorted(served))
    assert got == sorted(served)
    assert any(p.endswith("/points/scroll") for p, _ in calls)


def test_a_genuinely_net_new_pack_costs_no_scroll(monkeypatch):
    """B05's real urls: the twin serves none of them, so the guard counts and
    stops. That is what makes it safe to run against prod, where an unindexed
    or unnecessary scroll saturates the API pool (memory
    `prod-serving-box-fragile-scrolls`)."""
    urls = [r["url"] for r in SERVED["b05_rfc_editor_net_new"]]
    assert all(r["twin_rows"] == 0 for r in SERVED["b05_rfc_editor_net_new"])
    calls = _stub_qdrant(monkeypatch, {})
    assert P.already_served_urls("http://x", "aw_e8", urls) == []
    assert [p for p, _ in calls] == ["/collections/aw_e8/points/count"], (
        "a net-new pack must not scroll")


def _plan_with(monkeypatch, served, chunks=CHUNKS, vectors=VECTORS, **kw):
    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: served)
    monkeypatch.setattr(P.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    return P.plan(chunks_dir=chunks, vectors_dir=vectors, qdrant="http://x",
                  collection="aw_e8", mrl=256, identity_check="pack-declared",
                  identity_reason="task 629 test",
                  family_payload={"source_type": "docs", "trust_tier": 0.6},
                  title_from=("source_metadata.title",), **kw)


def test_plan_refuses_a_net_new_pack_that_lands_on_served_pages(monkeypatch):
    """The B02 shape: pages the target already serves, ids that do not collide.

    The verbatim `already_served_urls: N of M` is the contract -- an operator
    reading a refusal has to be able to see the size of what they were about
    to duplicate without opening a receipt.
    """
    urls = [r["url"] for r in SERVED["b02_service_public_fiche"]]
    with pytest.raises(P.PromotionError, match=r"already_served_urls: 3 of \d+"):
        _plan_with(monkeypatch, urls)


def test_a_named_supersede_reason_turns_the_refusal_into_a_count(monkeypatch):
    """It is a gate, not a wall: the replacement operation exists and is named.
    What it may not be is silent."""
    urls = [r["url"] for r in SERVED["b02_service_public_fiche"]]
    receipt = _plan_with(monkeypatch, urls,
                         allow_supersede="ruled by Tim, task 629 memo")
    assert receipt["rows"]["already_served_urls"] == 3
    assert receipt["allow_supersede"]


# ------------------------------------------ task 664: the 5% drop-and-land
# ceiling. Real incident, 2026-09-26: cc-authority-frontier-w7-tranche5
# refused WHOLE for 2 of 17,417 urls (0.01%) already served under other
# ids. Coordinator's ruling: drop the served rows, land the rest, refuse
# whole only past 5%. `_served_drop_decision` is a pure helper (no network,
# no vectors fixture) precisely because plan()'s own real fixture has only
# 3 urls total -- too small to exercise a sub-ceiling case end to end.

def test_served_drop_decision_computes_the_share_and_the_drop_set():
    joined = {f"pid-{i}": (None, None, {"url": f"http://x/{i}"}, None)
             for i in range(100)}
    served = ["http://x/0", "http://x/1"]  # 2 of 100 -- 2%, under the ceiling
    pids, share = P._served_drop_decision(joined, served)
    assert pids == frozenset({"pid-0", "pid-1"})
    assert share == 0.02


def test_served_drop_decision_at_exactly_the_ceiling_is_not_over_it():
    joined = {f"pid-{i}": (None, None, {"url": f"http://x/{i}"}, None)
             for i in range(100)}
    served = [f"http://x/{i}" for i in range(5)]  # 5 of 100 -- exactly 5%
    _pids, share = P._served_drop_decision(joined, served)
    assert share == 0.05
    assert not (share > P.ALREADY_SERVED_FAIL_SHARE_MAX)


def test_served_drop_decision_past_the_ceiling():
    joined = {f"pid-{i}": (None, None, {"url": f"http://x/{i}"}, None)
             for i in range(100)}
    served = [f"http://x/{i}" for i in range(6)]  # 6 of 100 -- 6%
    _pids, share = P._served_drop_decision(joined, served)
    assert share > P.ALREADY_SERVED_FAIL_SHARE_MAX


def test_served_drop_decision_empty_pack_is_zero_share_not_a_crash():
    assert P._served_drop_decision({}, []) == (frozenset(), 0.0)


def _synthetic_rows_and_vectors(n: int):
    """`n` synthetic rows/vectors cloned from the real 3-row fixture (same
    schema, distinct url/point_id per row -- pack-declared identity trusts
    the given point_id as-is, so no re-minting is needed) plus a matching
    `(matrix, vector_ids)` pair, so `plan()` can run its REAL pipeline
    (project_rows -> vector join -> already_served check) at a scale the
    on-disk fixture is too small for."""
    import numpy as np
    base = licensed_records()[0]
    rows, ids, vecs = [], [], []
    real_matrix, _real_ids = P.read_vectors(VECTORS, 256)
    for i in range(n):
        pid = f"synthetic-pid-{i:04d}"
        rows.append({**base, "url": f"https://example.org/page-{i:04d}",
                    "point_id": pid})
        ids.append(pid)
        vecs.append(real_matrix[0])
    return rows, (np.stack(vecs), ids)


def test_plan_drops_a_small_served_share_and_lands_the_rest(monkeypatch):
    """End to end through plan() -- 2 of 100 urls (2%, under the ceiling)
    already served: plan() must NOT raise, must drop exactly those 2 rows
    from `_joined`, and must record `already_served_skipped`."""
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(100)
    served = [rows[0]["url"], rows[1]["url"]]

    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "read_records", lambda *_a, **_k: iter(rows))
    monkeypatch.setattr(P, "read_vectors", lambda *_a, **_k: (matrix, vector_ids))
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: served)
    monkeypatch.setattr(P.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})

    receipt = P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
                     collection="aw_e8", mrl=256, **PACK_DECLARED)
    assert receipt["rows"]["already_served_urls"] == 2
    assert receipt["_already_served_pids"] == frozenset(
        {"synthetic-pid-0000", "synthetic-pid-0001"})
    # unfiltered here -- drain()/the CLI drop these AFTER plan() returns
    # (same pattern as a mechanical-content drop), so plan() itself must
    # still report the full population it measured.
    assert len(receipt["_joined"]) == 100


def test_plan_still_refuses_past_the_5_percent_ceiling_even_with_many_urls(
        monkeypatch):
    """No regression at scale: 6 of 100 (6%) must still refuse whole, the
    same as the small real-fixture case above."""
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(100)
    served = [r["url"] for r in rows[:6]]

    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "read_records", lambda *_a, **_k: iter(rows))
    monkeypatch.setattr(P, "read_vectors", lambda *_a, **_k: (matrix, vector_ids))
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: served)
    monkeypatch.setattr(P.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})

    with pytest.raises(P.PromotionError, match=r"already_served_urls: 6 of 100"):
        P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
              collection="aw_e8", mrl=256, **PACK_DECLARED)


def test_plan_raises_already_landed_error_not_a_bare_promotion_error(monkeypatch):
    """Task 664, 2026-09-26 incident: a caller applying the same written_ids
    to TWO targets (twin, then prod) needs to tell "this population already
    landed HERE" apart from every other `plan()` refusal without parsing a
    message string -- `promote_queue.py`'s resume-after-partial-write logic
    catches `AlreadyLandedError` specifically. Still a `PromotionError`
    subclass, so every EXISTING bare-`PromotionError`/`REFUSAL_EXCEPTIONS`
    catch site keeps working unchanged."""
    clean = _plan_with(monkeypatch, [])
    written_ids = frozenset(clean["_joined"])
    assert written_ids, "fixture must actually produce rows"

    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "already_present",
                        lambda *_a, **_k: sorted(written_ids))
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(P.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    with pytest.raises(P.AlreadyLandedError) as exc_info:
        P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
              collection="aw_e8", mrl=256, identity_check="pack-declared",
              identity_reason="task 629 test",
              family_payload={"source_type": "docs", "trust_tier": 0.6},
              title_from=("source_metadata.title",))
    assert exc_info.value.ids == written_ids
    assert isinstance(exc_info.value, P.PromotionError)


def test_already_landed_error_carries_the_present_subset_not_only_ids(monkeypatch):
    """Task 690, 2026-10-02: ONE already-present id of 169,070 raised this
    error, and the drain had no way to name that id -- only the full
    population rode on the exception. `present` must be exactly the subset
    the target already holds."""
    clean = _plan_with(monkeypatch, [])
    written_ids = sorted(clean["_joined"])
    assert len(written_ids) >= 2, "fixture must produce at least two rows"
    one = written_ids[:1]

    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: list(one))
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: [])
    with pytest.raises(P.AlreadyLandedError) as exc_info:
        P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
              collection="aw_e8", mrl=256, identity_check="pack-declared",
              identity_reason="task 629 test",
              family_payload={"source_type": "docs", "trust_tier": 0.6},
              title_from=("source_metadata.title",))
    assert exc_info.value.present == frozenset(one)
    assert exc_info.value.ids == frozenset(written_ids)


def test_a_clean_pack_reports_zero_rather_than_omitting_the_count(monkeypatch):
    """B05's shape. `already_served_urls: 0 of M` in the receipt is what makes
    a net-new claim checkable after the fact; a count that only appears on
    failure teaches nobody anything."""
    receipt = _plan_with(monkeypatch, [])
    assert receipt["rows"]["already_served_urls"] == 0
    assert receipt["rows"]["distinct_urls"] == receipt["rows"]["projected"]


def test_the_retired_remint_ruling_is_corrected_in_place_not_deleted():
    """`remint-b1`'s premise was falsified, and the module says so.

    A retired ruling that simply disappears leaves the next reader to re-derive
    why the mode exists and to reach for it again.
    """
    doc = P.__doc__
    assert "was WRONG" in doc and "252 combinations" in doc
    assert "RULED 2026-09-23" in doc
    assert "61,212 / 61,212" in doc
    assert "remint-b1" in P.IDENTITY_MODES, (
        "the mode stays available for a pack whose ids really cannot be "
        "re-derived; what changed is which one W1 uses")


# ------------------------------------------------------- licence lift (641)
def test_lift_license_reads_a_nested_path():
    rec = {"url": "https://apple.stackexchange.com/a/9",
          "source_metadata": {"accepted_answer": {"license": "CC-BY-SA-2.5"}}}
    out, why = P.lift_license(
        rec, path="source_metadata.accepted_answer.license")
    assert out["license"] == "CC-BY-SA-2.5"
    assert why is None


def test_lift_license_never_overwrites_a_top_level_value():
    rec = {"url": "https://x", "license": "already-here",
          "source_metadata": {"accepted_answer": {"license": "different"}}}
    out, why = P.lift_license(
        rec, path="source_metadata.accepted_answer.license")
    assert out["license"] == "already-here"
    assert why is None
    assert out is rec, "no copy made when there is nothing to lift"


def test_lift_license_names_the_reason_when_the_path_is_absent():
    rec = {"url": "https://x", "source_metadata": {}}
    out, why = P.lift_license(rec, path="source_metadata.accepted_answer.license")
    assert "license" not in out
    assert "no value at source_metadata.accepted_answer.license" in why


def test_project_rows_license_from_lifts_a_real_pool_a_shaped_row(monkeypatch):
    """The real B01 fixture rows carry no `license` at all (TEST_LICENCE is
    added by hand elsewhere in this file) -- prove --license-from admits one
    of them by lifting a synthetic nested field, the same shape
    stackexchange_nonso_qa_p0's real rows use."""
    rows = records()
    mutated = [{**r, "source_metadata": {"accepted_answer": {
        "license": "CC-BY-SA-3.0"}}} for r in rows]
    joined, skipped, _modes, _vk, _undated = P.project_rows(
        iter(mutated), consent=None, attribution=None, license_version=None,
        license_from="source_metadata.accepted_answer.license",
        **PACK_DECLARED)
    assert skipped == 0
    assert len(joined) == len(rows)
    for _pid, (_d, _s, payload, _rec) in joined.items():
        assert payload["license"] == "CC-BY-SA-3.0"


def test_project_rows_without_license_from_is_unchanged(monkeypatch):
    """No regression: omitting --license-from (the default) must project
    exactly what it did before this module gained the parameter."""
    rows = records()
    before = P.project_rows(iter(rows), consent=None, attribution=None,
                            license_version=None, **PACK_DECLARED)
    after = P.project_rows(iter(rows), consent=None, attribution=None,
                           license_version=None, license_from=None,
                           **PACK_DECLARED)
    assert set(before[0]) == set(after[0])
    assert before[1] == after[1]  # skipped count


# ------------------------------------------------------- family licence (641)
def test_family_license_stamps_a_row_with_no_host_rule_and_no_field(monkeypatch):
    """ADR-111's family-level (not per-host) basis: a real B01 row, with its
    domain swapped to one that resolves NO host rule, must otherwise be
    unlicensable -- and family_license fills exactly that gap."""
    rows = records()
    unresolvable = [{**r, "domain": "no-such-host-rule-example.test",
                    "url": r["url"].replace("sunfounder.com",
                                            "no-such-host-rule-example.test")}
                    for r in rows]
    with pytest.raises(P.PROJ.ProjectionError, match="unresolved license"):
        # fixture assumption: this domain really has no host rule --
        # require_license=True raises straight through project_rows(); the
        # next call proves family_license clears exactly this refusal.
        P.project_rows(iter(unresolvable), consent=None, attribution=None,
                       license_version=None, **PACK_DECLARED)

    joined, skipped, _modes, _vk, _undated = P.project_rows(
        iter(unresolvable), consent=None, attribution=None,
        license_version=None, family_license="ADR-111-opt-out-minimum-record",
        **PACK_DECLARED)
    assert skipped == 0
    assert len(joined) == len(rows)
    for _pid, (_d, _s, payload, _rec) in joined.items():
        assert payload["license"] == "ADR-111-opt-out-minimum-record"


def test_family_license_never_overwrites_a_row_that_already_resolves():
    """The real B01 rows DO resolve a host-rule license (sunfounder.com) --
    family_license must not stamp over that."""
    rows = records()
    joined, _skipped, _modes, _vk, _undated = P.project_rows(
        iter(rows), consent=None, attribution=None, license_version=None,
        family_license="SHOULD-NOT-APPEAR", **PACK_DECLARED)
    for _pid, (_d, _s, payload, _rec) in joined.items():
        assert payload["license"] != "SHOULD-NOT-APPEAR"


# --------------------------------------------------------- _api error body
def test_api_includes_the_response_body_on_an_http_error(monkeypatch):
    """Bug found live, task 641, 2026-09-23: the real news-backlog apply
    aborted with a bare `HTTP Error 400: Bad Request` -- no indication of
    WHY (a vector name/dimension mismatch, a bad payload type, a malformed
    filter, a point id format). urllib's HTTPError carries Qdrant's actual
    error body, readable exactly once on the exception object; `_api` must
    read it before it's gone and put it in the message every caller sees."""
    import io
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=300):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", hdrs=None,
            fp=io.BytesIO(
                b'{"status":{"error":"Wrong input: Vector dimension error: '
                b'expected dim: 1024, got 256"},"time":0.0001}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(P.PromotionError) as exc_info:
        P._api("http://x", "/collections/aw_e8/points?wait=true",
              {"points": []}, method="PUT")
    msg = str(exc_info.value)
    assert "400" in msg
    assert "Vector dimension error" in msg
    assert "expected dim: 1024, got 256" in msg


def test_project_rows_exclude_ids_drops_before_payload_carry(monkeypatch):
    """Orchestrator ruling, 2026-09-23 (CC tranche 1, task 641): a row a
    caller has already determined is out-of-scope-language must be gone
    BEFORE it reaches anything downstream, including the payload_carry
    census -- excluded the same way `undated` rows are, never a partial
    join."""
    rows = records()
    excluded_id = rows[0]["point_id"]
    joined, _skipped, modes, _vk, _undated = P.project_rows(
        iter(rows), consent=None, attribution=None, license_version=None,
        exclude_ids=frozenset({excluded_id}), **PACK_DECLARED)
    assert excluded_id not in joined
    assert len(joined) == len(rows) - 1
    assert modes["excluded_out_of_scope_language"] == 1
    dropped = modes["out_of_scope_dropped_rows"]
    assert dropped[0]["point_id"] == excluded_id
    assert dropped[0]["reason"] == "out_of_scope_language"


def test_project_rows_without_exclude_ids_is_unchanged():
    """No regression: omitting `exclude_ids` (the default) must project
    exactly what it did before this module gained the parameter -- no
    `excluded_out_of_scope_language` key at all, not even 0."""
    rows = records()
    joined, _skipped, modes, _vk, _undated = P.project_rows(
        iter(rows), consent=None, attribution=None, license_version=None,
        **PACK_DECLARED)
    assert len(joined) == len(rows)
    assert "excluded_out_of_scope_language" not in modes
    assert "out_of_scope_dropped_rows" not in modes


def test_project_rows_exclude_ids_mapping_carries_its_own_reason():
    """Tim's ruling 2026-09-24 (CC tranche 1, task 652): `exclude_ids` as a
    `{id: reason}` mapping drops with THAT row's own reason, not the
    frozenset default -- two different populations excluded in the same
    call, each correctly labeled."""
    rows = records()
    id_a, id_b = rows[0]["point_id"], rows[1]["point_id"]
    joined, _skipped, modes, _vk, _undated = P.project_rows(
        iter(rows), consent=None, attribution=None, license_version=None,
        exclude_ids={id_a: "out_of_scope_language", id_b: "no_confident_lang"},
        **PACK_DECLARED)
    assert id_a not in joined and id_b not in joined
    dropped = {d["point_id"]: d["reason"] for d in modes["out_of_scope_dropped_rows"]}
    assert dropped[id_a] == "out_of_scope_language"
    assert dropped[id_b] == "no_confident_lang"
    assert modes["excluded_out_of_scope_language"] == 2


def test_plan_reports_out_of_scope_language_dropped_rows(monkeypatch):
    """End to end through plan() -- the real caller -- with a stubbed
    persisted-inputs/payload-carry/licence chain (this fixture's own
    identity/consent concerns are orthogonal to the exclude_ids wiring
    under test)."""
    rows = records()
    excluded_id = rows[0]["point_id"]
    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "read_records", lambda *_a, **_k: iter(rows))
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(P.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    real_vectors, real_ids = P.read_vectors(VECTORS, 256)
    monkeypatch.setattr(P, "read_vectors", lambda *_a, **_k: (real_vectors, real_ids))

    receipt = P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
                     collection="aw_e8", mrl=256,
                     exclude_ids=frozenset({excluded_id}), **PACK_DECLARED)
    assert receipt["rows"]["excluded_out_of_scope_language"] == 1
    dropped = receipt["out_of_scope_language_dropped"]
    assert dropped[0]["point_id"] == excluded_id
    assert excluded_id not in receipt["_joined"]


def test_api_caps_a_very_long_error_body(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=300):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", hdrs=None,
            fp=io.BytesIO(b"x" * 20000))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(P.PromotionError) as exc_info:
        P._api("http://x", "/collections/aw_e8/points", method="GET")
    assert len(str(exc_info.value)) < 5000


def test_api_retries_a_raw_timeouterror_and_succeeds(monkeypatch):
    """Real shape, task 641, 2026-09-23: a PUT to prod timed out at 300s
    under heavy unrelated build-box load, raised as a raw TimeoutError (not
    wrapped in URLError) from deep inside http.client's response read.
    Idempotent PUT -- safe, and must be retried rather than aborting the
    whole drain."""
    import io
    import urllib.request

    calls = {"n": 0}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status": "ok", "result": {}}'

    def flaky_urlopen(req, timeout=300):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    result = P._api("http://x", "/collections/aw_e8/points?wait=true",
                    {"points": []}, method="PUT")
    assert result == {}
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_api_gives_up_after_max_timeout_retries(monkeypatch):
    import urllib.request

    calls = {"n": 0}

    def always_times_out(req, timeout=300):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", always_times_out)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        P._api("http://x", "/collections/aw_e8/points", method="GET")
    assert calls["n"] == P._API_TIMEOUT_RETRIES + 1  # first try + N retries


def test_api_does_not_retry_a_non_timeout_urlerror(monkeypatch):
    """Connection refused, DNS failure, etc. are real failures, not
    transient contention -- must not be retried as if they were."""
    import urllib.error
    import urllib.request

    calls = {"n": 0}

    def refused(req, timeout=300):
        calls["n"] += 1
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(urllib.error.URLError):
        P._api("http://x", "/collections/aw_e8/points", method="GET")
    assert calls["n"] == 1  # never retried


# ------------------------------------------------ lookup width (task 689)
def test_lookups_are_identical_at_any_width(monkeypatch):
    """Task 689: the already-present / already-served round trips may overlap,
    but the answer, its order and the calls made must be what the sequential
    loop gave -- a net-new batch still costs a count and no scroll."""
    import threading
    lock = threading.Lock()
    calls = []
    held = {f"id{i}" for i in range(0, 4000, 7)}
    served = {f"https://x.test/{i}": 3 for i in range(0, 3000, 11)}

    def fake(url, path, body=None, method=None):
        with lock:
            calls.append((path, json.dumps(body, sort_keys=True)))
        if path.endswith("/points/count"):
            asked = body["filter"]["must"][0]["match"]["any"]
            return {"count": sum(served.get(u, 0) for u in asked)}
        if path.endswith("/points/scroll"):
            asked = body["filter"]["must"][0]["match"]["any"]
            return {"points": [{"payload": {"url": u}} for u in asked if u in served],
                    "next_page_offset": None}
        return [{"id": i} for i in body["ids"] if i in held]

    monkeypatch.setattr(P, "_api", fake)
    ids = [f"id{i}" for i in range(4000)]
    urls = [f"https://x.test/{i}" for i in range(3000)]
    seq = (P.already_present("http://x", "c", ids),
           P.already_served_urls("http://x", "c", urls))
    seq_calls = sorted(calls)
    calls.clear()
    par = (P.already_present("http://x", "c", ids, workers=8),
           P.already_served_urls("http://x", "c", urls, workers=8))
    assert par == seq, "width changed the answer or its order"
    assert sorted(calls) == seq_calls, "width changed which calls were made"
    assert seq[0] and seq[1], "fixture must exercise hits"


def test_a_failing_chunk_fails_the_whole_lookup_at_any_width(monkeypatch):
    """Task 689 review: one chunk's PromotionError must surface from the
    parallel path, never be swallowed into a partial 'not present' answer."""
    def fake(url, path, body=None, method=None):
        if "id13" in body["ids"]:
            raise P.PromotionError("boom")
        return []

    monkeypatch.setattr(P, "_api", fake)
    ids = [f"id{i}" for i in range(4000)]
    for workers in (1, 8):
        with pytest.raises(P.PromotionError, match="boom"):
            P.already_present("http://x", "c", ids, workers=workers)


# ------------------------------------------ incident 2026-10-07: prod is never asked the URL question
def test_plan_against_prod_does_not_ask_the_url_question(monkeypatch):
    """A prod target (the alias, or the tunnel) is not asked `already_served_urls`: the twin leg
    decided it, and a scroll per hit batch on the serving box made search answer 503."""
    def boom(*_a, **_k):
        raise AssertionError("already_served_urls was asked of prod")
    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", boom)
    monkeypatch.setattr(P.PI, "assert_pages_persisted", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    prod_tunnel = "http://127.0.0.1:%d" % P.PG.PROD_PORT
    rec = P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant=prod_tunnel,
                 collection="aw_prod", mrl=256, identity_check="pack-declared",
                 identity_reason="incident test",
                 family_payload={"source_type": "docs", "trust_tier": 0.6},
                 title_from=("source_metadata.title",))
    assert rec["rows"]["already_served_urls"] is None
    assert "not asked of prod" in rec["rows"]["already_served_checked_against"]
    assert rec["_already_served_pids"] == frozenset()


# ------------------------------------------ task 2224: projection-rejected terminal
def _plan_synthetic(monkeypatch, rows, matrix, vector_ids):
    monkeypatch.setattr(P, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(P, "read_records", lambda *_a, **_k: iter(rows))
    monkeypatch.setattr(P, "read_vectors", lambda *_a, **_k: (matrix, vector_ids))
    monkeypatch.setattr(P, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(P, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(P.PI, "assert_pages_persisted", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(P.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    return P.plan(chunks_dir=CHUNKS, vectors_dir=VECTORS, qdrant="http://x",
                  collection="aw_e8", mrl=256, **PACK_DECLARED)


def test_a_search_result_row_is_a_reported_terminal_and_its_vector_is_not_written(monkeypatch):
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(4)
    rows[2] = {**rows[2], "url": "https://example.org/search?q=steuer"}
    receipt = _plan_synthetic(monkeypatch, rows, matrix, vector_ids)
    rep = receipt["projection_rejected"]
    assert rep["total"] == 1 and rep["by_reason"] == {"projection-rejected: search-result-url": 1}
    assert rep["rows"][0]["point_id"] == "synthetic-pid-0002"
    assert "synthetic-pid-0002" not in receipt["_joined"]
    assert len(receipt["_joined"]) == 3
    assert receipt["rows"]["skipped_by_projection"] == 1


def test_a_vector_with_no_row_for_another_cause_is_still_refused(monkeypatch):
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(4)
    rows = rows[:3]  # the 4th vector has no row and no declared terminal
    with pytest.raises(P.PromotionError, match="vector/text join is not total"):
        _plan_synthetic(monkeypatch, rows, matrix, vector_ids)


def test_a_pack_with_zero_rejected_rows_plans_exactly_as_before(monkeypatch):
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(4)
    receipt = _plan_synthetic(monkeypatch, rows, matrix, vector_ids)
    assert "projection_rejected" not in receipt
    assert sorted(receipt["_joined"]) == vector_ids


# ------------------------------------------ task 2242: shared declared id, one reason vocabulary
def test_an_empty_text_row_is_reason_other_not_a_search_url(monkeypatch):
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(4)
    rows[1] = {**rows[1], "text": ""}
    receipt = _plan_synthetic(monkeypatch, rows, matrix, vector_ids)
    assert receipt["projection_rejected"]["by_reason"] == {"projection-rejected: other": 1}


def test_the_plan_reasons_are_composes_reasons():
    from cytir.index import compose
    from cytir.core import search_result_url as SRU
    assert compose.REJECTED_REASON == SRU.REJECTED_REASON
    assert SRU.rejected_reason("https://example.org/search?q=x") == SRU.REASON_SEARCH_RESULT_URL
    assert SRU.rejected_reason("") == SRU.REASON_OTHER


@pytest.mark.parametrize("rejected_first", [True, False])
def test_a_declared_id_shared_by_an_accepted_and_a_rejected_record_is_refused(monkeypatch, rejected_first):
    rows, (matrix, vector_ids) = _synthetic_rows_and_vectors(4)
    twin = {**rows[1], "url": "https://example.org/search?q=steuer"}
    rows = [rows[0], twin, rows[1], *rows[2:]] if rejected_first else [*rows, twin]
    with pytest.raises(P.PromotionError, match=r"1 declared point_id\(s\) are shared by a record "
                                              r"the projection accepts and a record it rejects.*"
                                              r"first: synthetic-pid-0001"):
        _plan_synthetic(monkeypatch, rows, matrix, vector_ids)


def test_a_shared_id_whose_accepted_record_is_excluded_by_minted_id_still_plans():
    rows, (_m, _ids) = _synthetic_rows_and_vectors(3)
    rows = [{**r, "license": TEST_LICENCE} for r in rows]
    out, *_ = projected([rows[1]], identity_check="remint-b1", identity_reason="probe")
    minted = next(iter(out))
    assert minted != rows[1]["point_id"]
    twin = {**rows[1], "url": "https://example.org/search?q=steuer"}
    projected([rows[0], rows[1], twin, rows[2]], identity_check="remint-b1",
              identity_reason="probe", exclude_ids={minted: "served"})


def test_a_shared_id_whose_accepted_record_is_i1_merged_away_still_plans():
    rows, (_m, _ids) = _synthetic_rows_and_vectors(3)
    rows = [{**r, "license": TEST_LICENCE} for r in rows]
    x = {**rows[2]}
    z = {**rows[2], "point_id": "synthetic-pid-0001"}  # same url+text, smaller id: the survivor
    y = {**rows[2], "url": "https://example.org/search?q=steuer"}  # declared 0002, rejected
    projected([rows[0], x, z, y], id_rule="I1")
