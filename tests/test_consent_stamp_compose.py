"""Task 684 / ruling 20: a newly composed pack carries consent flags on every row.

The rows are REAL (sampled from the build box promotion queue on 2026-10-02, text cut to
200 chars; `testdata/task684_compose_real_rows.json`), one family class each:

  dump family, contract grant        de_openlegaldata_v1, stackoverflow_qa, stackexchange_nonso_qa_p0
  official dump (ADR-076 pattern)    loda (Etalab-2.0, `consent_basis: official-dump-license`)
  crawled host with a cache verdict  cc_authority_frontier_w7 (its own flags removed)
  sealed live-per-origin contract    eric_official_pdf, openalex_academic_dump (r6)

and the refusals: no cache record, an undeclared family, a licence the instrument rule
cannot read, a declined host, a missing cache directory.
"""
import gzip
import tempfile
import json
import pathlib

import pytest

import cytir.consent.stamp as ST
import cytir.index.compose as compose
from cytir.consent.derive import WRITTEN_BASES
from cytir.index import payload_profile as PP

HERE = pathlib.Path(__file__).resolve().parent
from tests.helpers.sealed_evidence import require_sealed_probe, sealed_eric_probe  # noqa: F401,E402
ROWS = json.loads((HERE / "testdata" / "task684_compose_real_rows.json").read_text())
AS_OF = "2026-11-09T00:00:00Z"
_REAL_DEFAULT_CACHE_DIR = ST.default_cache_dir      # before the autouse fixture patches it


def _case(family, n=2):
    return [c for c in ROWS if c["family"] == family][:n]


def _ctx(cache=None):
    c = ST.load_context(as_of=AS_OF, cache_dir=tempfile.mkdtemp())   # an empty dir; the cache is injected below
    c.cache_index = cache or {}
    return c


def _write(dirpath, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    with gzip.open(dirpath / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _read(dirpath):
    with gzip.open(next(pathlib.Path(dirpath).glob("shard-*.texts.jsonl.gz")), "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _stamp(tmp_path, family, rows, ctx=None, **kw):
    src, out = tmp_path / "in", tmp_path / "out"
    _write(src, rows)
    rep = ST.stamp_chunks_dir(str(src), str(out), ctx or _ctx(), family=family, **kw)
    return rep, _read(out)


SEALED_ERIC_AT = "2026-11-08T05:14:56Z"      # files.eric.ed.gov in the task 2531 pool-a consent record


@pytest.mark.parametrize("family,rule,basis,checked", [
    ("de_openlegaldata_v1", "r0-contract-grant", "contract-grant", AS_OF),
    ("stackexchange_nonso_qa_p0", "r2-stackexchange", "dump-licence", AS_OF),
    ("stackoverflow_qa", "r2-stackexchange", "dump-licence", AS_OF),
    ("eric_official_pdf", "r6-contract-sealed-probe", "contract-sealed-probe", SEALED_ERIC_AT),
    ("openalex_academic_dump", "r1-openalex", "dump-licence", AS_OF),
])
def test_declared_family_rows_carry_all_four_fields(tmp_path, family, rule, basis, checked):
    if family == "eric_official_pdf":
        require_sealed_probe()
    cases = _case(family)
    rows = [c["row"] for c in cases]
    rep, out = _stamp(tmp_path, family, rows, license_from=cases[0]["license_from"],
                      family_license=cases[0]["family_license"])
    assert rep["rows_in"] == rep["rows_out"] == rep["verified_with_flags"] == len(rows)
    assert list(rep["rules"]) == [f"A|{rule}"]
    for r in out:
        assert r["search_ok"] is True and r["train_ok"] is True
        assert r["consent_basis"] == basis and r["consent_checked_at"] == checked
        assert basis in WRITTEN_BASES


def test_official_dump_row_keeps_its_dump_basis_and_gets_flags(tmp_path):
    rows = [c["row"] for c in _case("loda_exact")]
    assert all(r["consent_basis"] == "official-dump-license" and "search_ok" not in r for r in rows)
    rep, out = _stamp(tmp_path, "loda_exact", rows)
    assert list(rep["rules"]) == ["A|r5-official-dump"]
    assert all(r["search_ok"] is True and r["train_ok"] is True
               and r["consent_basis"] == "official-dump-license" for r in out)


def test_crawled_host_takes_the_cache_verdict_and_its_own_checked_at(tmp_path):
    rows = []
    for c in _case("cc_authority_frontier_w7"):
        r = dict(c["row"])
        for k in ("search_ok", "train_ok", "consent_checked_at"):
            r.pop(k, None)
        rows.append(r)
    cache = {}
    for r in rows:
        host = ST.host_of(r["url"]) if hasattr(ST, "host_of") else r["domain"]
        cache[r["domain"]] = {"cc": {"status": "ok", "search_ok": True, "train_ok": False,
                                     "checked_at": "2026-09-26T09:53:07Z"}}
    rep, out = _stamp(tmp_path, "cc_authority_frontier_w7", rows, ctx=_ctx(cache))
    assert list(rep["rules"]) == ["B|r3-host-cache"]
    assert all(r["search_ok"] is True and r["train_ok"] is False
               and r["consent_basis"] == "host-cache"
               and r["consent_checked_at"] == "2026-09-26T09:53:07Z" for r in out)


def test_rows_with_flags_are_kept_as_they_are(tmp_path):
    rows = [c["row"] for c in _case("cc_authority_frontier_w7")]
    assert all(r["search_ok"] is True for r in rows)
    rep, out = _stamp(tmp_path, "cc_authority_frontier_w7", rows)
    assert rep["class_e_kept"] == len(rows) and out == rows


@pytest.mark.usefixtures("sealed_eric_probe")
def test_restamping_a_stamped_pack_changes_nothing(tmp_path):
    rows = [c["row"] for c in _case("eric_official_pdf")]
    _rep, once = _stamp(tmp_path / "a", "eric_official_pdf", rows)
    rep2, twice = _stamp(tmp_path / "b", "eric_official_pdf", once)
    assert twice == once and rep2["class_e_kept"] == len(once)


# ---- refusals: nothing is written, nothing is defaulted ---------------------------------

def _refuses(tmp_path, family, rows, match, ctx=None, **kw):
    src, out = tmp_path / "in", tmp_path / "out"
    _write(src, rows)
    with pytest.raises(ST.ConsentStampRefused, match=match):
        ST.stamp_chunks_dir(str(src), str(out), ctx or _ctx(), family=family, **kw)
    assert not out.exists() and not (tmp_path / "out.partial").exists()


def _no_flags(family):
    out = []
    for c in _case(family):
        r = dict(c["row"])
        for k in ("search_ok", "train_ok", "consent_checked_at"):
            r.pop(k, None)
        out.append(r)
    return out


def test_crawled_row_with_no_cache_record_refuses(tmp_path):
    _refuses(tmp_path, "cc_authority_frontier_w7", _no_flags("cc_authority_frontier_w7"),
             "no-usable-cache-record")


def test_undeclared_family_refuses_every_unflagged_row(tmp_path):
    cache = {r["domain"]: {"cc": {"status": "ok", "search_ok": True, "train_ok": True,
                                  "checked_at": "2026-09-26T00:00:00Z"}}
             for r in _no_flags("cc_authority_frontier_w7")}
    _refuses(tmp_path, "family_nobody_declared", _no_flags("cc_authority_frontier_w7"),
             "family-undeclared", ctx=_ctx(cache))


def test_instrument_family_with_unreadable_licence_refuses(tmp_path):
    rows = [c["row"] for c in _case("stackoverflow_qa")]
    _refuses(tmp_path, "stackoverflow_qa", rows, "instrument-licence-absent-or-mismatch",
             license_from="source_metadata.no_such_path")


def test_a_declined_host_never_gets_search_ok_false_stamped(tmp_path):
    cache = {r["domain"]: {"cc": {"status": "ok", "search_ok": False, "train_ok": False,
                                  "checked_at": "2026-09-26T00:00:00Z"}}
             for r in _no_flags("cc_authority_frontier_w7")}
    _refuses(tmp_path, "cc_authority_frontier_w7", _no_flags("cc_authority_frontier_w7"),
             "cache-search-false-on-served-row", ctx=_ctx(cache))


def test_arxiv_without_the_per_paper_licence_map_refuses(tmp_path):
    row = {"url": "https://arxiv.org/abs/2101.00001", "license": "arXiv-metadata-CC0", "text": "x"}
    _refuses(tmp_path, "named_slice:arxiv", [row], "arxiv-licence-map-not-supplied")


def test_one_bad_row_refuses_the_whole_pack(tmp_path):
    good = [c["row"] for c in _case("eric_official_pdf")]
    bad = {"url": "https://example.org/x", "text": "x", "license": "none declared"}
    # eric is r6 for any host, so use a family that needs the cache for the bad row
    cc = _no_flags("cc_authority_frontier_w7")
    cache = {cc[0]["domain"]: {"cc": {"status": "ok", "search_ok": True, "train_ok": True,
                                      "checked_at": "2026-09-26T00:00:00Z"}}}
    other = dict(cc[1], domain="unprobed.example", url="https://unprobed.example/p")
    _refuses(tmp_path, "cc_authority_frontier_w7", [cc[0], other], "1 of 2", ctx=_ctx(cache))
    assert good and bad


def test_missing_cache_directory_refuses_instead_of_reading_as_empty(tmp_path):
    with pytest.raises(ST.ConsentStampRefused, match="does not exist"):
        ST.load_context(as_of=AS_OF, cache_dir=str(tmp_path / "nope"))


# ---- compose end to end (real ERIC rows with identity) ---------------------------------

ERIC_ROWS = HERE.parent / "tests" / "fixtures" / "index_compose" / "task657_eric_real_rows.jsonl"


def _vec(vdir, n):
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": n, "dim": 1024, "dtype": "float16"}))


@pytest.mark.usefixtures("sealed_eric_probe")
def test_compose_pack_seals_a_pack_whose_rows_all_carry_the_flags(tmp_path):
    rows = [json.loads(line) for line in ERIC_ROWS.read_text().splitlines() if line.strip()]
    assert not any("search_ok" in r for r in rows)
    _write(tmp_path / "chunks", rows)
    _vec(tmp_path / "vectors", len(rows))
    res = compose.compose_pack(
        pack_id="t684-eric", family="eric_official_pdf", chunks_dir=str(tmp_path / "chunks"),
        vectors_dir=str(tmp_path / "vectors"), collection="aw_e8", mrl=256,
        queue_dir=str(tmp_path / "queue"), consent_ctx=_ctx())
    man = res["manifest"]
    assert man["pack_chunks"].endswith("chunks-consent-stamped")
    assert man["consent_stamp"]["verified_with_flags"] == man["rows"] == len(rows)
    out = _read(man["pack_chunks"])
    assert len(out) == len(rows) and all(
        r["search_ok"] is True and r["train_ok"] is True
        and r["consent_checked_at"] == SEALED_ERIC_AT      # the sealed record's OWN time
        and r["consent_basis"] == "contract-sealed-probe" for r in out)
    # identity is unchanged: the stamp added payload fields only
    assert [r["point_id"] for r in out] == [r["point_id"] for r in rows]


def test_compose_pack_refuses_before_any_manifest_when_a_row_cannot_be_stamped(tmp_path):
    rows = _no_flags("cc_authority_frontier_w7")
    _write(tmp_path / "chunks", rows)
    _vec(tmp_path / "vectors", len(rows))
    with pytest.raises(ST.ConsentStampRefused):
        compose.compose_pack(
            pack_id="t684-cc", family="cc_authority_frontier_w7", chunks_dir=str(tmp_path / "chunks"),
            vectors_dir=str(tmp_path / "vectors"), collection="aw_e8", mrl=256,
            queue_dir=str(tmp_path / "queue"), consent_ctx=_ctx(),
            anchor_fn=lambda r: r["anchor"] if "anchor" in r else "x")
    assert not (tmp_path / "queue" / "t684-cc").exists()


# ---- the payload profile: raw on the twin, coded on the served collection ---------------

def test_every_written_basis_has_a_code_on_the_served_profile_and_stays_raw_on_the_twin():
    served = PP.Projector.for_collection("aw_e9")
    twin = PP.Projector.for_collection("aw_e8")
    for basis in sorted(WRITTEN_BASES):
        p = {"search_ok": True, "train_ok": True, "consent_basis": basis, "consent_checked_at": AS_OF}
        assert twin.project(p)["consent_basis"] == basis
        coded = served.project(p)["consent_basis"]
        assert isinstance(coded, int), f"{basis!r} has no code in data/payload-enum-tables.v1.json"
        assert PP.decode_payload(served.project(p), "normalized", served.tables)["consent_basis"] == basis


def test_default_cache_dir_is_the_resolvers_cache_dir(monkeypatch):
    import cytir.consent.resolver as R
    assert _REAL_DEFAULT_CACHE_DIR() == R.CACHE_DIR


# ---- r6: sealed per-host probe ------------------------------------------------------------

from cytir.consent import derive as D  # noqa: E402


def _eric_row(url="https://files.eric.ed.gov/fulltext/ED1.pdf"):
    return {"url": url, "text": "x"}


def _derive(ctx, row=None, family="eric_official_pdf"):
    return D.derive_consent_flags(row or _eric_row(), ctx, family_id=family)


@pytest.mark.usefixtures("sealed_eric_probe")
def test_r6_stamps_the_sealed_record_and_its_own_checked_at():
    d = _derive(_ctx())
    assert (d.rule, d.search_ok, d.train_ok, d.consent_basis, d.consent_checked_at) == (
        "r6-contract-sealed-probe", True, True, "contract-sealed-probe", SEALED_ERIC_AT)


def test_r6_refuses_a_host_absent_from_the_sealed_artifact():
    d = _derive(_ctx(), _eric_row("https://elsewhere.example/a.pdf"))
    assert d.klass == "D" and "host-not-in-sealed-artifact" in d.rule


def test_r6_refuses_a_cache_record_of_any_other_status():
    for status in ("unknown", "error", "held", None):
        cache = {"files.eric.ed.gov": {"cc": {"status": status, "search_ok": None, "train_ok": None,
                                                "checked_at": None}}}
        d = _derive(_ctx(cache))
        assert d.klass == "D" and d.rule.startswith("D:held-cache-record-not-usable"), (status, d.rule)


@pytest.mark.usefixtures("sealed_eric_probe")
def test_r6_refuses_a_sealed_record_older_than_30_days():
    ctx = _ctx()
    ctx.as_of = "2026-12-08T05:14:57Z"                 # 30 d + 1 s after the sealed probe
    assert "older-than-30d" in _derive(ctx).rule
    ctx.as_of = "2026-12-08T05:14:56Z"                 # exactly 30 d: still evidence
    assert _derive(ctx).rule == "r6-contract-sealed-probe"


@pytest.mark.usefixtures("sealed_eric_probe")
def test_r6_needs_sealed_flags_equal_to_the_contract_and_a_usable_record():
    for k, v in (("train_ok", False), ("search_ok", False), ("status", "held"), ("checked_at", None)):
        ctx = _ctx()
        ctx.sealed_probes = json.loads(json.dumps(ctx.sealed_probes))
        ctx.sealed_probes["eric_official_pdf"]["files.eric.ed.gov"][k] = v
        d = _derive(ctx)
        assert d.klass == "D" and "r6:" in d.rule, (k, d.rule)


@pytest.mark.usefixtures("sealed_eric_probe")
def test_r6_never_trusts_a_pointer_string():
    fams = json.loads(json.dumps(_ctx().families))
    for f in fams:
        if f["family_id"] == "eric_official_pdf":
            f["consent"]["evidence"] = ["tbd"]
    assert D.load_sealed_probes(fams, str(ST.REPO)).get("eric_official_pdf") is None
    fams2 = json.loads(json.dumps(_ctx().families))
    for f in fams2:
        if f["family_id"] == "eric_official_pdf":
            f["consent"]["evidence"] = ["records/no-such-file.json#host_consent_resolution"]
    with pytest.raises(D.SealedEvidenceRefused):   # a named file that is gone is refused, not empty
        D.load_sealed_probes(fams2, str(ST.REPO))
    ctx = _ctx()
    ctx.sealed_probes = None
    assert _derive(ctx).klass == "D"


def test_r6_never_serves_a_dump_route_family_or_a_non_declared_one():
    ctx = _ctx()
    ctx.sealed_probes = {"openalex_academic_dump": {"openalex.org": {
        "status": "ok", "search_ok": True, "train_ok": True, "checked_at": SEALED_ERIC_AT}}}
    # no CC0 licence -> r1 refuses; r6 must not catch it
    d = D.derive_consent_flags({"url": "https://openalex.org/W1", "license": "none"}, ctx,
                               family_id="openalex_academic_dump")
    assert d.klass == "D" and d.rule.startswith("D:instrument-licence-absent")
    fam = dict(ctx.by_id["eric_official_pdf"], route="bulk-dump")
    assert D._sealed_probe(ctx, "eric_official_pdf", fam, "files.eric.ed.gov")[0] is None
    fam = dict(ctx.by_id["eric_official_pdf"], status="backfilled")
    assert D._sealed_probe(ctx, "eric_official_pdf", fam, "files.eric.ed.gov")[0] is None


@pytest.mark.usefixtures("sealed_eric_probe")
def test_r6_artifact_eligible_families_are_named_by_the_loaded_artifacts():
    """The ARTIFACT-ELIGIBLE ceiling (not reach: a family takes r6 only while its host has no cache record, and only two do today). The families r6 can stamp at all: a declared live-per-origin contract, a host in
    its sealed artifact with the contract's flags, not dump-route. Asserted so the reach
    cannot grow unnoticed."""
    ctx = _ctx()
    reach = sorted({fid for fid, hosts in ctx.sealed_probes.items() for h in hosts
                    if D._sealed_probe(ctx, fid, ctx.by_id[fid], h)[0] is not None})
    assert reach == ELIGIBLE, reach


ELIGIBLE = ["context7_pinned_docs", "eric_official_pdf", "github_agent_resources", "rfc_editor_net_new",
         "statcan_wds", "technical_manufacturer_catalog"]


def test_sealed_age_bound_is_the_resolvers_ttl():
    import cytir.consent.resolver as R
    assert D.SEALED_MAX_AGE_DAYS == R.MAX_AGE_DAYS


def test_a_refusing_pack_is_refused_before_the_gpu_embed(tmp_path, monkeypatch):
    rows = _no_flags("cc_authority_frontier_w7")
    _write(tmp_path / "chunks", rows)
    called = []
    monkeypatch.setattr(compose, "embed_stage", lambda *a, **k: called.append(1) or 0)
    with pytest.raises(ST.ConsentStampRefused):
        compose.embed_compose_enqueue(
            pack_id="t684-pre", family="cc_authority_frontier_w7", chunks_dir=str(tmp_path / "chunks"),
            vectors_dir=str(tmp_path / "vectors"), collection="aw_e8", mrl=256,
            queue_dir=str(tmp_path / "queue"), consent_ctx=_ctx(), log=lambda m: None)
    assert called == [] and not (tmp_path / "queue").exists()
    assert not (tmp_path / "chunks-consent-stamped").exists()      # the dry run wrote nothing


def test_a_backfill_resume_under_other_rules_is_refused_before_any_qdrant_call():
    from cytir.consent import apply as A

    def api(*a, **k):
        raise AssertionError("no call may happen before the rules_version check")
    with pytest.raises(RuntimeError, match="REFUSED: saved state was written under rules"):
        A.apply_twin(api, "aw_e8", _ctx(), dry=True, state={"rules_version": "adr115.v1", "offset": 5})


def test_drain_census_counts_rows_without_flags_and_never_refuses():
    from cytir.index import promote_queue as PQ
    pay = [{"search_ok": True, "train_ok": False}, {}, {"search_ok": True}, {"train_ok": None}]
    assert PQ.consent_flag_census(iter(pay)) == {
        "rows": 4, "missing_both_flags": 2, "missing_any_flag": 3}


def test_memo_key_separates_two_hosts_of_one_family(tmp_path):
    """Two unflagged crawled rows, same licence view, different hosts: one host has a cache
    verdict, the other none. A key that dropped the host would stamp both or refuse both."""
    a, b = (dict(r) for r in _no_flags("cc_authority_frontier_w7"))
    a.update(domain="probed.example", url="https://probed.example/x")
    b.update(domain="unprobed.example", url="https://unprobed.example/y")
    cache = {"probed.example": {"cc": {"status": "ok", "search_ok": True, "train_ok": True,
                                        "checked_at": "2026-09-26T00:00:00Z"}}}
    _refuses(tmp_path, "cc_authority_frontier_w7", [a, b], "1 of 2", ctx=_ctx(cache))


def test_newest_usable_variant_wins_not_a_fixed_variant_order():
    # real build-box case: github.com cc=2026-09-08 train_ok=True, direct=2026-09-28 train_ok=False
    idx = {"github.com": {
        "cc": {"status": "ok", "search_ok": True, "train_ok": True, "checked_at": "2026-09-08T10:00:00Z"},
        "direct": {"status": "ok", "search_ok": True, "train_ok": False, "checked_at": "2026-09-28T10:00:00Z"}}}
    rec, label = D.cache_lookup(idx, "github.com")
    assert rec["train_ok"] is False and label == "exact:direct"
    # the newer variant being unusable, the older one still resolves
    idx["github.com"]["direct"]["status"] = "unknown"
    rec, label = D.cache_lookup(idx, "github.com")
    assert rec["train_ok"] is True and label == "exact:cc"
    # equal timestamps: deterministic, cc first
    t = "2026-09-28T10:00:00Z"
    idx = {"h.org": {"direct": {"status": "ok", "search_ok": True, "train_ok": False, "checked_at": t},
                     "cc": {"status": "ok", "search_ok": True, "train_ok": True, "checked_at": t}}}
    assert D.cache_lookup(idx, "h.org")[1] == "exact:cc"
