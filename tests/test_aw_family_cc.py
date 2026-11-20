#!/usr/bin/env python3
"""Exemplar 3 of the source-family interface (task 683, S7): the Common Crawl route, dry run.

`cc_authority_frontier_w7` is the one family whose contract is fully DECLARED (consent basis,
licence instrument, chunk policy, terminal schema version). The fixture is a tiny SEALED
tranche, synthetic by construction (`683-scripts/build_cc_fixture.py`: eight capture
records, seven range replies; no Common Crawl byte is fetched anywhere in this file).

What the tests prove, and what they do not:

  * the framework drives the route end to end offline: records -> consent (CC mode, per host,
    BEFORE any byte) -> range read through the rate limiter -> WARC unwrap -> the one extractor
    and the one chunk dispatcher under the contract-declared policy -> a pack; every record
    reaches exactly one terminal and the table balances;
  * the adapter adds nothing to the core: the rows equal a direct `process_html` call over the
    same sealed pages under the same selection (the `verify_authority_pack_equivalence.py`
    pattern: prove the thing the port changed, here that it changed nothing);
  * the contract and the process must agree: a run whose dispatcher resolves to another chunk
    policy than the family declares is refused before a unit is read;
  * the CC politeness rule (<= 5 requests/s, ADR-014) is code, not a comment, and an error is
    never an empty.

NOT proven here: that a REAL Common Crawl WARC member parses (one range read, ~100 KB, would);
resolving records from the CC index (`cc_columnar.py`, still a pipeline module).
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIX = HERE / "fixtures" / "cc_tranche"
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAMILY = "cc_authority_frontier_w7"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
import hashlib  # noqa: E402

from cytir.core import model_pin  # noqa: E402

POLICY_SHA = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()    # the byte pin, from the file
TOKENIZER_REVISION = model_pin.EMBED_MODEL_REVISION     # arctic-embed-l-v2's pin
TODAY = "2026-09-30"
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": False,
      "checked_at": "2026-09-30T00:00:00Z"}

#: golden, pinned 2026-09-30 from the sealed tranche under the declared policy (`stamps.prov_id`
#: blanked, see `_normalised_sha`)
GOLDEN = {"records": 4, "sha256": "ef262416d181055076565fe590bcdc23b5d4e382496a17e0325d0b5a6a842e3a"}


@pytest.fixture(autouse=True)
def _declared_policy(monkeypatch):
    """Select the family's declared chunk policy the way a candidate run does."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", POLICY_SHA)
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", TOKENIZER_REVISION)
    yield
    # `chunk_markdown_channels` ACTIVATES the selected policy in `cytir.chunk.chunker`'s module
    # globals (and the tokenizer pin); left there it breaks every later test that expects the
    # import-time default (test_chunk.py went red in the full run). Put it back.
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


def _consent(h, direct):
    if h == "blocked.example":
        return {"host": h, "bucket": "declined", "status": "ok", "search_ok": False,
                "train_ok": False}
    assert direct is False, "Common Crawl provenance resolves in CC mode, never direct"
    return {**OK, "host": h}


def _run(tmp_path, **kw):
    from cytir.acquire.tape import Tape
    from cytir.sources import run_family
    tape = Tape.load(FIX / "tape.json")
    manifest = run_family.run_family(
        FAMILY, out_dir=tmp_path, today=TODAY, http=tape.client(),
        consent=kw.pop("consent", _consent),
        options=kw.pop("options", {"records": str(FIX / "records.jsonl")}),
        log=lambda m: None, **kw)
    return manifest, tape


# -- the route end to end ---------------------------------------------------------------

def test_every_record_reaches_one_terminal_and_the_table_balances(tmp_path, word_only_chunking_allowed):
    manifest, tape = _run(tmp_path)
    assert tape.unexpected == []
    t = manifest["terminals"]
    assert t["units_in"] == 8
    assert (t["units_admitted"], t["units_excluded"], t["units_quarantined"]) == (4, 4, 0)
    assert t["excluded_reasons"] == {
        "captured_http_404": 1,                         # the capture's own status
        "cc_content_truncated:length": 1,               # CC flagged it; never read
        "cc_mirror_miss": 1,                            # the mirror's 403
        "consent_declined:blocked.example": 1}          # declined BEFORE any byte was read
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]
    assert manifest["consent"]["direct_crawl"] is False


def test_the_consent_decision_precedes_the_read(tmp_path):
    """The declined host's record is never requested; the truncated one neither."""
    _m, tape = _run(tmp_path)
    ranges = [k[3] for k in tape.served]
    recs = [json.loads(line) for line in (FIX / "records.jsonl").read_text().splitlines()]
    blocked = next(r for r in recs if "blocked.example" in r["url"])
    truncated = next(r for r in recs if r["content_truncated"])
    for r in (blocked, truncated):
        assert f"bytes={r['offset']}-{r['offset'] + r['length'] - 1}" not in ranges


def test_the_manifest_carries_the_declared_contract(tmp_path):
    manifest, _ = _run(tmp_path)
    c = manifest["contract"]
    assert c["status"] == "declared" and c["unknowns"] == []
    assert c["consent_basis"].startswith("per-origin robots.txt -> AIPREF")
    assert c["chunk"]["declared"].startswith("chunk-policy-struct-md-768w")
    assert c["chunk"]["sha256"] == POLICY_SHA and c["chunk"]["registry_status"] == "current"
    assert manifest["discover"]["chunker"]["policy_sha256"] == POLICY_SHA
    assert manifest["discover"]["max_requests_per_s"] == 5.0
    assert manifest["discover"]["bytes_to_read"] == sum(
        json.loads(line)["length"] for line in (FIX / "records.jsonl").read_text().splitlines())


def test_the_seal_keeps_the_member_as_it_arrived(tmp_path):
    import hashlib
    from cytir.acquire import warc
    _run(tmp_path)
    index = [json.loads(line) for line in (tmp_path / "l0/index.jsonl").read_text().splitlines()]
    assert len(index) == 5                              # 4 pages + the captured-404 member
    for row in index:
        data = (tmp_path / "l0" / row["sha256"][:2] / row["sha256"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        rec = warc.parse_warc_record(data)              # still the gzip WARC member
        assert rec.warc_headers["warc-target-uri"] == row["url"]
        assert rec.status in (200, 404)
        assert row["meta"]["offset"] >= 0 and row["meta"]["stamps"]["tier"] == "C"


# -- the adapter adds nothing to the core ------------------------------------------------

def test_rows_equal_a_direct_process_html_call_over_the_same_pages(tmp_path, word_only_chunking_allowed):
    from cytir.acquire import warc
    from cytir.core import common as C
    manifest, _ = _run(tmp_path)
    got = [json.loads(line) for line in
           (tmp_path / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]

    pages = json.loads((HERE / "fixtures/s5_acquire_extract_chunk/pages.json")
                       .read_text(encoding="utf-8"))
    want = []
    for url in pages:                                   # the four admitted, in record order
        meta = {"content": True, "links": False, "lang_default": "en",
                "digest": None,
                "passage_common": {"domain": "example.org", "tier": "C",
                                   "source_type": "docs", "trust_tier": 0.65,
                                   "last_seen": "2026-08-15"}}
        recs = [json.loads(line) for line in (FIX / "records.jsonl").read_text().splitlines()]
        rec = next(r for r in recs if r["url"] == url)
        meta["digest"] = rec["digest"]
        passages, _links = C.process_html(url, pages[url], meta)
        for p in passages:
            row = p.payload()
            row["text"], row["point_id"] = p.text, p.point_id
            want.append(row)
    assert all(r.get("family_id") for r in got)         # ADR-124: the door stamps every row
    got = [{k: v for k, v in r.items() if k != "family_id"} for r in got]
    assert got == want
    assert len(got) == manifest["records"]["count"] >= 4
    assert {r["stamps"]["chunk"] for r in got} == {4}    # chunk.generic epoch 4 (declared)
    assert {r["last_seen"] for r in got} == {"2026-08-15"}    # the capture date, not the run's
    del warc


def _normalised_sha(pack_dir) -> str:
    """sha256 of the records with `stamps.prov_id` blanked (kept so the golden stays pinned; since
    task 771 `prov_id` is checkout-independent, see `test_prov_id_is_the_same_rule_on_any_checkout`)."""
    import hashlib
    rows = [json.loads(line) for line in
            (pack_dir / "pack/records.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["stamps"]["prov_id"] = "X"
        r.pop("family_id", None)          # ADR-124: the door's stamp is not the golden's
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_golden_records(tmp_path, word_only_chunking_allowed):
    manifest, _ = _run(tmp_path)
    assert manifest["records"]["count"] == GOLDEN["records"]
    assert _normalised_sha(tmp_path) == GOLDEN["sha256"]
    a = (tmp_path / "pack/records.jsonl").read_bytes()
    manifest2, _ = _run(tmp_path / "again")
    assert (tmp_path / "again/pack/records.jsonl").read_bytes() == a       # reproducible


# -- the contract and the process must agree --------------------------------------------

def test_a_dispatcher_that_selects_another_policy_is_refused_before_any_read(monkeypatch,
                                                                             tmp_path):
    from cytir.sources import registry
    monkeypatch.setenv("AW_CHUNKER", "legacy")
    for k in ("AW_CHUNK_POLICY", "AW_CHUNK_POLICY_SHA256", "AW_CHUNK_TOKENIZER_REVISION"):
        monkeypatch.delenv(k)
    from cytir.acquire.tape import Tape
    tape = Tape.load(FIX / "tape.json")
    with pytest.raises(registry.FamilyRefused, match="declares chunk policy"):
        from cytir.sources import run_family
        run_family.run_family(FAMILY, out_dir=tmp_path, today=TODAY, http=tape.client(),
                              consent=_consent,
                              options={"records": str(FIX / "records.jsonl")},
                              log=lambda m: None)
    assert tape.served == []                            # nothing was read
    assert not (tmp_path / "pack").exists()


def test_a_record_without_the_discovery_stamps_is_refused_at_discover(tmp_path):
    from cytir.sources.families import cc
    bad = tmp_path / "bad.jsonl"
    rec = json.loads((FIX / "records.jsonl").read_text().splitlines()[0])
    del rec["trust_tier"]
    bad.write_text(json.dumps(rec) + "\n")
    with pytest.raises(cc.RecordsError, match="trust_tier"):
        cc.load_records(bad)


# -- politeness is code; an error is not an empty ---------------------------------------

def test_the_rate_limit_is_five_per_second_and_cannot_be_raised_by_argument():
    from cytir.acquire import warc
    assert warc.MAX_REQUESTS_PER_S == 5.0
    with pytest.raises(ValueError, match="ADR-014"):
        warc.RateLimiter(6.0)
    with pytest.raises(ValueError):
        warc.RateLimiter(0)
    now = [0.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    lim = warc.RateLimiter(5.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(6):
        lim.wait()
    assert sum(slept) == pytest.approx(5 * 0.2)          # 6 calls, 5 gaps of 200 ms


def _rec():
    return {"warc_filename": "f.warc.gz", "offset": 10, "length": 5}


def _client(statuses):
    calls = []

    def handler(request):
        calls.append(request.headers["range"])
        st = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return httpx.Response(st, content=b"abc")

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_transient_errors_are_retried_then_raised_never_returned_as_empty():
    from cytir.acquire import warc
    lim = warc.RateLimiter(5.0, sleep=lambda s: None)
    c, calls = _client([429, 503, 206])
    assert warc.range_get(c, _rec(), limiter=lim, sleep=lambda s: None) == b"abc"
    assert calls == ["bytes=10-14"] * 3
    c, calls = _client([503])
    with pytest.raises(warc.WarcReadError, match="after 3 attempts"):
        warc.range_get(c, _rec(), limiter=lim, sleep=lambda s: None)
    c, _ = _client([403])
    assert warc.range_get(c, _rec(), limiter=lim, sleep=lambda s: None) is None   # a real miss


def test_a_non_member_is_an_error_not_a_body():
    import gzip
    from cytir.acquire import warc
    with pytest.raises(warc.WarcReadError):
        warc.parse_warc_record(b"not gzip")
    with pytest.raises(warc.WarcReadError, match="structure"):
        warc.parse_warc_record(gzip.compress(b"WARC/1.0\r\nWARC-Type: warcinfo\r\n\r\nx"))


# -- the dry run, through the one CLI ----------------------------------------------------

def test_dry_run_through_aw_source_run(capsys, tmp_path):
    from cytir import cli
    rc = cli.main(["source", "run", FAMILY, "--dry-run", "--skip-consent",
                   "--out-dir", str(tmp_path / "never"),
                   "--opt", f"records={FIX / 'records.jsonl'}",
                   "--opt", f"tape={FIX / 'tape.json'}"])
    out = capsys.readouterr().out
    assert rc == 0
    plan = json.loads(out[out.index("{"):])
    assert plan["dry_run"] and len(plan["units"]) == 8 and plan["consent_checked"] is False
    assert plan["discover"]["records"] == 8
    assert not (tmp_path / "never").exists()


def test_aw_source_families_lists_the_three_exemplars(capsys):
    from cytir import cli
    assert cli.main(["source", "families"]) == 0
    out = capsys.readouterr().out
    for fid in ("fedlex_exact", "scratch_elcom_docs_exact", FAMILY):
        assert fid in out


# -- the embed precondition: the consent refusal S6 left open ----------------------------
# (`embed_pack` and `--embed` are retired, slice 8 step 3; `assert_embeddable` is the
# precondition every embed door shares, and `cyt index build` will call it.)

def test_a_declared_family_pack_passes_the_embed_precondition(tmp_path):
    from cytir.sources import run_family
    manifest, _ = _run(tmp_path)
    m, recs = run_family.load_pack(tmp_path / "pack")
    gate = run_family.assert_embeddable(m, recs, log=lambda m: None)
    assert gate["rows"] == manifest["records"]["count"]


def test_embed_is_refused_for_an_unchecked_pack_and_a_declined_host(tmp_path, word_only_chunking_allowed):
    from cytir.consent import block
    from cytir.sources import run_family
    # 1. consent skipped at compose time
    _run(tmp_path / "skipped", skip_consent=True)
    m, recs = run_family.load_pack(tmp_path / "skipped/pack")
    with pytest.raises(run_family.EmbedRefused, match="consent check skipped"):
        run_family.assert_embeddable(m, recs)
    # 2. a host with a recorded blocking decision (registry, no network)
    _run(tmp_path / "live")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": "x", "hosts": {"example.org": {
        "state": "declined", "decided_on": "2026-09-30", "unblock_condition": "none"}}}))
    m, recs = run_family.load_pack(tmp_path / "live/pack")
    with pytest.raises(block.ConsentBlocked, match="EMBED REFUSED"):
        run_family.assert_embeddable(m, recs, registry_path=str(reg))


def test_prov_id_is_the_same_rule_on_any_checkout(tmp_path, monkeypatch):
    import shutil
    from cytir.core import common as C
    copy = tmp_path / "elsewhere" / POLICY_FILE.name
    copy.parent.mkdir()
    shutil.copy(POLICY_FILE, copy)
    here = C.chunker_selection().stamps()
    monkeypatch.setenv("AW_CHUNK_POLICY", str(copy))
    there = C.chunker_selection().stamps()
    assert here == there
