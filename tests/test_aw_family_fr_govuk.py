"""Task 699: the Federal Register leg of `us_federal_register_ecfr` and `govuk_content`, on REAL API
bytes (fixtures copied from federalregister.gov and www.gov.uk on 2026-10-03), through
`cyt source run`'s own door under the contract's declared chunk policy. The network is an
`httpx.MockTransport` over the same bytes.

Pins: the request pacing, the retry and the refusals (a truncated list, a 0-record run); one
record per kept document with its citation / canonical URL / licence; the GPO header and footer
and every tag are gone; short pages and non-content items are counted, never dropped silently;
the document ledger balances; the windows driver is resumable and refuses a run that stages 0 rows.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import httpx
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import api_fetch as AF  # noqa: E402

FR = HERE / "fixtures" / "us_fr"
GU = HERE / "fixtures" / "govuk"
TODAY = "2026-10-03"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-10-03T00:00:00Z"}


@pytest.fixture(autouse=True)
def _declared_policy(monkeypatch):
    from cytir.core import model_pin
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256",
                       hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest())
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", model_pin.EMBED_MODEL_REVISION)
    monkeypatch.setattr("time.sleep", lambda s: None)
    yield
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


def _run(family, tmp_path, http, **kw):
    from cytir.sources import run_family
    return run_family.run_family(
        family, out_dir=tmp_path / "o", today=TODAY, consent=lambda h, d: {**OK, "host": h},
        http=http, log=lambda m: None, **kw)


def _rows(tmp_path):
    return [json.loads(x) for x in
            (tmp_path / "o" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- pacing, windows

def test_pacer_keeps_the_mean_rate_under_the_cap():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s
    pacer = AF.Pacer(4.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(9):
        pacer.wait()
    assert t[0] == pytest.approx(8 * 0.25)            # 9 requests, 8 gaps of 1/4 s


def test_get_retries_a_503_then_returns_and_raises_after_the_retries():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503 if len(calls) < 3 else 200, json={"ok": True})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert AF.get(client, "https://x.example/a", AF.Pacer(1000), sleep=lambda s: None).status_code == 200
    assert len(calls) == 3
    always = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(RuntimeError, match="failed after"):
        AF.get(always, "https://x.example/a", AF.Pacer(1000), sleep=lambda s: None)


def test_a_404_is_returned_to_the_caller_not_retried():
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: calls.append(1) or httpx.Response(404)))
    assert AF.get(client, "https://x.example/a", AF.Pacer(1000)).status_code == 404
    assert len(calls) == 1


def test_month_bins_and_windows_cover_the_range_without_gap_or_overlap():
    import datetime as dt
    bins = AF.month_bins(dt.date(2023, 11, 15), dt.date(2024, 2, 10))
    assert bins == [("2023-11-15", "2023-11-30"), ("2023-12-01", "2023-12-31"),
                    ("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-10")]
    wins = AF.pack_windows([(a, b, n) for (a, b), n in zip(bins, (0, 60, 50, 5))], cap=100)
    assert [(w["from"], w["to"], w["docs"]) for w in wins] == [
        ("2023-11-15", "2023-12-31", 60), ("2024-01-01", "2024-02-10", 55)]
    assert wins[0]["key"].startswith("w000-")
    big = AF.pack_windows([("2024-01-01", "2024-01-31", 500)], cap=100)
    assert len(big) == 1 and big[0]["docs"] == 500            # one bin is never split


# ----------------------------------------------------------------------------- Federal Register

PICKED = json.loads((FR / "picked.json").read_text())
RAW = {d["document_number"]: (FR / f"raw-{d['document_number']}.txt").read_bytes() for d in PICKED}


def _fr_client(results=None, count=None, text_status=None):
    results = PICKED if results is None else results

    def handler(req: httpx.Request):
        assert req.url.host == "www.federalregister.gov"
        if req.url.path.endswith("/documents.json"):
            return httpx.Response(200, json={"count": len(results) if count is None else count,
                                             "results": results})
        num = req.url.path.rsplit("/", 1)[-1].removesuffix(".txt")
        if text_status:
            return httpx.Response(text_status)
        return httpx.Response(200, content=RAW[num], headers={"content-type": "text/plain"})
    return httpx.Client(transport=httpx.MockTransport(handler))


FR_OPTS = {"leg": "fr", "from": "2024-03-01", "to": "2024-03-31", "max_req_s": "1000",
           "bulk_dates": "none"}   # a replay: 2024 is the bulk route's date unless declared otherwise


def test_fr_one_record_set_per_document_and_the_ledger_balances(tmp_path, word_only_chunking_allowed):
    m = _run("us_federal_register_ecfr", tmp_path, _fr_client(), options=FR_OPTS)
    rows = _rows(tmp_path)
    d = m["discover"]["documents"]
    assert d["seen"] == len(PICKED) == d["admitted"] + d.get("body_under_word_floor", 0)
    assert d["admitted"] >= 4 and m["terminals"]["units_in"] == 1
    urls = {r["url"] for r in rows}
    assert urls <= {p["html_url"] for p in PICKED} and len(urls) == d["admitted"]
    for r in rows:
        assert r["domain"] == "federalregister.gov" and r["license"] == "public-domain"
        assert r["tier"] == "A" and r["lang"] == "en" and r["last_seen"] == TODAY
        assert r["point_id"] and "<" not in r["text"] and "BILLING CODE" not in r["text"]
        assert "[FR Doc" not in r["text"] and "From the Federal Register Online" not in r["text"]
    first = " ".join(r["text"] for r in rows if r["url"] == PICKED[0]["html_url"])
    assert f"Federal Register {PICKED[0]['citation']}" in first and "published 2024-03-04" in first


def test_fr_header_and_footer_are_stripped_and_lines_are_joined():
    from cytir.sources.families.federal_register import raw_text_to_body
    body = raw_text_to_body(RAW["2024-04436"])
    assert body.startswith("DEPARTMENT OF JUSTICE")
    assert "Agency Information Collection Activities; Proposed eCollection eComments" in body
    assert "\n" not in body.split("\n\n")[2]                 # a hard-wrapped paragraph is one line
    assert not body.rstrip().endswith("BILLING CODE 4410-FX-P")


def test_fr_the_printed_page_banner_and_running_heads_are_dropped():
    from cytir.sources.families.federal_register import raw_text_to_body
    body = raw_text_to_body(RAW["2024-03008"])
    assert body.startswith("DEPARTMENT OF TRANSPORTATION")
    assert "This section of the FEDERAL REGISTER contains notices" not in body
    assert "Federal Register / Vol." not in body and "[[Page" not in body
    assert "SUMMARY: The FAA proposes to adopt a new airworthiness directive" in body


def test_fr_a_missing_text_is_counted_and_a_run_with_no_records_refuses(tmp_path):
    with pytest.raises(Exception) as exc:
        _run("us_federal_register_ecfr", tmp_path, _fr_client(text_status=404), options=FR_OPTS)
    assert "month_without_documents" in str(exc.value) or "0 records" in str(exc.value)


def test_fr_a_list_at_the_api_cap_refuses_instead_of_truncating(tmp_path):
    with pytest.raises(Exception) as exc:
        _run("us_federal_register_ecfr", tmp_path, _fr_client(count=10000), options=FR_OPTS)
    assert "cap" in str(exc.value) or "failed" in str(exc.value)


def test_fr_a_list_shorter_than_its_count_refuses(tmp_path):
    with pytest.raises(Exception):
        _run("us_federal_register_ecfr", tmp_path, _fr_client(count=len(PICKED) + 3), options=FR_OPTS)


def test_fr_plan_windows_counts_each_month_once(tmp_path):
    from cytir.sources import registry
    from cytir.sources.family import RunContext
    seen = []

    def handler(req):
        seen.append(req.url.params["conditions[publication_date][gte]"])
        return httpx.Response(200, json={"count": 40, "results": []})
    adapter, _row, spec = registry.admit("us_federal_register_ecfr", strict=False)
    ctx = RunContext(out_dir=tmp_path, today=TODAY, chunk_spec=spec,
                     options={"leg": "fr", "from": "2024-01-01", "to": "2024-03-31",
                              "max_req_s": "1000"},
                     http=httpx.Client(transport=httpx.MockTransport(handler)))
    wins = adapter.plan_windows(ctx, cap=100)
    assert seen == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert [(w["from"], w["to"], w["docs"]) for w in wins] == [
        ("2024-01-01", "2024-02-29", 80), ("2024-03-01", "2024-03-31", 40)]


# ----------------------------------------------------------------------------- GOV.UK

def _gu(name):
    return json.loads((GU / f"{name}.json").read_text())


ITEMS = {i["base_path"]: i for i in (_gu(n) for n in (
    "fmt-guide", "fmt-answer", "fmt-hmrc_manual_section", "fmt-detailed_guide", "guidance",
    "fatality_notice", "research_for_development_output", "decision", "vat-registration"))}


def _gu_client(total=None, links=None, missing=()):
    links = list(ITEMS) if links is None else links

    def handler(req: httpx.Request):
        assert req.url.host == "www.gov.uk"
        if req.url.path == "/api/search.json":
            q = req.url.params
            start, count = int(q.get("start", 0)), int(q.get("count", 10))
            return httpx.Response(200, json={
                "total": len(links) if total is None else total,
                "results": [{"link": x} for x in links[start:start + count]]})
        path = req.url.path.removeprefix("/api/content")
        if path in missing:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=ITEMS[path])
    return httpx.Client(transport=httpx.MockTransport(handler))


GU_OPTS = {"from": "2024-01-01", "to": "2024-01-31", "max_req_s": "1000"}


def test_govuk_composes_items_and_counts_every_exclusion(tmp_path, word_only_chunking_allowed):
    m = _run("govuk_content", tmp_path, _gu_client(), options=GU_OPTS)
    rows = _rows(tmp_path)
    d = m["discover"]["documents"]
    excluded = sum(v for k, v in d.items() if k not in ("seen", "admitted"))
    assert d["seen"] == len(ITEMS) == d["admitted"] + excluded
    assert d["schema_redirect"] == 1 and d.get("body_under_word_floor", 0) >= 1
    assert d["admitted"] >= 4
    for r in rows:
        assert r["domain"] == "www.gov.uk" and r["license"] == "OGL-UK-3.0"
        assert r["url"].startswith("https://www.gov.uk/") and "<" not in r["text"]
        assert r["tier"] == "A" and r["source_type"] == "official" and r["point_id"]
    guide = " ".join(r["text"] for r in rows if r["url"].endswith("/change-address-driving-licence"))
    assert guide.startswith("Change the address on your driving licence")
    assert all(part["title"] in guide for part in ITEMS["/change-address-driving-licence"]["details"]["parts"])
    assert "Open Government Licence v3.0" in guide


@pytest.mark.parametrize("family,opts", [("govuk_content", GU_OPTS),
                                         ("us_federal_register_ecfr", FR_OPTS)])
def test_a_run_without_the_declared_chunk_policy_is_refused(tmp_path, monkeypatch, family, opts):
    from cytir.sources.registry import FamilyRefused
    monkeypatch.delenv("AW_CHUNKER")
    monkeypatch.delenv("AW_CHUNK_POLICY")
    monkeypatch.delenv("AW_CHUNK_POLICY_SHA256")
    http = _gu_client() if family == "govuk_content" else _fr_client()
    with pytest.raises(FamilyRefused, match="chunk policy"):
        _run(family, tmp_path, http, options=opts)


def test_govuk_a_listing_shorter_than_its_total_refuses(tmp_path):
    with pytest.raises(Exception):
        _run("govuk_content", tmp_path, _gu_client(total=len(ITEMS) + 5), options=GU_OPTS)


def test_govuk_a_404_content_item_is_counted_not_fatal(tmp_path, word_only_chunking_allowed):
    m = _run("govuk_content", tmp_path, _gu_client(missing=("/change-address-driving-licence",)),
             options=GU_OPTS)
    assert m["discover"]["documents"]["content_404"] == 1


def test_govuk_links_off_gov_uk_are_counted_and_not_fetched(tmp_path, word_only_chunking_allowed):
    m = _run("govuk_content", tmp_path,
             _gu_client(links=list(ITEMS) + ["https://example.org/x"]), options=GU_OPTS)
    assert m["discover"]["documents"]["link_not_on_gov_uk"] == 1


def test_govuk_before_2001_is_one_unit_then_months():
    from cytir.sources import registry
    from cytir.sources.family import RunContext
    adapter, _row, spec = registry.admit("govuk_content", strict=False)
    ctx = RunContext(out_dir=pathlib.Path("."), today="2001-03-15", chunk_spec=spec,
                     options={"from": "1900-01-01", "to": "2001-03-15"})
    units = adapter.discover(ctx)
    assert [u.key for u in units] == ["before-2001", "govuk-2001-01", "govuk-2001-02",
                                      "govuk-2001-03"]
    assert units[0].meta == {"gte": None, "lte": "2000-12-31"}
    assert units[-1].meta["lte"] == "2001-03-15"


def test_html_to_markdown_headings_lists_tables_and_no_tags():
    from cytir.sources.families.govuk_content import html_to_markdown
    md = html_to_markdown(
        '<div class="govspeak"><h2 id="a">Basic</h2><p>One <a href="/x">link</a>.</p>'
        "<ul><li>first</li><li>second<ul><li>nested</li></ul></li></ul>"
        "<table><tr><th>Day</th><th>Ties</th></tr><tr><td>16</td><td>4</td></tr></table>"
        '<img src="x.jpg"><script>alert(1)</script></div>')
    assert md.split("\n\n") == ["### Basic", "One link.", "- first", "- second", "- nested",
                                "Day | Ties", "16 | 4"]


# ----------------------------------------------------------------------------- windows driver

def _fake_pack(root, key, rows):
    pack = root / key / "pack"
    pack.mkdir(parents=True)
    (pack / "manifest.json").write_text(json.dumps({
        "records": {"count": rows}, "discover": {"documents": {"seen": rows}},
        "terminals": {"units_in": 1, "units_quarantined": 0, "quarantined_reasons": {}},
        "units": [{"unit": key, "terminal": "admitted"}]}))


def test_run_windows_is_resumable_and_refuses_a_run_that_stages_nothing(tmp_path, monkeypatch):
    from cytir.sources import family_stage as FS
    from cytir.sources import run_family, run_windows as RW
    wins = [{"key": "w000-a", "from": "2024-01-01", "to": "2024-01-31", "docs": 5},
            {"key": "w001-b", "from": "2024-02-01", "to": "2024-02-29", "docs": 5}]
    (tmp_path / "WINDOWS.json").write_text(json.dumps(
        {"family": "x", "options": {"leg": "fr"}, "windows": wins}))
    ran, staged = [], []

    def fake_run(family_id, *, out_dir, options, **kw):
        ran.append((out_dir.name, options["from"]))
        _fake_pack(out_dir.parent, out_dir.name, 3)
    monkeypatch.setattr(run_family, "run_family", fake_run)
    monkeypatch.setattr(FS, "stage_family_pack",
                        lambda pack, **kw: staged.append(pack.parent.name) or [{"pack_id": "p"}])
    monkeypatch.setattr(FS, "list_stages", lambda *a, **k: [])
    kw = dict(root=tmp_path, corpus_root=tmp_path / "c", options={"leg": "fr"}, today=TODAY,
              cap=10, twin="t", prod="p")
    st = RW.run_all("us_federal_register_ecfr", max_windows=1, **kw)
    assert st["status"] == "partial" and ran == [("w001-b", "2024-02-01")]   # newest first
    st = RW.run_all("us_federal_register_ecfr", **kw)
    assert st["status"] == "complete" and st["rows_staged"] == 6
    assert [r[0] for r in ran] == ["w001-b", "w000-a"]           # w001-b was NOT fetched again
    assert (tmp_path / "w000-a" / "DONE.json").is_file() and (tmp_path / "NEXT.md").is_file()
    assert "python -m cytir.sources.run_windows" in (tmp_path / "NEXT.md").read_text()


def test_run_windows_refuses_when_every_window_is_empty(tmp_path, monkeypatch):
    from cytir.sources import run_family, run_windows as RW
    (tmp_path / "WINDOWS.json").write_text(json.dumps(
        {"family": "x", "options": {}, "windows": [
            {"key": "w000-a", "from": "2024-01-01", "to": "2024-01-31", "docs": 5}]}))

    def empty(family_id, **kw):
        raise run_family.RunError("x: 1 unit(s) read, 0 records admitted ({}); nothing composed")
    monkeypatch.setattr(run_family, "run_family", empty)
    with pytest.raises(RW.WindowsError, match="0 rows staged"):
        RW.run_all("us_federal_register_ecfr", root=tmp_path, corpus_root=tmp_path / "c",
                   options={}, today=TODAY, cap=10, twin="t", prod="p")


def test_run_windows_will_not_change_a_plan_under_a_running_fetch(tmp_path):
    from cytir.sources import run_windows as RW
    (tmp_path / "WINDOWS.json").write_text(json.dumps(
        {"family": "x", "options": {"leg": "fr"}, "windows": [{"key": "w000"}]}))
    with pytest.raises(RW.WindowsError, match="planned with options"):
        RW.plan("us_federal_register_ecfr", tmp_path, {"leg": "ecfr"}, TODAY, 10)


_HELD_FIXTURES = ["held-" + t + ".json" for t in (
    "traffic_commissioner_regulatory_decision", "utaac_decision", "tax_tribunal_decision",
    "employment_appeal_tribunal_decision", "sfo_case", "residential_property_tribunal_decision",
    "asylum_support_decision", "decision")] + ["fatality_notice.json"]


@pytest.mark.parametrize("name", _HELD_FIXTURES)
def test_govuk_types_that_name_private_persons_are_held_counted_and_never_staged(name):
    """Tim's ruling 24: real items (w039 captures / the Content API, 2026-10-03; long strings trimmed)
    of every held type are held at extract with a named reason and yield no document."""
    from cytir.sources.families import govuk_content as G
    item = json.loads((GU / name).read_text())
    dtype = item["document_type"]
    assert dtype in G.HELD_PERSONAL_DATA_TYPES

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}

    cap = type("Cap", (), {"body": json.dumps(item).encode()})()
    assert G.GovukContentFamily().extract(cap, Ctx) == []
    assert Ctx.notes["documents"] == {"seen": 0, "admitted": 0, f"held_personal_data_{dtype}": 1}


def test_govuk_an_ordinary_guidance_page_is_still_admitted():
    from cytir.sources.families import govuk_content as G
    item = json.loads((GU / "fmt-detailed_guide.json").read_text())
    assert item["document_type"] not in G.HELD_PERSONAL_DATA_TYPES

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}

    cap = type("Cap", (), {"body": json.dumps(item).encode()})()
    docs = G.GovukContentFamily().extract(cap, Ctx)
    assert len(docs) == 1 and Ctx.notes["documents"]["admitted"] == 1


def test_govuk_employment_tribunal_link_stubs_are_held_at_extract():
    """Real items of 2026-10-02. A multi-attachment decision (its body = several 'Read the full
    decision in ...' lines) passes the 30-word floor; the declared type rule still holds it."""
    from cytir.sources.families import govuk_content as G
    a = json.loads((GU / "govuk_et_stub_stares.json").read_text())
    b = json.loads((GU / "govuk_et_stub_dumitrescu.json").read_text())
    both = json.loads(json.dumps(a))
    both["details"]["body"] = a["details"]["body"] + b["details"]["body"]
    assert len(G.compose_body(both["details"]).split()) >= G.MIN_BODY_WORDS  # the floor alone lets it in
    assert a["document_type"] == "employment_tribunal_decision"

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}

    fam = G.GovukContentFamily()
    for item in (a, b, both):
        cap = type("Cap", (), {"body": json.dumps(item).encode()})()
        assert fam.extract(cap, Ctx) == []
    assert Ctx.notes["documents"]["link_only_employment_tribunal_decision"] == 3
    assert Ctx.notes["documents"]["admitted"] == 0


# ----------------------------------------------------------------------------- D2: positive list

_LISTED = json.loads((GU / "listed-types.json").read_text())


def _extract(item):
    from cytir.sources.families import govuk_content as G

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}

    cap = type("Cap", (), {"body": json.dumps(item).encode()})()
    return G.GovukContentFamily().extract(cap, Ctx), Ctx.notes["documents"]


def test_govuk_the_positive_list_is_the_53_listed_types_and_no_held_type():
    from cytir.sources.families import govuk_content as G
    assert len(G.ADMITTED_DOCUMENT_TYPES) == 53
    assert not (G.ADMITTED_DOCUMENT_TYPES & (G.HELD_PERSONAL_DATA_TYPES | G.LINK_ONLY_DOCUMENT_TYPES))
    assert set(_LISTED["types"]) == G.ADMITTED_DOCUMENT_TYPES
    assert {i["document_type"] for i in _LISTED["items"]} == G.ADMITTED_DOCUMENT_TYPES


@pytest.mark.parametrize("item", _LISTED["items"], ids=lambda i: i["document_type"])
def test_govuk_every_listed_type_is_never_held_for_its_type(item):
    """A real Content API item of each listed type (2026-10-03, strings trimmed): never held by type
    (it may still fall under the word floor, which is a different, counted reason)."""
    docs, notes = _extract(item)
    held = [k for k in notes if k.startswith(("held_", "link_only_"))]
    assert held == [], held
    assert len(docs) == notes["admitted"] <= 1


def test_govuk_most_listed_types_admit_a_real_item():
    admitted = sum(len(_extract(i)[0]) for i in _LISTED["items"])
    assert admitted >= 40, admitted    # the rest fall under the word floor (a different, counted reason)


def test_govuk_an_unlisted_or_new_type_is_held_by_default_and_counted():
    item = json.loads(json.dumps(_LISTED["items"][0]))
    for dtype in ("taxon", "a_type_created_next_year", ""):
        item["document_type"] = dtype
        docs, notes = _extract(item)
        assert docs == []
        assert notes == {"seen": 0, "admitted": 0, f"held_unlisted_type_{dtype or 'none'}": 1}


# ----------------------------------------------------------------------------- D3: FR debarment

_DEB = json.loads((FR / "debarment-notices.json").read_text())


def _fr_extract(num):
    from cytir.sources.families.federal_register import FederalRegisterLeg

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}
        options = {}

    cap = type("Cap", (), {"meta": {"doc": _DEB[num]["doc"]}, "body": _DEB[num]["raw"].encode()})()
    return FederalRegisterLeg(None).extract(cap, Ctx), Ctx.notes["documents"]


@pytest.mark.parametrize("num", ["2026-18581", "2026-06399", "2026-07067", "2026-07864"])
def test_fr_debarment_notices_are_held_at_extract_and_counted(num):
    """Real notices (w116/w117): FCC E-Rate debarment (names a party), FCC suspension, FDA final order
    (the person is the title)."""
    docs, notes = _fr_extract(num)
    assert docs == [] and notes == {"seen": 0, "admitted": 0, "held_personal_data_debarment_notice": 1}


def test_fr_a_rule_about_debarment_rules_is_still_admitted():
    docs, notes = _fr_extract("2026-06864")
    assert _DEB["2026-06864"]["doc"]["type"] == "Rule" and "Debarment" in _DEB["2026-06864"]["doc"]["title"]
    assert len(docs) == 1 and notes["admitted"] == 1


# ----------------------------------------------------------------------------- named-person orders (19:15Z)

_CLASSES = json.loads((FR / "held-classes.json").read_text())


@pytest.mark.parametrize("k", ["dea", "bis", "ofac"])
def test_fr_named_person_order_classes_hold_every_hit_and_no_near_miss(k):
    """Real notices of the agency (API, 2026-10-03): >= 30 hits held with the class reason, >= 30
    near-misses of the same agency admitted (applications, quotas, advisory committees, tariffs,
    information collections, State notices of other kinds)."""
    from cytir.sources.families.federal_register import held_notice_reason
    c = _CLASSES[k]
    assert len(c["hits"]) >= 20 and len(c["near_misses"]) >= 20
    for d in c["hits"]:
        assert held_notice_reason(d["type"], d["title"], d["agencies"]) == c["reason"], d["title"]
        assert held_notice_reason("Rule", d["title"], d["agencies"]) is None      # only Notices
    for d in c["near_misses"]:
        assert held_notice_reason(d["type"], d["title"], d["agencies"]) is None, d["title"]
# ----------------------------------------------------------------------------- D1: the route split

def in_bulk_all(docs):
    return [x for x in docs if "2000-01-01" <= x["publication_date"] <= "2025-11-30"]


def test_fr_api_leg_refuses_a_date_the_bulk_route_owns_and_never_fetches_its_text(tmp_path):
    """Declared default split (bulk 2000-01-01..2025-11-30): the 2024-03 documents are the bulk
    route's; the API leg counts `route_date_bulk` and makes no text request (a run with no records
    refuses, as a run that stages nothing must)."""
    from cytir.sources import run_family
    calls = []
    inner = _fr_client()._transport.handler

    def spy(req):
        calls.append(req.url.path)
        return inner(req)
    client = httpx.Client(transport=httpx.MockTransport(spy))
    opts = {k: v for k, v in FR_OPTS.items() if k != "bulk_dates"}
    try:
        m = _run("us_federal_register_ecfr", tmp_path, client, options=opts)
    except run_family.RunError:
        m = None                      # a run that admits nothing may refuse; the ledger is below
    if m is not None:
        d = m["discover"]["documents"]
        in_bulk = [x for x in PICKED if "2000-01-01" <= x["publication_date"] <= "2025-11-30"]
        assert d["route_date_bulk"] == len(in_bulk) < d["seen"] == len(PICKED)   # the 1996 notice is the API's
    texts = [c for c in calls if c.endswith(".txt")]
    assert len(texts) == len(PICKED) - len(in_bulk_all(PICKED)) == 1      # only the 1996 notice
    assert [c for c in calls if c.endswith("/documents.json")]


def test_fr_api_rows_record_their_route_in_the_master_bundle(tmp_path, word_only_chunking_allowed):
    _run("us_federal_register_ecfr", tmp_path, _fr_client(), options=FR_OPTS)
    assert all(r["_master"]["route"] == "federalregister-api" for r in _rows(tmp_path))


def test_fr_an_identical_chunk_of_one_document_is_dropped_and_counted(word_only_chunking_allowed):
    """Real shape (govinfo XML 2000, 2 of 31k documents repeat a block): two identical chunks of one
    document mint one point id; a second row would fail the whole year as duplicate_point_id."""
    from cytir.sources.families.federal_register import FederalRegisterLeg
    from cytir.sources.family import Doc
    block = " ".join(f"w{i}x" for i in range(700))

    class Ctx:
        today = TODAY
        notes = {}

    doc = Doc(url="https://www.federalregister.gov/documents/2000/11/15/00-29020/x",
              text="# Title\n\n" + "\n\n".join([block, block]), title="Title", lang="en",
              extra={"body": block})
    rows = FederalRegisterLeg(None).chunk(doc, Ctx)
    assert len(rows) == 1 and len({r["point_id"] for r in rows}) == 1
    assert Ctx.notes["chunks_dropped_identical_in_document"] == 1


def test_fr_a_month_without_duplicates_stages_byte_identically_with_and_without_the_collapse(tmp_path, monkeypatch, word_only_chunking_allowed):
    """Characterization (task 759): the SAME fixture month staged twice in ONE environment, once
    with the stage-door collapse off (the pre-759 path) and once on, gives byte-identical
    `records.jsonl` and a collapsed count of 0. (A pinned sha was refused: `stamps.prov_id` hashes the ABSOLUTE
    path of the chunk-policy file, so the bytes differ per checkout directory; every other field
    is identical.)"""
    from cytir.sources import run_family
    with monkeypatch.context() as off_door:
        off_door.setattr(run_family, "collapse_identical_rows", lambda rows: (rows, []))
        _run("us_federal_register_ecfr", tmp_path / "off", _fr_client(), options=FR_OPTS)
    m = _run("us_federal_register_ecfr", tmp_path / "on", _fr_client(), options=FR_OPTS)
    off = (tmp_path / "off" / "o" / "pack" / "records.jsonl").read_bytes()
    on = (tmp_path / "on" / "o" / "pack" / "records.jsonl").read_bytes()
    assert off and off == on
    assert m["collapsed_identical_rows"] == 0 and "collapsed_detail" not in m
