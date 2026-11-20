"""Task 702: the govinfo bulk leg (`leg=govinfo`) of `us_federal_register_ecfr`, on REAL bytes:
five Federal Register issues (2000-01-18, 2005-06-15, 2010-03-01, 2017-08-02, 2025-09-30), copied
from govinfo.gov/bulkdata/FR on 2026-10-03 and trimmed to 3-4 whole documents each
(`fixtures/us_fr_govinfo/`), with the federalregister.gov API listing of the same day (the real
`document_number`, `html_url`, title, agencies for those documents + one API-only document).
The network is an `httpx.MockTransport` over the same bytes. NOTHING here touches govinfo.

Pins: the route is refused until the contract names it; the XML flattens to clean paragraphs on
every era's markup; the record URL is the API leg's `html_url` for the same document; a document
the listing does not know is excluded, not given a guessed URL; the ledger balances; the overlap
guard drops a bulk row whose URL the index already serves (the point id differs between routes,
the URL does not); a dry drain plan on the fixtures admits every row.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import zipfile

import httpx
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources.families import federal_register_govinfo as GI  # noqa: E402

FX = HERE / "fixtures" / "us_fr_govinfo"
DATES = ("2000-01-18", "2005-06-15", "2010-03-01", "2017-08-02", "2025-09-30")
TODAY = "2026-10-03"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-10-03T00:00:00Z"}


def _xml(d: str) -> bytes:
    return (FX / f"fx-FR-{d}.xml").read_bytes()


def _api(d: str) -> dict:
    return json.loads((FX / f"fx-api-{d}.json").read_text())


@pytest.fixture(autouse=True)
def _declared_policy(monkeypatch):
    from cytir.core import model_pin
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest())
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", model_pin.EMBED_MODEL_REVISION)
    monkeypatch.setattr("time.sleep", lambda s: None)
    # `api_fetch` binds `sleep=time.sleep` as a keyword default at def time, so the line above never
    # reaches the pacer: patch the bound defaults (task 1982; the run slept ~0.5 s per request).
    from cytir.sources import api_fetch
    monkeypatch.setitem(api_fetch.Pacer.__init__.__kwdefaults__, "sleep", lambda s: None)
    monkeypatch.setitem(api_fetch.get.__kwdefaults__, "sleep", lambda s: None)
    yield
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


@pytest.fixture
def declared_contracts(tmp_path):
    """The real contracts file with the POLICY PR's one change applied: the route token."""
    doc = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())
    rows = doc["families"] if isinstance(doc.get("families"), list) else list(doc["families"].values())
    row = next(r for r in rows if r["family_id"] == "us_federal_register_ecfr")
    row["acquisition"]["route_allowed"] += f"; {GI.ROUTE_TOKEN} year zips"
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(doc))
    return path


def _zip(year: int, dates) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for d in dates:
            if d.startswith(str(year)):
                zf.writestr(f"{d[5:7]}/FR-{d}.xml", _xml(d))
    return buf.getvalue()


def _http(year: int, *, dates=DATES, drop_from_listing=(), calls=None, retitle=None):
    listed = [dict(r, **(retitle or {}).get(r["document_number"], {}))
              for d in dates if d.startswith(str(year)) for r in _api(d)["api_results"]
              if r["document_number"] not in drop_from_listing]

    def handler(req: httpx.Request):
        if calls is not None:
            calls.append(str(req.url))
        if req.url.host == "www.govinfo.gov":
            assert str(req.url) == GI.BULK_URL.format(year=year)
            return httpx.Response(200, content=_zip(year, dates),
                                  headers={"content-type": "application/zip"})
        q = dict(req.url.params)
        a, b = q["conditions[publication_date][gte]"], q["conditions[publication_date][lte]"]
        rows = [r for r in listed if a <= r["publication_date"] <= b]
        return httpx.Response(200, json={"count": len(rows), "results": rows})
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _run(tmp_path, year, contracts, opts=None, **kw):
    from cytir.sources import run_family
    return run_family.run_family(
        "us_federal_register_ecfr", out_dir=tmp_path / "o", today=TODAY,
        options={"leg": "govinfo", "from": f"{year}-01-01", "to": f"{year}-12-31", **(opts or {})},
        consent=lambda h, d: {**OK, "host": h}, http=_http(year, **kw), log=lambda m: None,
        contracts_path=contracts)


def _rows(tmp_path):
    return [json.loads(x) for x in
            (tmp_path / "o" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------ the route gate

def test_the_leg_refuses_until_the_contract_names_the_route():
    from cytir.sources.family import RunContext
    leg = GI.GovinfoLeg(object())
    for contract in (None, {}, {"acquisition": {"route_allowed": "federalregister.gov documents API"}}):
        ctx = RunContext(out_dir=pathlib.Path("."), today=TODAY, contract=contract,
                         options={"leg": "govinfo"})
        with pytest.raises(GI.RouteNotDeclared, match="POLICY PR"):
            leg.discover(ctx)


def test_the_bulk_tree_starts_at_2000_and_the_default_years_are_the_declared_dates():
    with pytest.raises(ValueError, match="belong to the API route"):
        GI.GovinfoLeg.years({"from": "1999-01-01", "to": "2001-12-31"}, TODAY)
    with pytest.raises(ValueError, match="belong to the API route"):
        GI.GovinfoLeg.years({"from": "2024-01-01", "to": "2026-12-31"}, TODAY)
    assert GI.GovinfoLeg.years({}, TODAY)[0] == 2000 and GI.GovinfoLeg.years({}, TODAY)[-1] == 2025
    with pytest.raises(ValueError, match="no dates"):
        GI.GovinfoLeg.years({"bulk_dates": "none"}, TODAY)


# ------------------------------------------------------------------------------ D1 (b)-(d): the split

def test_the_date_decides_the_route():
    """(b) the publication date decides: 2000-01-01..2025-11-30 is the bulk route's, every other date
    (1994-1999, 2025-12-01 on) the API route's."""
    from cytir.sources.families import federal_register as FR
    assert FR.route_for("2025-11-30", {}) == FR.ROUTE_BULK and FR.route_for("2025-12-01", {}) == FR.ROUTE_API
    assert FR.route_for("1999-12-31", {}) == FR.ROUTE_API and FR.route_for("2000-01-01", {}) == FR.ROUTE_BULK
    assert FR.route_for("2024-03-04", {"bulk_dates": "none"}) == FR.ROUTE_API


def test_the_split_is_recorded_and_a_document_dated_for_the_api_route_is_refused(tmp_path, declared_contracts, word_only_chunking_allowed):
    man = _run(tmp_path, 2025, declared_contracts, opts={"bulk_dates": "2000-01-01..2025-12-31"})
    assert man["discover"]["route_split"] == {"bulk_dates": ["2000-01-01", "2025-12-31"],
                                              "this_route": "govinfo-bulk-xml"}
    docs = man["discover"]["documents"]
    assert docs["admitted"] == docs["seen"] > 0
    leg = GI.GovinfoLeg(object())

    class Ctx:
        notes = {"documents": {"seen": 0, "admitted": 0}}
        options = {}         # the declared default: the API route owns 2025-12-01 on
    listing = {"X-1": {"document_number": "X-1", "publication_date": "2025-12-02", "type": "Notice",
                       "title": "A notice", "html_url": "https://www.federalregister.gov/documents/x",
                       "agencies": []}}
    assert leg._admit("X-1", ["word " * 40], listing, set(), Ctx) is None
    assert Ctx.notes["documents"]["route_date_api"] == 1


def test_every_bulk_row_records_its_route_in_the_master_bundle(tmp_path, declared_contracts, word_only_chunking_allowed):
    """(d) provenance per row: `_master.route` (master-only, stripped from the compact dump)."""
    _run(tmp_path, 2010, declared_contracts)
    rows = _rows(tmp_path)
    assert rows and all(r["_master"]["route"] == "govinfo-bulk-xml" for r in rows)


def test_a_debarment_notice_is_held_on_the_bulk_route_too(tmp_path, declared_contracts, word_only_chunking_allowed):
    """D3 applies on both routes: the listing's type + title decide (a person is the title)."""
    num = _api("2010-03-01")["in_xml"][0]
    man = _run(tmp_path, 2010, declared_contracts, retitle={
        num: {"type": "Notice", "title": "Jane Roe: Final Debarment Order"}})
    docs = man["discover"]["documents"]
    assert docs["held_personal_data_debarment_notice"] == 1 and docs["seen"] == docs["admitted"] + 1
    assert "Jane Roe" not in json.dumps(_rows(tmp_path))


# ------------------------------------------------------------------------------ the XML, every era

@pytest.mark.parametrize("d", DATES)
def test_every_fixture_issue_parses_to_clean_documents_with_the_listing_numbers(d):
    got = list(GI.parse_issue(_xml(d)))
    assert [n for n, _t, _p in got] == _api(d)["in_xml"]
    for number, tag, paras in got:
        text = "\n\n".join(paras)
        assert len(text.split()) >= GI.MIN_BODY_WORDS, (number, tag)
        assert "<" not in text and ">" not in text and "BILCOD" not in text, number
        assert "FR Doc." not in text.split("\n\n")[-1], "the FRDOC line is dropped"
        assert " ." not in text and " ," not in text, "inline markup leaves no stray space"


def test_a_table_flattens_to_rows():
    from lxml import etree
    el = etree.fromstring("<N><GPOTABLE><ROW><ENT>a</ENT><ENT>b <E>c</E></ENT></ROW></GPOTABLE>"
                          "<P>after</P></N>")
    assert GI.flatten(el) == ["a | b c", "after"]


# ------------------------------------------------------------------------------ the run

@pytest.mark.parametrize("year,date", [(2000, "2000-01-18"), (2005, "2005-06-15"),
                                       (2010, "2010-03-01"), (2017, "2017-08-02"),
                                       (2025, "2025-09-30")])
def test_record_urls_are_the_api_legs_html_url_for_the_same_document(tmp_path, declared_contracts,
                                                                    year, date, word_only_chunking_allowed):
    man = _run(tmp_path, year, declared_contracts)
    api = _api(date)
    want = {r["html_url"] for r in api["api_results"] if r["document_number"] in api["in_xml"]}
    rows = _rows(tmp_path)
    assert {r["url"] for r in rows} == want
    assert all(r["license"] == "public-domain" and r["domain"] == "federalregister.gov" for r in rows)
    docs = man["discover"]["documents"]
    assert docs["seen"] == docs["admitted"] == len(api["in_xml"])
    rec = man["discover"]["reconcile"]
    assert rec["xml_documents"] == len(api["in_xml"]) and rec["api_only"] == 1   # the listed extra


def test_a_document_the_listing_does_not_know_is_excluded_not_given_a_guessed_url(tmp_path, declared_contracts, word_only_chunking_allowed):
    gone = _api("2010-03-01")["in_xml"][0]
    man = _run(tmp_path, 2010, declared_contracts, drop_from_listing={gone})
    docs = man["discover"]["documents"]
    assert docs["no_api_url"] == 1 and docs["admitted"] == 2
    assert docs["seen"] == docs["admitted"] + docs["no_api_url"]       # the ledger balances
    assert gone not in json.dumps(_rows(tmp_path))


def test_the_listing_and_the_zip_are_both_sealed(tmp_path, declared_contracts, word_only_chunking_allowed):
    _run(tmp_path, 2005, declared_contracts)
    idx = [json.loads(x) for x in (tmp_path / "o" / "l0" / "index.jsonl").read_text().splitlines()]
    assert [r["unit"] for r in idx] == ["fr-list-2005", "fr-zip-2005"]


def test_only_the_declared_hosts_and_the_year_zip_are_requested(tmp_path, declared_contracts, word_only_chunking_allowed):
    calls: list[str] = []
    _run(tmp_path, 2017, declared_contracts, calls=calls)
    zips = [c for c in calls if "govinfo.gov" in c]
    assert zips == ["https://www.govinfo.gov/bulkdata/FR/2017/FR-2017.zip"]
    assert all(("govinfo.gov" in c) or ("federalregister.gov/api/v1/documents.json" in c) for c in calls)
    assert sum("documents.json" in c for c in calls) == 12          # one listing page per month


# ------------------------------------------------------------------------------ identity and overlap

def _stage(tmp_path, year, contracts):
    from cytir.sources import family_stage as FS
    _run(tmp_path, year, contracts)
    corpus = tmp_path / "corpus"
    staged = FS.stage_family_pack(tmp_path / "o" / "pack", corpus_root=corpus,
                                  family_id="us_federal_register_ecfr", log=lambda m: None)
    return FS, corpus, staged


def test_a_bulk_row_whose_url_is_already_served_is_excluded_by_the_overlap_guard(tmp_path, declared_contracts, word_only_chunking_allowed):
    """The two routes mint different point ids for one document (the id hashes the chunk text),
    but the URL is the same: the guard keys on it, so the bulk run duplicates nothing."""
    FS, corpus, _ = _stage(tmp_path, 2010, declared_contracts)
    (_n, stage), = FS.list_stages(corpus, "us_federal_register_ecfr")
    rows = _rows(tmp_path)
    served_url = rows[0]["url"]
    out = FS.check_overlap(
        stage, targets={"twin": ("http://t", "c"), "prod": ("http://p", "c")},
        exclude_served_urls=True, log=lambda m: None,
        retrieve=lambda u, c, ids: set(),                  # no id collides: the ids differ
        served_urls=lambda u, c, urls: {x for x in urls if x == served_url})
    assert out["ids_served"] == 0 and out["urls_served"] == 1
    excl = {json.loads(x)["point_id"] for x in (stage / "overlap-ids.jsonl").read_text().splitlines()}
    assert excl == {r["point_id"] for r in rows if r["url"] == served_url}
    assert out["rows_net_new"] == len(rows) - len(excl)


# ------------------------------------------------------------------------------ dry drain plan

def _zero_vectors(vectors_dir: pathlib.Path, ids: list[str], dim: int = 1024) -> None:
    """A properly shaped sealed vector shard (zeros: only shape and ids matter here)."""
    import numpy as np
    vectors_dir.mkdir(parents=True, exist_ok=True)
    np.zeros((len(ids), dim), dtype="<f2").tofile(vectors_dir / "shard-0000.vectors.f16")
    (vectors_dir / "shard-0000.row_ids.jsonl").write_text(
        "".join(json.dumps({"point_id": i}) + "\n" for i in ids))
    (vectors_dir / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": len(ids), "dim": dim,
         "dtype": "float16"}))


def test_a_dry_drain_plan_admits_every_fixture_row(tmp_path, declared_contracts, monkeypatch, word_only_chunking_allowed):
    """Stage -> compose (zero vectors: only shape and ids matter) -> `promote_queue.dry_plan`
    against a stub index that holds nothing. Read-only; nothing is embedded or written."""
    import cytir.index.compose as compose
    from cytir.index import promote_queue as PQ
    from cytir.index import promote_sealed_pack as PSP

    FS, corpus, _ = _stage(tmp_path, 2010, declared_contracts)
    (_n, stage), = FS.list_stages(corpus, "us_federal_register_ecfr")
    rows = _rows(tmp_path)
    vectors = tmp_path / "vectors"
    _zero_vectors(vectors, [r["point_id"] for r in rows])
    res = compose.compose_pack(
        pack_id="fr-govinfo-dry", family="us_federal_register_ecfr", chunks_dir=str(stage / "chunks"),
        vectors_dir=str(vectors), collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        license_from="license", family_license=None)
    assert res["manifest"]["rows"] == len(rows)

    # the same stubs `test_promote_sealed_pack._plan_with` uses: the index holds nothing, and the
    # corpus-wide persisted-input / payload-carry / layout checks (which need the real corpus)
    # are not what this smoke is about -- it proves the rows join their vectors and pass `plan()`
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: set())
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    plan = PQ.dry_plan(res["manifest"], "http://stub")
    assert len(plan["_joined"]) == len(rows)           # every row is admitted, none refused


def test_the_plan_counts_month_by_month_inside_the_bulk_dates():
    """The API caps a long range's count at 10,000 (measured 2026-10-03: every year read 10000);
    the plan sums the months, and the last year stops at the bulk route's last day."""
    wins = GI.GovinfoLeg(object()).plan_windows(_http(2010), {"from": "2010-01-01", "to": "2010-12-31",
                                                              "max_req_s": "1000"}, TODAY)
    want = len([r for r in _api("2010-03-01")["api_results"]])
    assert [(w["from"], w["to"], w["docs"]) for w in wins] == [("2010-01-01", "2010-12-31", want)]
    wins = GI.GovinfoLeg(object()).plan_windows(
        _http(2025), {"from": "2025-01-01", "to": "2025-12-31", "bulk_dates": "2000-01-01..2025-09-15",
                      "max_req_s": "1000"}, TODAY)
    assert wins[0]["docs"] == 0          # the only listed 2025 documents are dated 2025-09-30


# ------------------------------------------------------------------------------ task 771: portable stamp

def test_the_stage_is_the_same_bytes_from_another_checkout_prov_id_included(tmp_path, declared_contracts, monkeypatch, word_only_chunking_allowed):
    """The policy file loaded from two different absolute paths (two checkouts) gives a
    byte-identical `records.jsonl`, `stamps.prov_id` included (it used to hash the absolute path)."""
    import shutil
    _run(tmp_path / "a", 2005, declared_contracts)
    elsewhere = tmp_path / "another" / "checkout" / "data" / POLICY_FILE.name
    elsewhere.parent.mkdir(parents=True)
    shutil.copy(POLICY_FILE, elsewhere)
    monkeypatch.setenv("AW_CHUNK_POLICY", str(elsewhere))
    _run(tmp_path / "b", 2005, declared_contracts)
    a = (tmp_path / "a" / "o" / "pack" / "records.jsonl").read_bytes()
    b = (tmp_path / "b" / "o" / "pack" / "records.jsonl").read_bytes()
    assert a == b and a
    prov = {json.loads(x)["stamps"]["prov_id"] for x in a.decode().splitlines()}
    assert len(prov) == 1 and str(tmp_path) not in a.decode()
