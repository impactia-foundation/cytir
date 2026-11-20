"""PDF replay route of the family Protocol (task 782): sealed L0 PDF bytes -> cytir.extract.pdf -> stage.

Two real PDFs from task 781's L0 sample (`tests/fixtures/pdf-replay/`, public institutional guidance,
no named natural person): a WEKO secretariat Merkblatt (8 pages, Docling) and an ElCom Merkblatt
(3 pages, liteparse, one image placeholder). The extraction results are the ones task 781 stored
(`<sha>.md` + `<sha>.json`), returned by a stand-in extractor, because liteparse and Docling are not
in CI; a real-engine test runs when liteparse is installed. Everything else is the real route: the
contract refusal, the listing, the consent door, the chunker under the declared policy, the ids, the
staged-text guard, the stage door, and the URL-group supersede composition.

Pins: a HELD contract row refuses with the field names and extracts nothing; image-only lines are
dropped; every chunk is inside the 2,560-token window and the guard sees no shortfall; ids follow the
contract's `minted_by` (both rules); the pack passes `family_stage.check_pack` and the prose hold-out;
the URL-group supersede removes the old `mint_point_id` rows by URL and the new rows arrive, with the
net change computed; `--from` stays refused for a live sitemap family.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.sources import contract, family_stage, registry, run_family, stage_out  # noqa: E402
from cytir.sources.families import pdf_replay  # noqa: E402

FIX = HERE / "fixtures" / "pdf-replay"
WEKO_SHA = "e1a9c7e895d7f9abea95993e318ccc8276050090c0549525b00d8d7c3e0601e1"
ELCOM_SHA = "e819f74d41a63cd26087c8e7e886ece1f6813f884200371258bb2ba285258d06"
WEKO_URL = ("https://www.weko.admin.ch/dam/de/sd-web/c01jCJs2aS1j/Merkblatt%20des%20Sekretariats"
            "%20der%20WEKO%20Wettbewerbsabreden%20auf%20Arbeitsm%C3%A4rkten.pdf")
ELCOM_URL = ("https://www.elcom.admin.ch/dam/de/sd-web/0EewzlT7Pr-l/Merkblatt%20der%20Pflichten"
             "%20f%C3%BCr%20Erzeuger.pdf")
WEKO_FAMILY, ELCOM_FAMILY = "scratch_weko_docs_exact", "scratch_elcom_docs_exact"
TODAY = "2026-10-08"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
WINDOW = 2560
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-10-08T00:00:00Z"}
RULE_TEXT_ANCHOR = "text-anchor-v0"
RULE_MINT = "cytir.core.ids.mint_point_id, the one re-mint rule"


def _consent(host, direct):
    return {**OK, "host": host}


@pytest.fixture(autouse=True)
def _declared_policy(monkeypatch):
    from cytir.core import model_pin
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest())
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", model_pin.EMBED_MODEL_REVISION)
    yield
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


@pytest.fixture
def pack_loads():
    """A real run reads the family's pack (`sources/`) for language, tier and trust. The public
    cut ships a pack whose extractor config is not in the cut, so a run cannot start there; the
    tests that need one are skipped in that tree and run in the full one."""
    from cytir.sources import pack_loader as PL
    try:
        PL.load_pack(PL.find_pack("ch-regulator-docs"))
    except Exception as exc:                        # noqa: BLE001 -- any load failure
        pytest.skip(f"the regulator pack does not load in this tree: {type(exc).__name__}")


# ----------------------------------------------------------------------------- the extractor

class Calls(list):
    pass


@pytest.fixture
def extractor(monkeypatch):
    """The stand-in for `cytir.extract.pdf.extract_pdf`: the markdown task 781 stored for each sha."""
    from cytir.extract import pdf as X

    calls = Calls()

    def fake(path):
        sha = pathlib.Path(path).stem
        calls.append(sha)
        md_file = FIX / f"{sha}.md"
        if md_file.is_file():
            meta = json.loads((FIX / f"{sha}.json").read_text())
            prov = meta["prov"]
            return X.ExtractResult(
                markdown=md_file.read_text(encoding="utf-8"), extractor=prov["extractor"],
                extractor_version=prov["extractor_version"], route_reason=prov["route_reason"],
                page_count=prov["page_count"], is_complex=None, table_dense_runs=None)
        return X.ExtractResult(markdown=SYNTH[sha], extractor="liteparse", extractor_version="t",
                               route_reason="test", page_count=1, is_complex=None,
                               table_dense_runs=None)

    monkeypatch.setattr(pdf_replay, "EXTRACTOR", fake)
    return calls


SYNTH: dict[str, str] = {}


# ----------------------------------------------------------------------------- the contract

def _fill(row: dict, minted_by: str) -> dict:
    """A COMPLETE row (a fake: the real values are Tim's POLICY PR), cloned from a declared
    family for the shapes and from the real regulator row for the rest."""
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    donor = next(f for f in doc["families"] if f["family_id"] == "us_federal_register_ecfr")
    row = copy.deepcopy(row)
    row["status"] = "declared"
    row["route"] = "direct"
    for k in ("chunk_policy", "representation", "title_shape", "gate_classes"):
        row[k] = copy.deepcopy(donor[k])
    row["licence"] = copy.deepcopy(donor["licence"])
    row["licence"]["spdx"] = "fake-licence-for-test"
    row["identity"] = copy.deepcopy(donor["identity"])
    row["identity"]["minted_by"] = minted_by
    row["unknowns"] = []
    return row


def _contracts(tmp_path, minted_by=RULE_MINT, *, hold=(), name="contracts.json"):
    """A contracts file where the WEKO and ElCom rows are complete, except the dotted `hold` fields,
    which are set back to `unknown` and listed in `unknowns`."""
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    for i, f in enumerate(doc["families"]):
        if f["family_id"] in (WEKO_FAMILY, ELCOM_FAMILY):
            row = _fill(f, minted_by)
            for dotted in hold:
                node = row
                *head, last = dotted.split(".")
                for h in head:
                    node = node[h]
                node[last] = "unknown"
            row["unknowns"] = sorted(hold)
            if hold:
                row["status"] = "backfilled"       # the only status that may hold an `unknown`
            doc["families"][i] = row
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _listing(tmp_path, rows):
    p = tmp_path / "l0" / "listing.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _weko_listing(tmp_path, **extra):
    return _listing(tmp_path, [{"url": WEKO_URL, "sha256": WEKO_SHA,
                                "path": str(FIX / f"{WEKO_SHA}.pdf"), "provenance": "weko_docs",
                                **extra}])


def _run(tmp_path, listing, contracts, family=WEKO_FAMILY, name="o", **kw):
    kw.setdefault("consent", _consent)
    return run_family.run_family(
        family, out_dir=tmp_path / name, today=TODAY, log=lambda m: None,
        options={pdf_replay.LISTING_OPTION: str(listing)}, contracts_path=contracts, **kw)


def _rows(tmp_path, name="o"):
    return [json.loads(x) for x in
            (tmp_path / name / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- 1. HELD refuses

ROW_FIELDS = ["chunk_policy.epoch", "chunk_policy.policy_id", "chunk_policy.sha256",
              "identity.minted_by", "licence.basis", "licence.instrument",
              "representation.dense_surface", "representation.ruling",
              "representation.sparse_surface", "route", "title_shape.mode"]


@pytest.mark.parametrize("field", ROW_FIELDS)
def test_a_held_field_refuses_by_name_before_any_extraction(tmp_path, extractor, field):
    contracts = _contracts(tmp_path, hold=[field])
    with pytest.raises(registry.FamilyRefused) as exc:
        _run(tmp_path, _weko_listing(tmp_path), contracts)
    msg = str(exc.value)
    # the contract's own words: a half-held chunk policy is refused at admission in its own terms
    assert field in msg or (field == "chunk_policy.sha256" and "sha256 unknown" in msg)
    assert extractor == [], "a HELD row must refuse before a PDF is extracted"
    assert not (tmp_path / "o" / "pack").exists()


# Ruling 31 / T19 (task 872) declared the last HELD field, `gate_classes[0] = civic`, on the five
# regulator rows. The real rows are now COMPLETE: the door admits all four PDF families.
PDF_FAMILIES = ("scratch_weko_docs_exact", "scratch_bav_docs_exact", "scratch_bazl_docs_exact",
                "scratch_elcom_docs_exact")


@pytest.mark.parametrize("family", PDF_FAMILIES + ("astra_html_exact",))
def test_the_real_regulator_rows_are_complete_and_civic(family):
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == family)
    assert row["gate_classes"] == ["civic"]
    assert "gate_classes[0]" not in contract.actual_unknowns(row)
    if family in PDF_FAMILIES:
        assert row["unknowns"] == [] and contract.actual_unknowns(row) == []
        assert pdf_replay.contract_refusal(row, family) is None
        assert pdf_replay.id_rule(row) in (pdf_replay.TEXT_ANCHOR_V0, pdf_replay.MINT_POINT_ID)
        assert pdf_replay.contract_licence(row)


def test_the_word_coverage_ruling_is_declared_for_the_replay_route():
    # skip-per-URL (ADR-114 am.2, ruling 31): the route declares the stage-guard floor that the
    # sealed block must carry, and the per-id line-coverage floor.
    assert pdf_replay.MIN_COVERAGE == 0.98
    assert pdf_replay.WORD_COVERAGE_RULING == {
        "mode": "skip-per-url", "adr": "ADR-114 am.2", "ruling": 31,
        "stage_guard_floor": 0.98, "per_id_line_coverage_floor": 0.8,
        "fallback": "snippet-guard-0.50"}
    doc = json.loads((REPO / contract.DEFAULT_PATH).read_text(encoding="utf-8"))
    for f in doc["families"]:
        if f["family_id"] in PDF_FAMILIES:
            assert any("skip-per-URL" in e for e in f["evidence"]), f["family_id"]


@pytest.mark.parametrize("family,url,sha", [
    (WEKO_FAMILY, WEKO_URL, WEKO_SHA), (ELCOM_FAMILY, ELCOM_URL, ELCOM_SHA)])
def test_the_door_opens_on_the_real_row_and_replays_one_real_l0_pdf(tmp_path, extractor, pack_loads, family, url, sha, word_only_chunking_allowed):
    """No fake contract: the committed row, the real sealed PDF of task 781's sample, the real
    chunker, ids, guard and stage door. The extractor is the stored 781 result (no engine in CI)."""
    listing = _listing(tmp_path, [{"url": url, "sha256": sha, "path": str(FIX / f"{sha}.pdf"),
                                   "provenance": "weko_docs"}])
    res = run_family.run_family(
        family, out_dir=tmp_path / "o", today=TODAY, consent=_consent, log=lambda m: None,
        options={pdf_replay.LISTING_OPTION: str(listing)})
    assert extractor == [sha]
    assert res is not None
    rows = _rows(tmp_path)
    assert rows and all(r["url"] == url for r in rows)


@pytest.mark.usefixtures("pack_loads")
def test_the_cli_refuses_a_held_row_and_writes_no_stage(tmp_path, extractor, capsys, monkeypatch):
    # The committed rows are complete (ruling 31), so the hold here is the live consent verdict.
    # It is injected: this test used to rely on the box having no network (CI), and on a box with
    # network the host cleared, the PDF was extracted, and the chunker preflight refused instead.
    monkeypatch.setattr(run_family, "default_consent",
                        lambda host, direct: {"host": host, "bucket": "error", "error": "held"})
    listing = _weko_listing(tmp_path)
    rc = run_family.main([WEKO_FAMILY, "--from", str(listing), "--out", str(tmp_path / "stage"),
                          "--today", TODAY])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "stage").exists() and extractor == []


@pytest.mark.usefixtures("pack_loads")
def test_the_cli_leaves_no_stage_when_the_tokenizer_is_unavailable(tmp_path, extractor, capsys,
                                                                   monkeypatch):
    from cytir.chunk import chunker
    monkeypatch.setattr(run_family, "default_consent", _consent)
    from cytir.chunk import token_counter

    def _no_tokenizer(*a, **k):
        raise OSError("tokenizer cache unreachable (test)")
    # deterministic on any box: whatever is cached, the resolver fails and the cache is empty
    monkeypatch.setattr(token_counter, "resolve_tokenizer", _no_tokenizer)
    monkeypatch.setattr(chunker, "_tokenizer_cache", {})
    monkeypatch.setitem(chunker._TOKENIZER_STATE, "status", None)
    monkeypatch.setattr(chunker, "_ALLOW_WORD_ONLY_FOR_TESTS", False)
    rc = run_family.main([WEKO_FAMILY, "--from", str(_weko_listing(tmp_path)),
                          "--out", str(tmp_path / "stage"), "--today", TODAY])
    assert rc == 1
    assert "tokenizer unavailable" in capsys.readouterr().err
    assert not (tmp_path / "stage").exists()


def test_an_id_rule_the_route_cannot_run_is_refused(tmp_path, extractor):
    contracts = _contracts(tmp_path, minted_by="some bespoke minter")
    with pytest.raises(registry.FamilyRefused, match="neither id rule"):
        _run(tmp_path, _weko_listing(tmp_path), contracts)
    both = _contracts(tmp_path, minted_by="text-anchor-v0 or mint_point_id", name="both.json")
    with pytest.raises(registry.FamilyRefused, match="both id rules"):
        _run(tmp_path, _weko_listing(tmp_path), both, name="o2")
    assert extractor == []


def test_a_family_without_the_route_refuses_a_listing(tmp_path, extractor):
    with pytest.raises(registry.FamilyRefused, match="no replay-l0 route"):
        run_family.run_family("swissmedic_html_exact", out_dir=tmp_path / "o", today=TODAY,
                              consent=_consent, log=lambda m: None,
                              options={pdf_replay.LISTING_OPTION: str(_weko_listing(tmp_path))})


# ----------------------------------------------------------------------------- 2. --from door

def test_from_is_allowed_for_the_replay_families_and_refused_for_a_live_sitemap_family():
    for fam in (WEKO_FAMILY, ELCOM_FAMILY, "scratch_bav_docs_exact", "scratch_bazl_docs_exact"):
        assert stage_out.offline_refusal(fam) is None
        assert stage_out.is_replay_family(fam)
    why = stage_out.offline_refusal("swissmedic_html_exact")
    assert why and "contacts hosts" in why
    assert not stage_out.is_replay_family("swissmedic_html_exact")


def test_a_listing_that_is_not_a_file_is_refused(tmp_path, extractor):
    with pytest.raises(stage_out.StageOutRefused, match="not a sealed L0 listing"):
        stage_out.run_to_stage(WEKO_FAMILY, out=tmp_path / "s", today=TODAY,
                               from_dir=tmp_path / "nowhere", contracts_path=_contracts(tmp_path))


# ----------------------------------------------------------------------------- 3. a run end to end

@pytest.mark.usefixtures("pack_loads")
def test_two_pdfs_become_rows_through_the_stage_door(tmp_path, extractor, word_only_chunking_allowed):
    contracts = _contracts(tmp_path)
    listing = _listing(tmp_path, [
        {"url": WEKO_URL, "sha256": WEKO_SHA, "path": str(FIX / f"{WEKO_SHA}.pdf"),
         "provenance": "weko_docs"},
        {"url": ELCOM_URL, "sha256": ELCOM_SHA, "path": str(FIX / f"{ELCOM_SHA}.pdf"),
         "provenance": "elcom_docs"}])
    # one listing, two hosts: each family takes its own host's url and counts the other
    m = _run(tmp_path, listing, contracts)
    t = m["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1
    assert t["excluded_reasons"] == {"host_not_this_family:www.elcom.admin.ch": 1}
    assert m["route"] == "replay-l0" and m["consent"]["checked"] is True
    assert m["contract"]["unknowns"] == []
    assert m["discover"]["id_rule"] == "mint_point_id"
    prov = m["discover"]["pdf_extract"][WEKO_URL]
    assert prov["extractor"] == "docling" and prov["page_count"] == 8 and prov["l0_sha256"] == WEKO_SHA
    # the pack is accepted by the door and nothing is held out as binary
    family_stage.check_pack(tmp_path / "o" / "pack", WEKO_FAMILY)
    held: list = []
    n = family_stage._validate_rows(tmp_path / "o" / "pack" / "records.jsonl", held)
    assert n == m["records"]["count"] and held == []
    rows = _rows(tmp_path)
    assert {r["url"] for r in rows} == {WEKO_URL}
    assert {r["license"] for r in rows} == {"fake-licence-for-test"}      # the CONTRACT's value
    assert {r["provenance"] for r in rows} == {"weko_docs"}
    assert all(r["title"].startswith("Merkblatt des Sekretariats der WEKO") for r in rows)
    assert all(r["domain"] == "weko.admin.ch" for r in rows)


@pytest.mark.usefixtures("pack_loads")
def test_aw_source_run_from_writes_a_stage(tmp_path, extractor, monkeypatch, word_only_chunking_allowed):
    contracts = _contracts(tmp_path)
    listing = _weko_listing(tmp_path)
    meta = stage_out.run_to_stage(WEKO_FAMILY, out=tmp_path / "stage", today=TODAY,
                                  from_dir=listing, contracts_path=contracts, consent=_consent,
                                  log=lambda m: None)
    assert meta["rows"] >= 1 and (tmp_path / "stage" / "chunks" / "shard-0000.texts.jsonl.gz").is_file()
    assert extractor == [WEKO_SHA]


@pytest.mark.usefixtures("pack_loads")
def test_the_directory_form_of_the_listing_works(tmp_path, extractor, word_only_chunking_allowed):
    listing = _weko_listing(tmp_path)
    assert pdf_replay.read_listing(listing.parent)[0].url == WEKO_URL
    m = _run(tmp_path, listing.parent, _contracts(tmp_path))
    assert m["terminals"]["units_admitted"] == 1


@pytest.mark.usefixtures("pack_loads")
def test_a_listing_with_a_wrong_hash_quarantines_the_unit_and_never_extracts(tmp_path, extractor):
    bad = _listing(tmp_path, [{"url": WEKO_URL, "sha256": "0" * 64,
                               "path": str(FIX / f"{WEKO_SHA}.pdf")}])
    with pytest.raises(run_family.RunError, match="every one of 1 unit"):
        _run(tmp_path, bad, _contracts(tmp_path))
    assert extractor == []


@pytest.mark.usefixtures("pack_loads")
def test_a_consent_decline_excludes_the_unit(tmp_path, extractor):
    def declined(host, direct):
        return {"host": host, "bucket": "declined"}
    m = _run(tmp_path, _weko_listing(tmp_path), _contracts(tmp_path), consent=declined)
    assert m["terminals"]["excluded_reasons"] == {"consent_declined:www.weko.admin.ch": 1}
    assert m["records"]["count"] == 0 and extractor == []


# ----------------------------------------------------------------------------- 4. image lines

def test_image_only_lines_are_dropped_in_both_engines_forms():
    md = ("## Title\n\n![](img_p1_1.png)\n\n![alt](img_p2_3.png)\n\n<!-- image -->\n\n"
          "Text with ![inline](x.png) image stays.\n\n\n\nLast.")
    out, n = pdf_replay.drop_image_lines(md)
    assert n == 3
    assert out == "## Title\n\nText with ![inline](x.png) image stays.\n\nLast."


@pytest.mark.usefixtures("pack_loads")
def test_no_row_holds_an_image_placeholder(tmp_path, extractor, word_only_chunking_allowed):
    listing = _listing(tmp_path, [{"url": ELCOM_URL, "sha256": ELCOM_SHA,
                                   "path": str(FIX / f"{ELCOM_SHA}.pdf")}])
    assert "![](img_p1_1.png)" in (FIX / f"{ELCOM_SHA}.md").read_text()
    m = _run(tmp_path, listing, _contracts(tmp_path), family=ELCOM_FAMILY)
    assert m["discover"]["pdf_extract"][ELCOM_URL]["image_lines_dropped"] == 1
    for r in _rows(tmp_path):
        assert "![](" not in r["text"] and "<!-- image -->" not in r["text"]


# ----------------------------------------------------------------------------- 5. the window

@pytest.mark.usefixtures("pack_loads")
def test_every_chunk_is_inside_the_window_and_the_guard_sees_no_shortfall(tmp_path, extractor, word_only_chunking_allowed):
    from cytir.chunk import chunker

    paras = [x for x in (FIX / f"{WEKO_SHA}.md").read_text().split("\n\n") if len(x.split()) > 25]
    long_md = "## Lange Verfügung\n\n" + "\n\n".join(
        f"Randziffer {i}. {paras[i % len(paras)]}" for i in range(120))
    sha = hashlib.sha256(b"%PDF-1.7 synthetic long").hexdigest()
    SYNTH[sha] = long_md
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic long")
    listing = _listing(tmp_path, [
        {"url": WEKO_URL, "sha256": WEKO_SHA, "path": str(FIX / f"{WEKO_SHA}.pdf")},
        {"url": "https://www.weko.admin.ch/dam/de/sd-web/x/lange-verfuegung-2026.pdf",
         "sha256": sha, "path": str(pdf)}])
    m = _run(tmp_path, listing, _contracts(tmp_path))
    rows = _rows(tmp_path)
    by_url: dict[str, list] = {}
    for r in rows:
        by_url.setdefault(r["url"], []).append(r)
    assert len(by_url) == 2 and len(by_url["https://www.weko.admin.ch/dam/de/sd-web/x/"
                                           "lange-verfuegung-2026.pdf"]) > 1
    assert max(len(r["text"].split()) for r in rows) <= 1152      # the policy's hard word max
    if chunker.tokenizer_status() == "ok":                      # CI has no tokenizer; the build box does
        assert max(chunker.token_len(r["text"]) for r in rows) <= WINDOW
    g = m["staged_text_guard"]
    assert g["threshold_declared"] and g["min_coverage"] == 0.98
    assert g["docs_checked"] == 2 and g["docs_unchecked"] == 0 and g["docs_shortfall"] == 0
    # the chunks hold the words of the extracted markdown (the guard's own measure, 5-grams)
    from cytir.sources import staged_text_guard as SG
    src, _ = pdf_replay.drop_image_lines((FIX / f"{WEKO_SHA}.md").read_text())
    assert SG.coverage(pdf_replay.guard_text(src), [r["text"] for r in by_url[WEKO_URL]]) >= 0.98


# ----------------------------------------------------------------------------- 6. id minting

def _expected_ids(rows, rule):
    from cytir.core import canon, common as C, ids
    out = []
    for r in rows:
        if rule == "text-anchor-v0":
            out.append(C.mint_passage_point_id(r["url"], r.get("anchor"), r["text"]))
        else:
            d = ids.content_digest(r["text"])
            _t, comp = ids.classify_tier(r.get("source_type"), r.get("anchor"), True, d,
                                         title=r.get("title"))
            out.append(ids.mint_point_id(canon.canonical_url(r["url"]), comp))
    return out


@pytest.mark.usefixtures("pack_loads")
def test_ids_follow_the_contracts_minted_by_for_both_rules(tmp_path, extractor, word_only_chunking_allowed):
    rows_by_rule = {}
    for rule, minted_by in (("text-anchor-v0", RULE_TEXT_ANCHOR), ("mint_point_id", RULE_MINT)):
        contracts = _contracts(tmp_path, minted_by=minted_by, name=f"{rule}.json")
        m = _run(tmp_path, _weko_listing(tmp_path), contracts, name=f"o-{rule}")
        assert m["discover"]["id_rule"] == rule
        rows = _rows(tmp_path, f"o-{rule}")
        assert [r["point_id"] for r in rows] == _expected_ids(rows, rule)
        rows_by_rule[rule] = rows
    a, b = rows_by_rule["text-anchor-v0"], rows_by_rule["mint_point_id"]
    assert [r["text"] for r in a] == [r["text"] for r in b], "same chunks under both rules"
    assert not {r["point_id"] for r in a} & {r["point_id"] for r in b}, "two id spaces"


# ----------------------------------------------------------------------------- 7. the prose guard

@pytest.mark.usefixtures("pack_loads")
def test_binary_text_never_reaches_a_stage_and_the_door_holds_it_out_if_it_did(tmp_path, extractor, word_only_chunking_allowed):
    """An extraction that is a PDF's own syntax read as text is stopped twice: the chunk gates admit
    no row from it and the staged-text guard quarantines the unit (its text did not reach the rows),
    and the stage door's own `binary_content`
    hold-out, which is not bypassed, would hold a row that got past them."""
    noise = ("%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
             "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n" * 40)
    sha = hashlib.sha256(b"%PDF-1.4 junk").hexdigest()
    SYNTH[sha] = noise
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"%PDF-1.4 junk")
    listing = _listing(tmp_path, [
        {"url": WEKO_URL, "sha256": WEKO_SHA, "path": str(FIX / f"{WEKO_SHA}.pdf")},
        {"url": "https://www.weko.admin.ch/dam/de/sd-web/y/rohdaten.pdf", "sha256": sha,
         "path": str(junk)}])
    m = _run(tmp_path, listing, _contracts(tmp_path))
    assert m["terminals"]["units_admitted"] == 1
    reasons = {**m["terminals"]["excluded_reasons"], **m["terminals"]["quarantined_reasons"]}
    assert sum(reasons.values()) == 1 and set(reasons) <= {"staged_text_shortfall", "chunk_empty"}
    # the door, fed a replay row whose text is binary, holds it out and counts it
    rows = _rows(tmp_path)
    bad = dict(rows[0], text=noise, point_id="00000000-0000-0000-0000-000000000001")
    pack = tmp_path / "bad.jsonl"
    pack.write_text(json.dumps(rows[1]) + "\n" + json.dumps(bad) + "\n")
    held: list = []
    family_stage._validate_rows(pack, held)
    assert [i for i, _u, _w in held] == [bad["point_id"]] and held[0][2].startswith("binary_content")
    with pytest.raises(family_stage.StageError, match="not prose"):
        family_stage._validate_rows(pack)


# ----------------------------------------------------------------------------- 8. real engine

def test_the_real_engine_reads_the_elcom_pdf():
    pytest.importorskip("liteparse", reason="real-engine half needs liteparse")
    from cytir.extract import pdf as X
    res = X.extract_pdf(FIX / f"{ELCOM_SHA}.pdf")
    assert res.page_count == 3 and len(res.markdown.split()) > 700


# ----------------------------------------------------------------------------- 9. supersede

def _legacy_rows(url, provenance, n):
    """The served `aw_b1` rows of a PDF url: its raw bytes read as text, chunked, ids from
    `mint_point_id` (Tier B, digest as anchor): what task 767 found on the twin."""
    from cytir.core import canon, ids
    raw = (FIX / f"{WEKO_SHA}.pdf").read_bytes().decode("latin-1")
    out = []
    for i in range(n):
        text = raw[i * 1500:(i + 1) * 1500]
        d = ids.content_digest(text)
        pid = ids.mint_point_id(canon.canonical_url(url), d)
        out.append((pid, {"url": url, "provenance": provenance, "source_type": "official",
                          "content_digest": d, "title": "Merkblatt%20des%20Sekretariats",
                          "snippet": " ".join(text.split()[:25]), "build_id": "aw_b1"}))
    return out


@pytest.mark.usefixtures("pack_loads")
def test_url_group_supersede_removes_the_old_rows_by_url_whatever_their_id_scheme(tmp_path, extractor, monkeypatch, word_only_chunking_allowed):
    """The served rows carry `mint_point_id` ids; the replacement rows carry the contract's rule
    (here `text-anchor-v0`, a different id space). The supersede names the old rows by URL group
    and provenance, never by id, so the swap is clean under either rule."""
    from cytir.index import promote_queue as Q
    from cytir.index import supersede as US
    from cytir.index import url_group_supersede as UGS

    old = _legacy_rows(WEKO_URL, "weko_docs", 7)
    card = ("card-1", {"url": WEKO_URL, "source_type": "entity_card", "provenance": "weko_docs"})
    other_lane = ("other-1", {"url": WEKO_URL, "provenance": "endorsement-discovery",
                              "source_type": "official"})
    elsewhere = _legacy_rows("https://www.weko.admin.ch/dam/de/sd-web/z/not-in-pack.pdf",
                             "weko_docs", 2)
    served = old + [card, other_lane] + elsewhere

    def members(_twin, _collection, urls):
        keys = {US.payload_group_key({"url": u}) for u in urls}
        out: dict = {}
        for pid, pl in served:
            k = US.payload_group_key(pl)
            if US.payload_group_key({"url": pl["url"]}) in keys:
                out.setdefault(k, []).append((pid, pl))
        return out
    monkeypatch.setattr(Q, "served_group_members", members)

    results = {}
    for rule, minted_by in (("text-anchor-v0", RULE_TEXT_ANCHOR), ("mint_point_id", RULE_MINT)):
        contracts = _contracts(tmp_path, minted_by=minted_by, name=f"{rule}.json")
        _run(tmp_path, _weko_listing(tmp_path), contracts, name=f"o-{rule}")
        new_rows = _rows(tmp_path, f"o-{rule}")
        pack = {r["point_id"]: {k: v for k, v in r.items() if k != "text"} for r in new_rows}
        groups, cards = UGS.pack_groups(pack, "weko_docs")
        plan = UGS.compute(pack, groups, cards, "weko_docs", twin_url="http://twin",
                           collection="aw_prod")
        old_ids = {pid for pid, _pl in old}
        assert set(plan["ids"]) == old_ids, "exactly the old weko_docs rows of the url leave"
        assert not old_ids & set(pack), "new ids are not old ids, under either rule"
        assert "card-1" not in plan["ids"] and "other-1" not in plan["ids"]
        assert plan["kept_other_family"] == 1 and plan["superseded_groups"] == 1
        assert plan["groups"] == 1 and plan["pack_rows"] == len(new_rows)
        # the public dump's change key: no digest survives the swap
        old_digests = {pl["content_digest"] for _pid, pl in old}
        assert not old_digests & {r["content_digest"] for r in new_rows}
        net = len(new_rows) - len(plan["ids"])
        results[rule] = {"removed": len(plan["ids"]), "added": len(new_rows), "net": net}
        # WORDS per page, not row existence: the new rows hold the whole extracted document ...
        doc_words = len(pdf_replay.drop_image_lines((FIX / f"{WEKO_SHA}.md").read_text())[0].split())
        assert sum(len(r["text"].split()) for r in new_rows) >= 0.98 * doc_words
        # ... and the drain's word-coverage guard (b) reads SNIPPET words (<= 25 per row), where
        # a few large chunks look small next to many small garbage rows. Measured, printed, and
        # carried to the task doc: it is a landing question, not a property of this route.
        new_snip = sum(Q._snippet_words(pl) for pl in pack.values())
        old_snip = sum(Q._snippet_words(pl) for _pid, pl in old)
        results[rule]["snippet_ratio"] = round(new_snip / old_snip, 3)
    assert results["text-anchor-v0"]["added"] == results["mint_point_id"]["added"]
    print("SUPERSEDE", json.dumps(results))


@pytest.mark.usefixtures("pack_loads")
def test_a_pack_row_without_provenance_cannot_open_a_group(tmp_path, extractor, word_only_chunking_allowed):
    from cytir.index import promote_queue as Q
    from cytir.index import url_group_supersede as UGS

    listing = _weko_listing(tmp_path)
    rows = json.loads(listing.read_text())
    rows.pop("provenance")
    listing.write_text(json.dumps(rows) + "\n")
    _run(tmp_path, listing, _contracts(tmp_path))
    pack = {r["point_id"]: {k: v for k, v in r.items() if k != "text"} for r in _rows(tmp_path)}
    with pytest.raises(Q.QueueError, match="provenance"):
        UGS.pack_groups(pack, "weko_docs")
