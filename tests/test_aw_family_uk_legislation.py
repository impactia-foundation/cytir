"""`uk_legislation` extract + chunk on the Protocol (task 2181). No network: ten saved AKN files
(Open Government Licence v3.0, legislation.gov.uk) under tests/fixtures/uk_legislation, all seven types."""
from __future__ import annotations

import pathlib

import pytest

from cytir.extract import uk_legislation_akn as UK
from cytir.sources import contract as contract_mod
from cytir.sources.family import Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.families.official_bulk import UkLegislationFamily

FIX = pathlib.Path(__file__).parent / "fixtures" / "uk_legislation"
TYPES = {"ukpga_2026_16": "ukpga", "ukpga_1990_33": "ukpga", "asp_2010_15": "asp", "nisr_2000_385": "nisr",
         "uksi_2005_3516": "uksi", "eur_2005_2174": "eur", "wsi_2010_2916": "wsi", "ssi_2010_468": "ssi"}


def _ctx() -> RunContext:
    return RunContext(out_dir=pathlib.Path("out"), today="2026-11-01", skip_consent=True,
                      contract=contract_mod.by_id(contract_mod.load())["uk_legislation"], log=lambda m: None)


def _cap(name: str) -> Captured:
    return Captured(Unit(key=name, url=f"file:{name}"), (FIX / f"{name}.akn").read_bytes())


@pytest.fixture(scope="module")
def fam():
    f = UkLegislationFamily()
    f.preflight(_ctx())
    return f


def test_all_seven_types_are_covered():
    assert set(TYPES.values()) == {"ukpga", "asp", "nisr", "uksi", "eur", "wsi", "ssi"}


@pytest.mark.parametrize("name", TYPES)
def test_extract_gives_cited_sections_with_title_and_citation(fam, name, word_only_chunking_allowed):
    docs = fam.extract(_cap(name), _ctx())
    assert any(d.extra["kind"] in ("provision", "schedule") for d in docs)
    for d in docs:
        assert d.text.startswith("# ") and d.lang == "en"
        assert d.extra["citation"] and d.extra["citation"] in d.title and d.extra["citation"] in d.text
        assert d.extra["label"] in d.title and f"## {d.extra['label']}" in d.text
        assert d.url.startswith("https://www.legislation.gov.uk/" + TYPES[name] + "/")
        assert d.source_parts is None or set(d.source_parts[0].text.split()) <= set(d.text.split())   # body words, headings aside
    assert len({d.url for d in docs}) == len(docs)               # one canonical URL per unit


def test_section_boundaries_and_url():
    item = UK.parse((FIX / "ukpga_1990_33.akn").read_bytes())
    prov = [s for s in item.sections if s.kind == "provision"]
    assert [s.label for s in prov] == ["Section 1", "Section 2"]
    assert prov[0].url == "https://www.legislation.gov.uk/ukpga/1990/33/section/1"
    assert "Section 2" not in prov[0].body                         # no text of the next section


def test_statutory_instrument_labels_and_no_empty_chunks(fam, word_only_chunking_allowed):
    ctx = _ctx()
    docs = fam.extract(_cap("uksi_2005_3516"), ctx)
    assert [d.extra["label"] for d in docs[1:]] == ["Article 1", "Article 2", "Article 3", "Article 4"]
    recs = [r for d in docs for r in fam.chunk(d, ctx)]
    assert recs and all(r["text"].strip() and r["dense_text"].strip() and r["sparse_text"].strip() for r in recs)
    assert all(r["stamps"]["chunk"] == 4 for r in recs)


@pytest.mark.parametrize("name", TYPES)
def test_chunk_ids_are_deterministic_and_unique(fam, name, word_only_chunking_allowed):
    ctx = _ctx()

    def run():
        return [(r["url"], r["anchor"], r["point_id"]) for d in fam.extract(_cap(name), ctx) for r in fam.chunk(d, ctx)]

    a, b = run(), run()
    assert a == b and len(set(a)) == len(a) and len({x[2] for x in a}) == len(a)


def test_a_long_section_splits_and_every_chunk_names_title_and_section(fam, word_only_chunking_allowed):
    body = "\n\n".join(f"({i}) The Secretary of State may by regulations make provision about matter {i} "
                       + "and the persons affected by it " * 12 for i in range(1, 60))
    url = "https://www.legislation.gov.uk/ukpga/2026/1/section/9"
    sec = UK.Section("section-9", "provision", "Section 9", "Long power", url, body)
    item = UK.Item("ukpga/2026/1", "ukpga", "Example Act 2026", "2026 c. 1", "https://www.legislation.gov.uk/ukpga/2026/1")
    doc = Doc(url=url, text=UK.compose_text(item, sec), title=UK.compose_title(item, sec), lang="en",
              anchor="section-9", extra={"document_id": "d" * 64})
    recs = fam.chunk(doc, _ctx())
    assert len(recs) > 1
    assert [r["anchor"] for r in recs] == [f"section-9:chunk-{i}" for i in range(len(recs))]
    assert all("Example Act 2026" in r["title"] and "Section 9" in r["title"] for r in recs)
    assert all("Section 9" in r["text"] for r in recs)


def test_repealed_text_is_dropped_and_current_text_kept():
    ns = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    xml = (f'<akomaNtoso xmlns="{ns}"><act><meta><identification>'
           '<FRBRWork><FRBRthis value="http://www.legislation.gov.uk/id/ukpga/2000/1"/><FRBRname value="2000 c. 1"/>'
           '</FRBRWork><FRBRExpression><FRBRlanguage language="eng"/></FRBRExpression></identification></meta>'
           '<preface><block name="title"><docTitle>Test Act 2000</docTitle></block></preface><body>'
           '<section eId="section-1"><num>1</num><heading>Kept</heading><content><p>Old words '
           '<del>repealed words</del><ins>new words</ins> end.<authorialNote><p>footnote</p></authorialNote></p>'
           f'<foreign><table xmlns="http://www.w3.org/1999/xhtml"><tr><th><p xmlns="{ns}">Col</p></th></tr>'
           f'<tr><td><p xmlns="{ns}">cell</p></td></tr></table></foreign></content></section>'
           '<section eId="section-2"><num>2</num><heading>Gone</heading><content><p><del>all repealed</del></p></content></section>'
           '</body></act></akomaNtoso>').encode()
    item = UK.parse(xml)
    assert [s.label for s in item.sections] == ["Section 1"] and item.empty_provisions == 1
    b = item.sections[0].body
    assert "new words" in b and "repealed" not in b and "footnote" not in b
    assert "| Col |" in b and "| cell |" in b


def test_non_item_bodies_are_excluded_with_a_reason(fam):
    for name, reason in (("redirect_stub", "not_akn_redirect_or_html"), ("metadata_only", "no_body_metadata_only")):
        with pytest.raises(UnitExcluded, match=reason):
            fam.extract(_cap(name), _ctx())
    with pytest.raises(UnitExcluded, match="bulk_original"):
        fam.extract(Captured(Unit(key="bulk-original", url="file:x"), b"{}"), _ctx())


def test_chunk_before_preflight_is_refused():
    with pytest.raises(RuntimeError, match="before preflight"):
        UkLegislationFamily().chunk(None, _ctx())


def test_contract_row_is_untouched_by_the_adapter():
    row = contract_mod.by_id(contract_mod.load())["uk_legislation"]
    assert row["chunk_policy"]["tokenizer_revision"] is None        # resolved at run time, row unchanged
    assert row["consent"]["resolution"] == "explicit-grant" and row["licence"]["spdx"] == "OGL-UK-3.0"


# ---- task 2196: --opt akn_dir (local Akoma Ntoso files, no network) ----------------------------

def _akn_ctx(d, **opts) -> RunContext:
    return RunContext(out_dir=pathlib.Path("out"), today="2026-11-01", skip_consent=True,
                      options={"akn_dir": str(d), **opts},
                      contract=contract_mod.by_id(contract_mod.load())["uk_legislation"], log=lambda m: None)


def test_akn_dir_discover_is_sorted_and_uri_comes_from_metadata(tmp_path):
    (tmp_path / "zzz.akn").write_bytes((FIX / "ukpga_1990_33.akn").read_bytes())    # misleading names
    (tmp_path / "aaa.xml").write_bytes((FIX / "uksi_2005_3516.akn").read_bytes())
    (tmp_path / "notes.txt").write_text("ignored")
    units = UkLegislationFamily().discover(_akn_ctx(tmp_path))
    assert [u.key for u in units] == ["id/uksi/2005/3516", "id/ukpga/1990/33"]            # file-name order, URI keys
    assert units[1].meta["source_url"] == "https://www.legislation.gov.uk/id/ukpga/1990/33"
    assert units[1].url == "https://www.legislation.gov.uk/id/ukpga/1990/33"         # the metadata URI, never a path


def test_akn_dir_wrong_host_and_userinfo_are_refused(tmp_path):
    src = (FIX / "ukpga_1990_33.akn").read_bytes()
    for i, host in enumerate(("http://evil.example/id/", "http://user:pw@www.legislation.gov.uk/id/")):
        (tmp_path / f"{i}.akn").write_bytes(src.replace(b"http://www.legislation.gov.uk/id/", host.encode()))
    f = UkLegislationFamily()
    units = f.discover(_akn_ctx(tmp_path))
    assert len(units) == 2
    for u in units:
        with pytest.raises(UnitExcluded, match="host_not_legislation"):
            f.acquire(u, _akn_ctx(tmp_path))


def test_akn_dir_seal_provenance_has_route_and_source_sha_and_no_credentials(fam, tmp_path, word_only_chunking_allowed):
    import hashlib
    body = (FIX / "ukpga_1990_33.akn").read_bytes()
    (tmp_path / "x.akn").write_bytes(body)
    ctx = _akn_ctx(tmp_path)
    [u] = fam.discover(ctx)
    [cap] = fam.acquire(u, ctx)
    assert cap.body == body
    assert cap.meta["route"] == "bulk-dump"
    assert cap.meta["source_file_sha256"] == hashlib.sha256(body).hexdigest()
    assert "@" not in str(cap.meta) and str(tmp_path) not in str(cap.meta)
    assert fam.extract(cap, ctx)


def test_akn_dir_unreadable_files_are_excluded_by_extract_not_dropped(fam, tmp_path):
    (tmp_path / "a.akn").write_bytes((FIX / "redirect_stub.akn").read_bytes())
    ctx = _akn_ctx(tmp_path)
    [u] = fam.discover(ctx)
    [cap] = fam.acquire(u, ctx)
    with pytest.raises(UnitExcluded):
        fam.extract(cap, ctx)


def test_akn_dir_empty_directory_refuses(tmp_path):
    with pytest.raises(ValueError, match="holds no"):
        UkLegislationFamily().discover(_akn_ctx(tmp_path))


def test_akn_dir_sealed_row_carries_no_local_path_and_no_file_uri(fam, tmp_path):
    import json
    from cytir.sources.run_family import seal
    d, out = tmp_path / "in", tmp_path / "out"
    d.mkdir(); out.mkdir(); (out / "l0").mkdir()
    (d / "x.akn").write_bytes((FIX / "ukpga_1990_33.akn").read_bytes())
    (d / "bad.akn").write_bytes(b"not xml")
    ctx = _akn_ctx(d)
    for u in fam.discover(ctx):
        if u.key.startswith("unreadable"):
            seal(out, Captured(u, b"not xml", {}))
        else:
            for cap in fam.acquire(u, ctx):
                seal(out, cap)
    raw = (out / "l0" / "index.jsonl").read_text()
    assert raw.count("\n") == 2
    assert str(tmp_path) not in raw and "file:" not in raw
    assert {json.loads(x)["url"] for x in raw.splitlines() if "ukpga" in x} == {"https://www.legislation.gov.uk/id/ukpga/1990/33"}


def test_every_record_carries_the_declared_licence_and_compact_payload_keeps_it(fam, word_only_chunking_allowed):
    from cytir.core import common as C, license as LIC
    want = LIC.resolve_license(None, "legislation.gov.uk", "www.legislation.gov.uk", "gov-legal", None)
    assert want == "OGL-UK-3.0"
    rows = [r for d in fam.extract(_cap("ukpga_1990_33"), _ctx()) for r in fam.chunk(d, _ctx())]
    assert rows and all(r["license"] == want for r in rows)
    assert all(C.compact_payload(r)["license"] == want for r in rows)


def test_family_declares_the_full_floor():
    assert UkLegislationFamily.staged_text_min_coverage == 1.0


def test_staged_text_guard_is_exact_at_the_default_floor_and_catches_a_lost_tail(fam, word_only_chunking_allowed):
    from cytir.sources import staged_text_guard as G
    for name in TYPES:
        for d in fam.extract(_cap(name), _ctx()):
            rows = [r["text"] for r in fam.chunk(d, _ctx())]
            if d.source_parts:
                [part] = d.source_parts
                assert G.part_coverage(part, rows) == 1.0, (name, d.anchor)
                assert G.part_coverage(part, [rows[0][: len(rows[0]) // 2]]) < 1.0 or len(part.text.split()) < 8


def test_a_body_with_no_words_is_unchecked_and_a_short_body_is_compared():
    from cytir.sources.families.official_bulk import _guard_words
    assert _guard_words("# Heading\n. . .\n") == ("", ())
    assert _guard_words("1 2 3") == ("1 2 3", ())
    assert _guard_words("alpha beta\n## H\ngamma delta") == ("alpha beta gamma delta", (2,))


@pytest.mark.parametrize("body", ["1 2", "6. 18J 39Z 45B", "(a) (b)", "(2) 3 -- ."])
def test_a_body_with_no_lettered_word_gets_no_guard_part(body):
    from cytir.sources.families.official_bulk import _guard_parts
    assert _guard_parts("doc-x", body) is None


def test_a_body_with_a_lettered_word_gets_a_guard_part():
    from cytir.sources.families.official_bulk import _guard_parts
    [part] = _guard_parts("doc-x", "Schedule 2 has effect.")
    assert part.text == "Schedule 2 has effect."


def _long_section():
    body = "\n\n".join(f"({i}) The Secretary of State may by regulations make provision about matter {i} "
                       + "and the persons affected by it " * 12 for i in range(1, 60))
    url = "https://www.legislation.gov.uk/ukpga/2026/1/section/9"
    sec = UK.Section("section-9", "provision", "Section 9", "Long power", url, body)
    item = UK.Item("ukpga/2026/1", "ukpga", "Example Act 2026", "2026 c. 1", "https://www.legislation.gov.uk/ukpga/2026/1")
    doc = Doc(url=url, text=UK.compose_text(item, sec), title=UK.compose_title(item, sec), lang="en",
              anchor="section-9", extra={"document_id": "d" * 64})
    return body, doc


def test_chunk_window_splits_are_declared_cuts_and_a_lost_tail_is_still_caught(fam, word_only_chunking_allowed):
    """Task 2200: a window split loses exactly the 5-grams on its boundary; they are declared as
    cuts, so a split section reads 1.0 and a dropped window still reads below it."""
    from cytir.sources import staged_text_guard as G
    from cytir.sources.families.official_bulk import _guard_parts
    body, doc = _long_section()
    rows = [r["text"] for r in fam.chunk(doc, _ctx())]
    assert len(rows) > 2
    [part] = _guard_parts("d" * 64, body, rows)
    assert len(part.cuts) >= len(rows) - 2
    assert G.part_coverage(part, rows) == 1.0
    [bare] = _guard_parts("d" * 64, body)
    assert G.part_coverage(bare, rows) < 1.0     # without the declared splits those 5-grams are lost
    assert G.part_coverage(part, rows[:-1]) < 1.0
    assert G.part_coverage(part, rows[:1] + rows[2:]) < 1.0


def test_row_cuts_are_found_by_order_not_by_the_first_repeat():
    from cytir.sources.families.official_bulk import _row_cuts
    words = ("a b c d e f g h " * 3 + "x1 x2 x3 x4 x5 x6 x7 x8 " + "a b c d e f g h " * 3).split()
    crumb = "Act > Section 1"
    rows = ["Act\n\nlead", crumb + "\n\n" + " ".join(words[:24]),
            crumb + "\n\n" + " ".join(words[24:])]
    assert _row_cuts(words, rows) == (24,)
    assert _row_cuts(words, []) == ()


def _uniq(n: int) -> list[str]:
    """n distinct lettered tokens (no repeated phrase, so a 5-gram occurs once)."""
    return [f"wd{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(n)]


def _unique_long_section():
    toks = _uniq(60 * 40)
    body = "\n\n".join(f"({i}) " + " ".join(toks[(i - 1) * 40:i * 40]) for i in range(1, 61))
    url = "https://www.legislation.gov.uk/ukpga/2026/1/section/9"
    sec = UK.Section("section-9", "provision", "Section 9", "Long power", url, body)
    item = UK.Item("ukpga/2026/1", "ukpga", "Example Act 2026", "2026 c. 1", "https://www.legislation.gov.uk/ukpga/2026/1")
    doc = Doc(url=url, text=UK.compose_text(item, sec), title=UK.compose_title(item, sec), lang="en",
              anchor="section-9", extra={"document_id": "d" * 64})
    return body, doc


def _lose_word(rows: list[str], k: int, offset: int) -> str:
    """Delete the body word at `offset` of row k (its breadcrumb paragraph stays); return the word."""
    crumb, sep, body = rows[k].partition("\n\n")
    assert " > " in crumb
    words = body.split()
    lost = words.pop(offset)
    rows[k] = crumb + sep + " ".join(words)
    return lost


@pytest.mark.parametrize("offset", [0, 3])
def test_a_unique_word_lost_at_or_just_after_a_window_start_reads_below_one(fam, offset, word_only_chunking_allowed):
    from cytir.sources import staged_text_guard as G
    from cytir.sources.families.official_bulk import _guard_parts
    body, doc = _unique_long_section()
    rows = [r["text"] for r in fam.chunk(doc, _ctx())]
    assert len(rows) > 2
    [part] = _guard_parts("d" * 64, body, rows)
    assert G.part_coverage(part, rows) == 1.0
    damaged = list(rows)
    lost = _lose_word(damaged, 1, offset)
    assert body.split().count(lost) == 1                     # the lost word is unique in the section
    [dpart] = _guard_parts("d" * 64, body, damaged)          # cuts rebuilt from the damaged rows, as production does
    assert G.part_coverage(dpart, damaged) < 1.0


def test_akn_dir_malformed_port_refuses_that_unit_and_keeps_the_directory(tmp_path):
    src = (FIX / "ukpga_1990_33.akn").read_bytes()
    (tmp_path / "0.akn").write_bytes(src.replace(b"http://www.legislation.gov.uk/id/", b"http://www.legislation.gov.uk:abc/id/"))
    (tmp_path / "1.akn").write_bytes(src)
    f = UkLegislationFamily()
    units = f.discover(_akn_ctx(tmp_path))
    assert len(units) == 2
    refused = [u for u in units if u.meta.get("refusal")]
    assert [u.meta["refusal"] for u in refused] == ["work_uri_host_not_legislation_gov_uk"]
    with pytest.raises(UnitExcluded, match="host_not_legislation"):
        f.acquire(refused[0], _akn_ctx(tmp_path))
    assert [u.key for u in units if not u.meta.get("refusal")] == ["id/ukpga/1990/33"]


def test_akn_dir_skips_symlinks_like_non_akn_files(tmp_path):
    d, outside = tmp_path / "in", tmp_path / "outside"
    d.mkdir(); outside.mkdir()
    real = outside / "real.akn"
    real.write_bytes((FIX / "ukpga_1990_33.akn").read_bytes())
    (d / "link.akn").symlink_to(real)
    with pytest.raises(ValueError, match="holds no"):        # the only *.akn entry is a symlink: no file
        UkLegislationFamily().discover(_akn_ctx(d))
    (d / "own.akn").write_bytes((FIX / "uksi_2005_3516.akn").read_bytes())
    units = UkLegislationFamily().discover(_akn_ctx(d))
    assert [u.key for u in units] == ["id/uksi/2005/3516"]


