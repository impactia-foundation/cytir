"""`service_public_dila_exact` on the family Protocol (task 1403).

No network. The archive is built in the test from two hand-written fiche members (the shape of
the DILA `vosdroits` zip), the same as the driver's own characterization test. Pins: registered
and conformant (chunk declared, task 2260); a `plan` dry run through the registry lists the F members and no
other; extract uses the driver's function (exact markdown, outlinks, empty fiche excluded,
duplicate URL excluded); consent and licence are copied from the row as they stand (`unknown`).
"""
from __future__ import annotations

import zipfile

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.service_public_dila_exact import ServicePublicDilaExactFamily
from cytir.sources.run_family import run_family

GOOD = b'''<Publication xmlns:dc="http://purl.org/dc/elements/1.1/"
  spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F1">
  <dc:title>Une fiche utile</dc:title><Introduction><Paragraphe>Texte utile
  <Lien href="https://example.fr/a"/></Paragraphe></Introduction></Publication>'''
EMPTY = b'''<Publication xmlns:dc="http://purl.org/dc/elements/1.1/"
  spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F2">
  <dc:title>Vide</dc:title></Publication>'''


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "dump.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("part/F1.xml", GOOD)
        z.writestr("part/F2.xml", EMPTY)
        z.writestr("part/F3.xml", GOOD)          # same canonical URL as F1
        z.writestr("part/N1.xml", b"<Navigation/>")
    return path


def _ctx(archive, tmp_path) -> RunContext:
    row = contract_mod.by_id(contract_mod.load())["service_public_dila_exact"]
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True,
                      options={"archive": str(archive)}, contract=row, log=lambda m: None)


def test_registered_and_conformant():
    assert "service_public_dila_exact" in registry.registered()
    adapter, row, spec = registry.admit("service_public_dila_exact")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "service_public_dila_exact"
    assert not getattr(adapter, "PARTIAL", ())
    assert adapter.chunk_policy_id == "declared"


def test_plan_dry_run_lists_only_fiche_members(archive, tmp_path):
    plan = run_family("service_public_dila_exact", out_dir=tmp_path / "o", today="2026-10-30",
                      skip_consent=True, options={"archive": str(archive)}, dry_run=True,
                      log=lambda m: None)
    assert plan["units"] == ["part/F1.xml", "part/F2.xml", "part/F3.xml"]
    assert plan["verdicts"] == []                    # no host is contacted
    assert plan["discover"]["files"] == 3


def test_extract_exact_markdown_and_terminals(archive, tmp_path):
    fam, ctx = ServicePublicDilaExactFamily(), _ctx(archive, tmp_path)
    units = list(fam.discover(ctx))
    caps = {u.key: fam.acquire(u, ctx)[0] for u in units}
    (doc,) = fam.extract(caps["part/F1.xml"], ctx)
    assert doc.title == "Une fiche utile" and doc.lang == "fr"
    assert doc.text.startswith("# Une fiche utile\n\nTexte utile")
    assert [x["target_url"] for x in doc.extra["outlinks"]] == ["https://example.fr/a"]
    with pytest.raises(UnitExcluded) as e:
        fam.extract(caps["part/F2.xml"], ctx)
    assert e.value.reason == "no_admissible_practical_fiche_markdown"
    with pytest.raises(UnitExcluded) as e:
        fam.extract(caps["part/F3.xml"], ctx)
    assert e.value.reason == "duplicate_canonical_url"


def test_terms_come_from_the_row_unknown_stays_unknown(archive, tmp_path):
    fam, ctx = ServicePublicDilaExactFamily(), _ctx(archive, tmp_path)
    fam.discover(ctx)
    terms = ctx.notes["contract_terms"]
    row = ctx.contract
    assert terms["search_ok"] == row["consent"]["search_ok"]
    assert terms["licence_basis"] == row["licence"]["basis"]


def test_unsafe_member_refused(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../F1.xml", b"<Publication/>")
    with pytest.raises(Exception, match="unsafe archive member"):
        list(ServicePublicDilaExactFamily().discover(_ctx(path, tmp_path)))


def test_a_run_names_the_excluded_units_and_their_reasons(archive, tmp_path):
    # task 2260: the row declares the generic epoch-4 policy, so the run now composes records and
    # writes a pack; the excluded count and reasons ride in the run summary: units_excluded == 2
    # (one empty fiche, one duplicate canonical URL). The run is the stage door's own entry point.
    from cytir.sources.run_family import load_pack
    out = tmp_path / "o"
    summary = run_family("service_public_dila_exact", out_dir=out, today="2026-10-30",
                         skip_consent=True, options={"archive": str(archive)}, log=lambda m: None)
    text = str(summary)
    assert "'no_admissible_practical_fiche_markdown': 1" in text
    assert "'duplicate_canonical_url': 1" in text
    _, recs = load_pack(out / "pack")
    assert recs and all("vosdroits" in r["url"] for r in recs)
    assert {r["license"] for r in recs} == {"Etalab-2.0"}


# ---- byte parity with the pre-move driver -------------------------------------------------
# Reference digests come from running the ORIGINAL driver (acquire_service_public_dump_preserved.py
# as of commit 9b23bc5089d2624222cafb8f1965bb2e539b42c1) in isolation on exactly this fixture.
# The only time the driver stamps is `fetched_at`, copied from the transport metadata written
# below, so no timestamp needs stripping. URL and vantage are fixed. Same fixture, moved code:
# RUN.json and every sealed stage receipt must hash the same.
_PARITY_URL = "https://example.test/vosdroits.zip"
_PARITY_RUN_JSON = "d372b926c99653ce9d9fec138fe311045b2347236dd1612b573e4b800255d19b"
_PARITY_STAGES = {  # stage -> (logical_sha256, sha256 of receipt.json)
    "original": ("799c392f7adb700f775cbebc5ec7f15dc4aefb934c2fcf2bc17f1da3d77b3187",
                 "e2f59fc456ad21137008794c5cdf5c6d056c0acd48dd4e7dee9e7a42e862b4e6"),
    "markdown": ("11ac4e72d9fe2081cc6eec5d189f618e9ad2eec12797d91dde28713080b3a9c4",
                 "7219b626f61d29b90559ae542558eea3a431c5fc184c375bd9d4297b320bd328"),
    "outlinks": ("cb5a096092c1c4844051bad50ae3a3af9ddffc7551f6c6cecd9809669cfae20a",
                 "1765825479bee45efeef9e06a6364b1d20bf39d9fc28acf1487650fcfc4e9e2c"),
    "quarantine": ("54f62d53d4636fa243352c858542e67096812716e0be60a04cb70d9400c545b0",
                   "1dd87358d8fdb4b3be25fa08a2493c6cb8ce09cf37bcd62211e47185393a4471"),
}


def test_seal_is_byte_identical_to_the_original_driver(tmp_path):
    import hashlib
    import json
    from cytir.sources.service_public import dump

    good = (b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" spUrl="https://www.service-public.gouv.fr/'
            b'particuliers/vosdroits/F1"><dc:title>Une fiche utile</dc:title><Introduction><Paragraphe>Texte utile '
            b'<Lien href="https://example.fr/a"/> https://example.org/b.</Paragraphe></Introduction></Publication>')
    empty = (b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" spUrl="https://www.service-public.gouv.fr/'
             b'particuliers/vosdroits/F2"><dc:title>Vide</dc:title></Publication>')
    src = tmp_path / "in"
    src.mkdir()
    archive, headers, meta = src / "dump.zip", src / "headers.txt", src / "meta.json"
    with zipfile.ZipFile(archive, "w") as z:
        for name, body in (("part/F1.xml", good), ("part/F2.xml", empty), ("part/F3.xml", good),
                           ("part/N1.xml", b"<Navigation/>")):
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 30, 0, 0, 0)), body)
    headers.write_bytes(b"HTTP/1.1 200 OK\r\n")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    meta.write_text(json.dumps({
        "schema": "aw.http-fetch-metadata.v1", "requested_url": _PARITY_URL, "http_status": 200,
        "final_url": _PARITY_URL, "fetched_at": "2026-10-30T00:00:00Z",
        "body": {"bytes": archive.stat().st_size, "sha256": sha(archive)},
        "response_headers": {"bytes": headers.stat().st_size, "sha256": sha(headers)}}))
    root = tmp_path / "run"
    run = dump.materialize(archive=archive, headers=headers, metadata=meta, root=root,
                           requested_url=_PARITY_URL, fetch_vantage="vantage-a")
    assert sha(root / "RUN.json") == _PARITY_RUN_JSON
    got = {k: (run["stages"][k]["logical_sha256"], sha(root / "stages" / k / "receipt.json"))
           for k in _PARITY_STAGES}
    assert got == _PARITY_STAGES
