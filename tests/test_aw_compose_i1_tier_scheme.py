"""Compose can verify the ids of the four regulator PDF replay families (task 876).

The replay door mints `mint_point_id` ids from the chunk's content digest (Tier B). The staged row
has no `anchor`, so the b1 check could not reproduce them and compose refused every such pack
(default `legacy-passage` scheme). The scheme `i1-tier-digest` reproduces them; this module pins
that it equals the door's own rule on the same rows, that a wrong id is refused, and that the
contract's declared rule for the four families is the one the scheme implements.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.index import compose  # noqa: E402
from cytir.sources.families import pdf_replay  # noqa: E402

FAMILIES = ("scratch_weko_docs_exact", "scratch_bav_docs_exact", "scratch_bazl_docs_exact",
            "scratch_elcom_docs_exact")
URL = "https://www.weko.admin.ch/dam/de/sd-web/abc123/Merkblatt%20Test.pdf"


def _rows(n=3):
    out = []
    for i in range(n):
        row = {"url": URL, "title": "Merkblatt Test", "source_type": "official",
               "text": f"Erster Absatz des Merkblatts, Teil {i}. " * 40}
        row["point_id"] = pdf_replay.mint_ids(pdf_replay.MINT_POINT_ID, row)
        out.append(row)
    return out


def _stage(tmp_path, rows):
    d = tmp_path / "chunks"
    d.mkdir()
    with gzip.open(d / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(d)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_four_families_declare_the_i1_scheme(family):
    assert compose.id_scheme_for_family(family) == compose.ID_SCHEME_I1_TIER
    assert compose.ID_SCHEME_I1_TIER in compose.ID_SCHEMES


def test_the_scheme_reproduces_the_doors_ids_row_for_row(tmp_path):
    rows = _rows()
    assert len({r["point_id"] for r in rows}) == 3
    for r in rows:
        assert compose._mint_i1_tier(r) == pdf_replay.mint_ids(pdf_replay.MINT_POINT_ID, r)
    got = compose.verify_identity(_stage(tmp_path, rows), scheme=compose.ID_SCHEME_I1_TIER)
    assert (got["checked"], got["matched"]) == (3, 3)


def test_a_wrong_id_is_refused(tmp_path):
    rows = _rows()
    rows[1]["point_id"] = "00000000-0000-5000-8000-000000000001"
    with pytest.raises(compose.ComposeError, match=r"identity check FAILED: 1/3"):
        compose.verify_identity(_stage(tmp_path, rows), scheme=compose.ID_SCHEME_I1_TIER)


def test_the_committed_contract_names_the_rule_the_scheme_implements():
    doc = json.loads((REPO / "data" / "source-family-contracts.v1.json").read_text())
    rows = doc if isinstance(doc, list) else (doc.get("families") or doc)
    rows = rows if isinstance(rows, list) else list(rows.values())
    by = {r.get("family_id"): r for r in rows}
    for family in FAMILIES:
        minted_by = str(by[family]["identity"]["minted_by"])
        assert pdf_replay.MINT_POINT_ID in minted_by and pdf_replay.TEXT_ANCHOR_V0 not in minted_by, \
            (family, minted_by[:80])


# ---- the manifest of a URL-group supersede pack ------------------------------------------------
def _stub_consent(monkeypatch):
    """The consent stamp is its own tested engine (task 684); here it only has to hand compose a
    stamped copy of the stage."""
    import shutil

    def stamp(chunks_dir, out_dir, ctx, **kw):
        if out_dir:
            shutil.copytree(chunks_dir, out_dir, dirs_exist_ok=True)
        return {"rows_in": 2, "rules": {}, "stamped_from": "test"}
    monkeypatch.setattr(compose.CONSENT_STAMP, "stamp_chunks_dir", stamp)
    monkeypatch.setattr(compose.CONSENT_STAMP, "load_context", lambda *a, **k: None)


def _compose(tmp_path, **kw):
    rows = _rows(2)
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    (vectors / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": 2, "dim": 1024,
         "dtype": "float16"}))
    res = compose.compose_pack(
        pack_id="t876", family="scratch_weko_docs_exact", chunks_dir=str(chunks),
        vectors_dir=str(vectors), collection="aw_e8", mrl=256, queue_dir=str(tmp_path / "queue"),
        **kw)
    return json.loads((pathlib.Path(res["pack_dir"]) / "manifest.json").read_text())


def test_a_supersede_pack_manifest_carries_the_two_fields_the_drain_reads(tmp_path, monkeypatch):
    _stub_consent(monkeypatch)
    m = _compose(tmp_path, supersede_url_group="task 876: replace garbage PDF rows")
    assert m["supersede"] == "url-group"
    assert m["allow_supersede"] == "task 876: replace garbage PDF rows"
    assert m["identity_check"] == "pack-declared" and "2/2" in m["identity_reason"]
    from cytir.index import url_group_supersede as UGS
    assert UGS.declared(m) is True and UGS.family_provenance(m) == "weko_docs"


def test_a_plain_compose_manifest_has_no_supersede_field(tmp_path, monkeypatch):
    _stub_consent(monkeypatch)
    m = _compose(tmp_path)
    assert "supersede" not in m and "allow_supersede" not in m


def test_an_empty_supersede_reason_is_refused(tmp_path, monkeypatch):
    _stub_consent(monkeypatch)
    with pytest.raises(compose.ComposeError, match="needs a reason"):
        _compose(tmp_path, supersede_url_group="  ")


def test_the_cli_takes_the_flag():
    ns = compose.build_parser().parse_args([
        "--pack-id", "p", "--family", "f", "--chunks-dir", "c", "--vectors-dir", "v",
        "--collection", "aw_e8", "--mrl", "256", "--supersede-url-group", "why"])
    assert ns.supersede_url_group == "why"
