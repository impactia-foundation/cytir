"""Parity of the KALI full-text producers after the move into the package (task 1436, S7).

A synthetic KALI archive (three conventions, five articles, one article missing from the sealed
archive so the base run is a recovery frontier) runs through derive and then combine. The sha256 of
every emitted row file is pinned; the pins were produced once from the pre-move scripts
(blobs b9e2c3e63b76c03491218699ed6d5dbfbde79d89 derive and 200b0a5379284d8e5e571f5250d481c7d932675f
combine at origin/main) in a scratch copy, so equality proves the move changed no emitted byte.
Neither producer reads a clock (receipt bytes are not pinned: only the row files)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from cytir.sources.kali import combine_production_markdown as COMBINE
from cytir.sources.kali import derive_production_markdown as DERIVE

PINNED = {
    "derive_markdown": "58a7a850ed7877f9724e7ca507de315cf36ce1f22a5f2261dde6aac4dd2713e1",
    "derive_outlinks": "e2cd5b38f4c76bc3d0506956fb8d90ba042c06be39fe321e3b363c4b33a41af0",
    "derive_missing": "2e175d25a98992f837a612e96800763232e755a789c0b69f8f91895b3bf6c42d",
    "combine_parents": "528f9bb31d487937d4b022d3b40aa8e46dc681ad736d060d84b59fb556a02184",
    "combine_outlinks": "d51c8bf47a52c8f793e00fcb8820a9d53c90c88475e20483310ba595819cb338",
}


def _article(num: str, cont: str, conv_title: str, body: str, links: str = "") -> bytes:
    ident = f"KALIARTI{num}"
    return (f"""<ARTICLE><META><META_COMMUN><ID>{ident}</ID></META_COMMUN>
<META_SPEC><META_ARTICLE><NUM>{num}</NUM><TITRE/><ETAT>VIGUEUR</ETAT>
<DATE_DEBUT>2020-01-01</DATE_DEBUT><DATE_FIN>2999-01-01</DATE_FIN><TYPE>AUTONOME</TYPE></META_ARTICLE></META_SPEC></META>
<CONTEXTE><TEXTE><TITRE_TXT>{conv_title}</TITRE_TXT><TM><TITRE_TM id="S{num}">Section {num}</TITRE_TM></TM></TEXTE>
<CONTENEUR cid="{cont}"/></CONTEXTE><BLOC_TEXTUEL><CONTENU>{body}</CONTENU></BLOC_TEXTUEL>
<LIENS>{links}</LIENS></ARTICLE>""").encode()


def _container(num: str, title: str, idcc: str) -> bytes:
    return (f"""<IDCC><META><META_COMMUN><ID>KALICONT{num}</ID></META_COMMUN>
<META_SPEC><META_CONTENEUR><TITRE>{title}</TITRE><ETAT>VIGUEUR</ETAT>
<NUM>{idcc}</NUM><DATE_PUBLI>2020-01-01</DATE_PUBLI></META_CONTENEUR></META_SPEC></META>
<STRUCTURE_TXT><TM><TITRE_TM>Texte de base</TITRE_TM>
<LIEN_TXT idtxt="KALITEXT{num}" titretxt="Accord initial {num}"/></TM></STRUCTURE_TXT>
<ACTS_PRO><ACT_PRO>Activite {num}</ACT_PRO></ACTS_PRO><NUMS_BROCH><NUM_BROCH>30{num}</NUM_BROCH></NUMS_BROCH></IDCC>"""
            ).encode()


ARTICLES = {
    "0001": _article("0001", "KALICONT0001", "Convention un", "<p>Salaire minimum.</p><p>Suite un.</p>",
                     '<LIEN id="KALIARTI0002" typelien="CITATION">Article deux</LIEN>'),
    "0002": _article("0002", "KALICONT0001", "Convention un", "<p>Duree du travail.</p><ul><li>a</li><li>b</li></ul>"),
    "0003": _article("0003", "KALICONT0002", "Convention deux", "<p>Preavis de <b>deux</b> mois.</p>",
                     '<LIEN typelien="CREATION">Sans cible</LIEN>'),
    "0004": _article("0004", "KALICONT0003", "Convention trois", "<p>Conges payes.<br/>Deuxieme ligne.</p>"),
    "0005": _article("0005", "KALICONT0003", "Convention trois", "<p>Texte recupere.</p>"),
}
CONTAINERS = {"0001": _container("0001", "Convention un", "1111"),
              "0002": _container("0002", "Convention deux", "2222"),
              "0003": _container("0003", "Convention trois", "3333")}
MISSING = "KALIARTI0005"


def _binding(path: Path) -> dict:
    body = path.read_bytes()
    root = path.as_posix().split("/artifacts/", 1)[1]
    return {"uri": "artifact://" + root, "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def _json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return _binding(path)


def _tar(path: Path) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        members = [(f"kali/container/KALICONT{n}.xml", b) for n, b in sorted(CONTAINERS.items())]
        members += [(f"kali/article/KALIARTI{n}.xml", b) for n, b in sorted(ARTICLES.items())
                    if f"KALIARTI{n}" != MISSING]
        for name, body in members:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as sink:
        sink.write(raw.getvalue())


def _membership(directory: Path) -> None:
    rows = []
    for n in sorted(ARTICLES):
        rows.append({"id": f"00000000-0000-0000-0000-00000000a{n}",
                     "payload": {"provenance": "kali_article", "title": f"Titre {n}",
                                 "url": "https://www.legifrance.gouv.fr/conv_coll/article/" + f"KALIARTI{n}"}})
    for n in sorted(CONTAINERS):
        rows.append({"id": f"00000000-0000-0000-0000-00000000c{n}",
                     "payload": {"provenance": "kali_conv", "title": f"Convention {n}",
                                 "url": "https://www.legifrance.gouv.fr/conv_coll/id/" + f"KALICONT{n}"}})
    shard = directory / "shard-0.jsonl.gz"
    directory.mkdir(parents=True)
    with shard.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as sink:
        for row in rows:
            sink.write((json.dumps(row, sort_keys=True) + "\n").encode())
    (directory / "RECEIPT.json").write_text(json.dumps(
        {"rows": len(rows), "shards": [{"file": shard.name, "bytes": shard.stat().st_size}]}))


def _gz_text(path: Path) -> bytes:
    with gzip.open(path, "rb") as source:
        return source.read()


def _gz_write(path: Path, rows: list[dict], canonical_json) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as sink:
        for row in rows:
            sink.write(canonical_json(row).encode() + b"\n")
    return _binding(path)


def run_chain(derive_mod, combine_mod, work: Path) -> dict:
    """Derive then combine on the synthetic archive; return the sha256 of every emitted row file."""
    from cytir.acquire.artifact_io import canonical_json
    root = work / "artifacts"
    archive = root / "l0/kali.tar.gz"
    _tar(archive)
    l0 = root / "l0/receipt.json"
    _json(l0, {"artifacts": [_binding(archive)]})
    membership = root / "membership"
    _membership(membership)
    base = root / "base"
    run = derive_mod.derive(artifact_root=root, membership=membership, l0_receipt=l0, output=base,
                            expected_articles=len(ARTICLES), expected_containers=len(CONTAINERS))
    assert run["status"] == "recovery-frontier" and run["counts"]["missing_source_ids"] == 1
    archive_sha = _binding(archive)["sha256"]
    row = derive_mod.parse_article(ARTICLES["0005"], member="increment/KALIARTI0005.xml",
                                   archive_sha256=archive_sha, baseline_point_ids=["x"],
                                   baseline_titles=["Titre 0005"], source_release="20260101-000000")
    link = {"schema": "aw.parent-source-links.v1", "source_id": row["source_id"],
            "source_record_id": row["source_record_id"], "url": row["url"],
            "article_anchor": row["anchor"], "source_version": row["source_version"],
            "parent_markdown_sha256": row["markdown_sha256"], "parent_identity": row["source_archive"],
            "links": row["source_links"], "unresolved_relations": row["source_relations_unresolved"]}
    recovery = root / "recovery"
    receipts = {
        "MARKDOWN-RECEIPT.json": _gz_write(recovery / "markdown-supplement.jsonl.gz", [row], canonical_json),
        "OUTLINKS-RECEIPT.json": _gz_write(recovery / "outlinks-supplement.jsonl.gz", [link], canonical_json),
    }
    for name, binding in receipts.items():
        _json(recovery / name, {"artifacts": [binding]})
    increment = root / "originals/increment.tar.gz"
    increment.parent.mkdir(parents=True)
    increment.write_bytes(b"increment")
    increment_receipt = root / "originals/increment-receipt.json"
    _json(increment_receipt, {"records": 1, "artifacts": [_binding(increment)]})
    _json(recovery / "RUN.json", {"status": "complete", "counts": {"recovered_source_ids": 1,
                                                                    "unresolved_source_ids": 0},
                                  "increments": _binding(increment_receipt)})
    combined = combine_mod.combine(artifact_root=root, base_root=base, recovery_root=recovery,
                                   output=root / "combined", expected=len(ARTICLES) + len(CONTAINERS))
    assert combined["status"] == "complete"

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    return {
        "derive_markdown": sha(_gz_text(base / "markdown.jsonl.gz")),
        "derive_outlinks": sha(_gz_text(base / "outlinks.jsonl.gz")),
        "derive_missing": sha((base / "missing-production-ids.json").read_bytes()),
        "combine_parents": sha(_gz_text(root / "combined/markdown/parents.jsonl.gz")),
        "combine_outlinks": sha(_gz_text(root / "combined/outlinks/parent-source-links.jsonl.gz")),
    }


def test_moved_producers_emit_the_pre_move_bytes(tmp_path):
    assert run_chain(DERIVE, COMBINE, tmp_path) == PINNED


def test_producer_versions_stay_frozen():
    # sealed receipts carry these versions; a rename would seal the same rows to different bytes
    assert DERIVE.VERSION == "kali-exact-production-anchored-markdown.v1"
    assert COMBINE.VERSION == "kali-exact-production-combine.v1"


def test_membership_count_drift_is_refused(tmp_path):
    _membership(tmp_path / "membership")
    with pytest.raises(ValueError, match="count drift"):
        DERIVE._membership(tmp_path / "membership", 99, 3)
