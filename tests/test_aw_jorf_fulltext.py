"""Parity of the JORF full-text producers after the S7 move (task 1437).

The pinned digests were produced ONCE from the original, pre-move scripts (blob shas in the task
doc) over the synthetic fixture below, then run again over the moved package modules. A change in
a digest is a change in sealed bytes. The fixture is synthetic JORFTEXT XML, with no clock and no
network; every writer is the deterministic gzip writer.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PINNED = {
    "content_markdown": "ec77f4415fe2dcd0bdc92713a849d9a34b12453c65e05d6267a00cefefbbb09b",
    "render_document": "5a17b03c89124e06bc2694161f8ab826382f8ec36c4ac432cf7f2fe433d2e456",
    "render_metadata_only": "ce487a6af8432af3d45a5918647b34cd5b9af446323a2cd1d2511eefe5ef8a37",
    "filter_full": "3a3e7a17fa2072d71493ec234aaa5bb4d9d9754622a3e24d4584941ccb55f761",
    "scan_archive_events": "f3c8b3ad878e600ffe1fca40c6fa48f6dcb91f5168dc5bb2048d8deb01835a6e",
    "finalize": "2367a7c18653d1d7fed2555845c48941285d514776bbfa3b83ea508826a9a5ee",
}


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _record(payload: bytes, member: str) -> dict:
    return {"archive_uri": "artifact://archives/jorf.tar.gz", "member_name": member,
            "xml_base64": base64.b64encode(payload).decode("ascii"), "xml_bytes": len(payload),
            "xml_sha256": hashlib.sha256(payload).hexdigest()}


def _render_fixture():
    target, section_id, article_id = "JORFTEXT000000000001", "JORFSCTA000000000001", "JORFARTI000000000001"
    version = (f"<TEXTELR><META><META_COMMUN><ID>{target}</ID><NATURE>DECRET</NATURE></META_COMMUN>"
               "<META_SPEC><META_TEXTE_VERSION><TITRE>Decret test</TITRE><TITREFULL>Decret test complet"
               "</TITREFULL></META_TEXTE_VERSION><META_TEXTE_CHRONICLE><DATE_PUBLI>2026-08-16</DATE_PUBLI>"
               "<DATE_TEXTE>2026-08-15</DATE_TEXTE><NOR>TEST0001</NOR><ORIGINE_PUBLI>JORF</ORIGINE_PUBLI>"
               "</META_TEXTE_CHRONICLE></META_SPEC></META><NOTICE><CONTENU><p>Notice integrale.</p>"
               "<ul><li>un</li><li>deux</li></ul></CONTENU></NOTICE></TEXTELR>").encode()
    struct = (f"<TEXTELR><META><META_COMMUN><ID>{target}</ID></META_COMMUN></META>"
              f"<STRUCT><LIEN_SECTION_TA id=\"{section_id}\"/></STRUCT></TEXTELR>").encode()
    section = (f"<SECTION_TA><ID>{section_id}</ID><TITRE_TA>Dispositions</TITRE_TA>"
               f"<CONTEXTE><TEXTE cid=\"{target}\"/></CONTEXTE>"
               f"<STRUCTURE_TA><LIEN_ART id=\"{article_id}\"/></STRUCTURE_TA></SECTION_TA>").encode()
    article = (f"<ARTICLE><META><META_COMMUN><ID>{article_id}</ID></META_COMMUN>"
               "<META_SPEC><META_ARTICLE><NUM>1</NUM></META_ARTICLE></META_SPEC></META>"
               f"<CONTEXTE><TEXTE cid=\"{target}\"/></CONTEXTE><BLOC_TEXTUEL><CONTENU><p>Corps integral.</p>"
               "<p>Suite exacte.</p><br/>fin</CONTENU></BLOC_TEXTUEL><LIENS><LIEN id=\"JORFARTI000000000002\" "
               "typelien=\"CITATION\">Article cite</LIEN></LIENS></ARTICLE>").encode()
    identity = {"point_id": "00000000-0000-0000-0000-000000000001", "title": "ancien"}
    text_row = {"records": {"version": _record(version, "version.xml"),
                             "struct": _record(struct, "struct.xml")}}
    linked = {section_id: {"jorftext_id": target, "record_kind": "section", "children": [article_id],
                           "record": _record(section, "section.xml")},
              article_id: {"jorftext_id": target, "record_kind": "article", "children": [],
                           "record": _record(article, "article.xml")}}
    return target, identity, text_row, linked


def _tar(path: Path, rows) -> dict:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in rows:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return {"uri": f"artifact://{path.name}", "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}


def _read(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _tree_digest(root: Path, names: list[str]) -> str:
    return _sha(b"\n".join(n.encode() + b"=" + _sha((root / n).read_bytes()).encode() for n in names))


def compute_digests(m: SimpleNamespace, tmp_path: Path) -> dict[str, str]:
    """`m` carries the modules under test: etree, content_markdown, render_document, filter_mod,
    recover_mod, writer_cls, seal. The same function ran over the pre-move scripts."""
    out: dict[str, str] = {}
    node = m.etree.fromstring(b"<CONTENU><p>A  b</p><div><ul><li>x</li><li>y</li></ul></div>t<br/>u</CONTENU>")
    out["content_markdown"] = _sha(_canon([m.content_markdown(node), m.content_markdown(None)]))

    target, identity, text_row, linked = _render_fixture()
    row, outlinks, counts = m.render_document(target, identity, text_row, linked)
    out["render_document"] = _sha(_canon([row, outlinks, counts]))
    text_row["records"]["struct"] = _record(
        f"<TEXTELR><META><META_COMMUN><ID>{target}</ID></META_COMMUN></META><STRUCT/></TEXTELR>".encode(),
        "s.xml")
    text_row["records"]["version"] = _record(
        f"<TEXTELR><META><META_COMMUN><ID>{target}</ID><NATURE>ARRETE</NATURE></META_COMMUN>"
        "<META_SPEC><META_TEXTE_VERSION><TITRE>Identite seule</TITRE></META_TEXTE_VERSION>"
        "</META_SPEC></META></TEXTELR>".encode(), "v.xml")
    out["render_metadata_only"] = _sha(_canon(list(m.render_document(target, identity, text_row, {}))))

    # filter: one full-article parent, one source-content residual
    root = tmp_path / "f"
    source = root / "source"
    source.mkdir(parents=True)
    markdown, outl = source / "markdown/parents.jsonl.gz", source / "outlinks/parents.jsonl.gz"
    full = {"source_record_id": "JORFTEXT1", "article_bodies": 2, "empty_article_records": 1,
            "baseline_production_point_ids": ["p1"], "markdown_sha256": "a" * 64,
            "parent_identity": {"source_id": "JORFTEXT1"}}
    resid = {"source_record_id": "JORFTEXT2", "article_bodies": 0, "body_sections": 1,
             "source_record_count": 2, "baseline_production_point_ids": ["p2"],
             "markdown_sha256": "b" * 64, "parent_identity": {"source_id": "JORFTEXT2"}}
    for path, rows in ((markdown, [full, resid]), (outl, [
            {"source_record_id": "JORFTEXT1", "links": [{"target": "u"}], "unresolved": [{"reason": "x"}],
             "explicit_empty": False},
            {"source_record_id": "JORFTEXT2", "links": [], "unresolved": [], "explicit_empty": True}])):
        w = m.writer_cls(path)
        for r in rows:
            w.write(r)
        w.close()
    stage = m.seal(artifact_root=root, artifact_paths=[markdown, outl],
                   receipt_path=source / "stages/materialization/receipt.json", records=2, fmt="test",
                   logical_sha256="c" * 64, producer_id="test", producer_version="1")
    run_path = source / "RUN.json"
    run_path.write_text(json.dumps({
        "schema": "aw.jorf-exact-full-markdown-run.v1", "version": "jorf-exact-full-markdown.v2",
        "status": "materialization-complete-profile-fidelity-chunking-pending",
        "counts": {"documents": 2}, "stages": {"materialization": stage}}))
    m.filter_mod.EXPECTED_SOURCE, m.filter_mod.EXPECTED_FULL, m.filter_mod.EXPECTED_RESIDUAL = 2, 1, 1
    output = root / "filtered"
    result = m.filter_mod.filter_full(artifact_root=root, parent_run=run_path, output=output)
    data = sorted(str(p.relative_to(output)) for p in output.rglob("*.jsonl.gz"))
    out["filter_full"] = _sha(_canon([result["counts"], data]) + _tree_digest(output, data).encode())

    # recovery: scan one archive, then finalize one partition
    t = tmp_path / "r"
    t.mkdir()
    struct = b"<TEXTELR><META><ID>JORFTEXT1</ID></META></TEXTELR>"
    version = b"<TEXTE_VERSION><META><ID>JORFTEXT1</ID></META></TEXTE_VERSION>"
    archive = t / "JORF_20260816-000000.tar.gz"
    ref = _tar(archive, [("x/liste_suppression_jorf.dat", b"jorf/global/texte/version/JORF/TEXT/a/JORFTEXT1\n"),
                         ("x/jorf/global/texte/struct/JORF/TEXT/a/JORFTEXT1.xml", struct),
                         ("x/jorf/global/texte/version/JORF/TEXT/a/JORFTEXT1.xml", version),
                         ("x/jorf/global/texte/struct/JORF/TEXT/a/JORFTEXT2.xml", b"ignored")])
    events = t / "events.jsonl.gz"
    w = m.writer_cls(events)
    scan = m.recover_mod._scan_archive(archive, ref, {"JORFTEXT1"}, 7, w)
    w.close()
    out["scan_archive_events"] = _sha(_canon([scan, _read(events)]))

    frontier = t / "frontier.jsonl.gz"
    w = m.writer_cls(frontier)
    w.write({"jorftext_id": "JORFTEXT1"})
    w.close()
    plan_root = t / "plan"
    plan_root.mkdir()
    plan_path = plan_root / "PLAN.json"
    plan_path.write_text(json.dumps({
        "schema": "aw.jorf-exact-archive-recovery-plan.v1",
        "identity_frontier": m.recover_mod._artifact(frontier, t),
        "partitions": [{"partition": 0, "archive_start": 0, "archive_end": 1}]}, sort_keys=True) + "\n")
    part = t / "parts/partition-0000"
    part.mkdir(parents=True)
    ev = part / "events.jsonl.gz"
    w = m.writer_cls(ev)
    s2 = b"<TEXTELR><META><ID>JORFTEXT1</ID></META><STRUCT><LIEN_ART id='JORFARTI1'/></STRUCT></TEXTELR>"
    for kind, payload in (("version", version), ("struct", s2)):
        w.write({"schema": "aw.jorf-archive-state-event.v1", "event": "upsert", "phase": 1,
                 "archive_ordinal": 1, "archive_uri": "artifact://inc.tar.gz", "member_ordinal": 1,
                 "member_name": f"x/texte/{kind}/JORFTEXT1.xml", "jorftext_id": "JORFTEXT1",
                 "record_kind": kind, "xml_bytes": len(payload), "xml_sha256": _sha(payload),
                 "xml_base64": base64.b64encode(payload).decode()})
    w.write({"schema": "aw.jorf-archive-state-event.v1", "event": "delete", "phase": 0, "archive_ordinal": 1,
             "archive_uri": "artifact://inc.tar.gz", "member_ordinal": 99, "line_ordinal": 0,
             "member_name": "liste_suppression_jorf.dat", "suppressed_path": "texte/version/JORFTEXT1",
             "jorftext_id": "JORFTEXT1", "record_kind": "version"})
    w.close()
    st = m.seal(artifact_root=t, artifact_paths=[ev], receipt_path=part / "stages/events/receipt.json",
                records=3, fmt="test", logical_sha256=_sha(b"events"), producer_id="test",
                producer_version="test")
    (part / "RUN.json").write_text(json.dumps({"schema": "aw.jorf-exact-archive-recovery-partition-run.v1",
                                               "status": "complete", "partition": 0,
                                               "stages": {"events": st}}, sort_keys=True) + "\n")
    final = t / "final"
    res = m.recover_mod.finalize(artifact_root=t, plan_path=plan_path, partitions_root=t / "parts",
                                 output=final)
    data = sorted(str(p.relative_to(final)) for p in final.rglob("*.jsonl.gz"))
    out["finalize"] = _sha(_canon([res["status"], res["counts"], data]) + _tree_digest(final, data).encode())
    return out


def _package_modules() -> SimpleNamespace:
    from lxml import etree

    from cytir.acquire import stage_receipt
    from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
    from cytir.extract.dila_content import content_markdown
    from cytir.sources.jorf import filter_full_parents, full_markdown, recover_archives
    return SimpleNamespace(etree=etree, content_markdown=content_markdown,
                           render_document=full_markdown._render_document, filter_mod=filter_full_parents,
                           recover_mod=recover_archives, writer_cls=DeterministicGzipWriter,
                           seal=stage_receipt.seal)


def test_moved_producers_match_the_pinned_pre_move_digests(tmp_path):
    got = compute_digests(_package_modules(), tmp_path)
    assert got == PINNED


def test_two_runs_agree(tmp_path):
    a = compute_digests(_package_modules(), tmp_path / "a")
    b = compute_digests(_package_modules(), tmp_path / "b")
    assert a == b


def test_pins_are_full_sha256():
    assert all(len(v) == 64 and int(v, 16) >= 0 for v in PINNED.values()), "pin the digests first"


def test_package_modules_import_without_side_effects():
    m = _package_modules()
    assert m.filter_mod.PRODUCER.endswith("filter_jorf_full_content_parents.py")


def test_render_fails_closed_when_structural_record_is_missing():
    m = _package_modules()
    target, identity, text_row, linked = _render_fixture()
    linked.pop("JORFSCTA000000000001")
    with pytest.raises(ValueError, match="linked record reachability drift"):
        m.render_document(target, identity, text_row, linked)
