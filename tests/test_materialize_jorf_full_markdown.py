import base64
import hashlib

import pytest

import cytir.sources.jorf.full_markdown as subject


def _record(payload: bytes, member: str) -> dict:
    return {
        "archive_uri": "artifact://archives/jorf.tar.gz",
        "member_name": member,
        "xml_base64": base64.b64encode(payload).decode("ascii"),
        "xml_bytes": len(payload),
        "xml_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture():
    target = "JORFTEXT000000000001"
    section_id = "JORFSCTA000000000001"
    article_id = "JORFARTI000000000001"
    version = f"""<TEXTELR><META><META_COMMUN><ID>{target}</ID><NATURE>DECRET</NATURE></META_COMMUN>
      <META_SPEC><META_TEXTE_VERSION><TITRE>Decret test</TITRE><TITREFULL>Decret test complet</TITREFULL></META_TEXTE_VERSION>
      <META_TEXTE_CHRONICLE><DATE_PUBLI>2026-08-16</DATE_PUBLI><DATE_TEXTE>2026-08-15</DATE_TEXTE>
      <NOR>TEST0001</NOR><ORIGINE_PUBLI>JORF</ORIGINE_PUBLI></META_TEXTE_CHRONICLE></META_SPEC></META>
      <NOTICE><CONTENU><p>Notice integrale.</p></CONTENU></NOTICE></TEXTELR>""".encode()
    struct = f"""<TEXTELR><META><META_COMMUN><ID>{target}</ID></META_COMMUN></META>
      <STRUCT><LIEN_SECTION_TA id=\"{section_id}\"/></STRUCT></TEXTELR>""".encode()
    section = f"""<SECTION_TA><ID>{section_id}</ID><TITRE_TA>Dispositions</TITRE_TA>
      <CONTEXTE><TEXTE cid=\"{target}\"/></CONTEXTE>
      <STRUCTURE_TA><LIEN_ART id=\"{article_id}\"/></STRUCTURE_TA></SECTION_TA>""".encode()
    article = f"""<ARTICLE><META><META_COMMUN><ID>{article_id}</ID></META_COMMUN>
      <META_SPEC><META_ARTICLE><NUM>1</NUM></META_ARTICLE></META_SPEC></META>
      <CONTEXTE><TEXTE cid=\"{target}\"/></CONTEXTE>
      <BLOC_TEXTUEL><CONTENU><p>Corps integral.</p><p>Suite exacte.</p></CONTENU></BLOC_TEXTUEL>
      <LIENS><LIEN id=\"JORFARTI000000000002\" typelien=\"CITATION\">Article cite</LIEN></LIENS>
      </ARTICLE>""".encode()
    identity = {"point_id": "00000000-0000-0000-0000-000000000001", "title": "ancien"}
    text_row = {"records": {"version": _record(version, "version.xml"),
                             "struct": _record(struct, "struct.xml")}}
    linked = {
        section_id: {"jorftext_id": target, "record_kind": "section",
                     "children": [article_id], "record": _record(section, "section.xml")},
        article_id: {"jorftext_id": target, "record_kind": "article",
                     "children": [], "record": _record(article, "article.xml")},
    }
    return target, identity, text_row, linked


def test_render_preserves_ordered_full_body_identity_and_parent_bound_links():
    target, identity, text_row, linked = _fixture()
    row, outlinks, counts = subject._render_document(target, identity, text_row, linked)

    assert row["source_record_id"] == target
    assert row["baseline_production_point_ids"] == [identity["point_id"]]
    assert "## Notice" in row["markdown"] and "Notice integrale." in row["markdown"]
    assert "## Dispositions {#JORFSCTA000000000001}" in row["markdown"]
    assert "### Article 1 {#JORFARTI000000000001}" in row["markdown"]
    assert "Corps integral." in row["markdown"] and "Suite exacte." in row["markdown"]
    assert row["markdown"].index("Dispositions") < row["markdown"].index("Article 1")
    assert row["source_record_count"] == 4
    assert row["parent_identity"]["source_id"] == target
    assert row["parent_identity"]["version"]["xml_sha256"] == text_row["records"]["version"]["xml_sha256"]
    assert row["source_version"] == {
        "nature": "DECRET", "published": "2026-08-16", "signed": "2026-08-15",
        "nor": "TEST0001", "official_publication": "JORF"}
    assert counts["article_bodies"] == 1 and counts["unresolved_links"] == 0
    assert outlinks["parent_id"] == row["point_id"]
    assert outlinks["links"][0]["target_id"] == "JORFARTI000000000002"
    assert outlinks["links"][0]["article_anchor"] == "JORFARTI000000000001"


def test_render_fails_closed_when_structural_record_is_missing():
    target, identity, text_row, linked = _fixture()
    linked.pop("JORFSCTA000000000001")
    with pytest.raises(ValueError, match="linked record reachability drift"):
        subject._render_document(target, identity, text_row, linked)


def test_metadata_only_document_remains_explicit_and_unadmitted():
    target, identity, text_row, _ = _fixture()
    text_row["records"]["struct"] = _record(
        f"<TEXTELR><META><META_COMMUN><ID>{target}</ID></META_COMMUN></META><STRUCT/></TEXTELR>".encode(),
        "struct-empty.xml",
    )
    text_row["records"]["version"] = _record(
        f"<TEXTELR><META><META_COMMUN><ID>{target}</ID><NATURE>ARRETE</NATURE></META_COMMUN>"
        f"<META_SPEC><META_TEXTE_VERSION><TITRE>Identite seule</TITRE></META_TEXTE_VERSION>"
        f"</META_SPEC></META></TEXTELR>".encode(),
        "version-metadata-only.xml",
    )
    row, outlinks, counts = subject._render_document(target, identity, text_row, {})
    assert counts["body_sections"] == 0
    assert counts["article_bodies"] == 0
    assert outlinks["explicit_empty"] is True
    assert row["source_record_count"] == 2
