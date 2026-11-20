import hashlib

import cytir.sources.kali.derive_production_markdown as subject


ARTICLE = b"""<ARTICLE><META><META_COMMUN><ID>KALIARTI0001</ID></META_COMMUN>
<META_SPEC><META_ARTICLE><NUM>2</NUM><TITRE/><ETAT>VIGUEUR</ETAT>
<DATE_DEBUT>2020-01-01</DATE_DEBUT><DATE_FIN>2999-01-01</DATE_FIN><TYPE/></META_ARTICLE></META_SPEC></META>
<CONTEXTE><TEXTE><TITRE_TXT>Convention test</TITRE_TXT><TM><TITRE_TM id="S1">Salaires</TITRE_TM></TM></TEXTE>
<CONTENEUR cid="KALICONT0001"/></CONTEXTE><BLOC_TEXTUEL><CONTENU><p>Texte complet.</p><p>Suite.</p></CONTENU></BLOC_TEXTUEL>
<LIENS><LIEN id="KALIARTI0002" typelien="CITATION">Article suivant</LIEN>
<LIEN typelien="CREATION">Source sans cible</LIEN></LIENS></ARTICLE>"""

CONTAINER = b"""<IDCC><META><META_COMMUN><ID>KALICONT0001</ID></META_COMMUN>
<META_SPEC><META_CONTENEUR><TITRE>Convention test</TITRE><ETAT>VIGUEUR</ETAT>
<NUM>1234</NUM><DATE_PUBLI>2020-01-01</DATE_PUBLI></META_CONTENEUR></META_SPEC></META>
<STRUCTURE_TXT><TM><TITRE_TM>Texte de base</TITRE_TM>
<LIEN_TXT idtxt="KALITEXT0001" titretxt="Accord initial"/></TM></STRUCTURE_TXT>
<ACTS_PRO><ACT_PRO>Activite test</ACT_PRO></ACTS_PRO><NUMS_BROCH><NUM_BROCH>3000</NUM_BROCH></NUMS_BROCH></IDCC>"""


def test_article_preserves_full_structure_identity_and_unresolved_relations():
    row = subject.parse_article(
        ARTICLE, member="kali/article/KALIARTI0001.xml", archive_sha256="a" * 64,
        baseline_point_ids=["00000000-0000-0000-0000-000000000001"],
        baseline_titles=["old title"])
    assert row["source_record_id"] == "KALIARTI0001"
    assert row["url"].endswith("/KALIARTI0001")
    assert "### Paragraph 2" in row["markdown"] and "Suite." in row["markdown"]
    assert row["source_links"][0]["target"].endswith("/KALIARTI0002")
    assert row["source_relations_unresolved"][0]["unresolved_reasons"] == ["target_absent"]
    assert row["source_record_sha256"] == hashlib.sha256(ARTICLE).hexdigest()
    replay = subject.parse_article(
        ARTICLE, member="delta/KALIARTI0001.xml", archive_sha256="c" * 64,
        baseline_point_ids=["00000000-0000-0000-0000-000000000001"],
        baseline_titles=[], source_release="20260815-010203")
    assert replay["source_version"]["source_release"] == "20260815-010203"
    assert replay["source_archive"]["release"] == "20260815-010203"


def test_container_preserves_structure_activities_and_source_identity():
    row = subject.parse_container(
        CONTAINER, member="kali/container/KALICONT0001.xml", archive_sha256="b" * 64,
        baseline_point_ids=["00000000-0000-0000-0000-000000000002"],
        baseline_titles=["Convention test"])
    assert row["source_record_id"] == "KALICONT0001"
    assert row["idcc"] == "1234"
    assert "Activite test" in row["markdown"]
    assert row["source_links"][0]["target"].endswith("/KALITEXT0001")
    assert row["baseline_production_titles"] == ["Convention test"]


def test_legacy_caps_are_not_present_in_reconstruction_source():
    source = __import__("inspect").getsource(subject)
    assert "MAX_ARTICLE_CHARS" not in source
    assert "MAX_ARTICLES_PER_CONV" not in source


def test_membership_identity_accepts_bound_top_level_id_and_payload_mirror():
    value = "00000000-0000-0000-0000-000000000003"
    assert subject._membership_point_id(
        {"id": value}, {"point_id": value}, "KALIARTI0001") == value
    try:
        subject._membership_point_id(
            {"id": value}, {"point_id": "different"}, "KALIARTI0001")
    except ValueError as exc:
        assert "identity conflict" in str(exc)
    else:
        raise AssertionError("conflicting membership point IDs accepted")
