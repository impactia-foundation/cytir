import hashlib

from cytir.extract import service_public_fiche as subject


def test_fiche_render_preserves_hierarchy_date_and_member_version():
    xml = b'''<Publication xmlns:dc="http://purl.org/dc/elements/1.1/"
      ID="F1" type="Fiche" spUrl="https://service-public.gouv.fr/F1">
      <dc:title>Faire une chose</dc:title><dc:identifier>F1</dc:identifier>
      <dc:date>modified 2026-08-01</dc:date><dc:type>Fiche</dc:type>
      <Texte><Chapitre><Titre><Paragraphe>Une section</Paragraphe></Titre>
      <Paragraphe>Texte complet.</Paragraphe></Chapitre></Texte></Publication>'''
    row = subject.render(xml, "a" * 64, "F1.xml")
    assert row["modified_date"] == "2026-08-01"
    assert "### Une section" in row["markdown"]
    assert "Texte complet." in row["markdown"]
    assert row["source_sha256"] == hashlib.sha256(xml).hexdigest()

