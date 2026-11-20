import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from cytir.extract import loda_current as subject


def _article(identifier="LEGIARTI0001", *, state="VIGUEUR", end=""):
    return f"""<ARTICLE><META><META_COMMUN><ID>{identifier}</ID></META_COMMUN>
    <META_SPEC><META_ARTICLE><ETAT>{state}</ETAT><NUM>1</NUM>
    <DATE_DEBUT>1996-01-01</DATE_DEBUT><DATE_FIN>{end}</DATE_FIN>
    </META_ARTICLE></META_SPEC></META><CONTEXTE><TEXTE>
    <TITRE_TXT>Loi test</TITRE_TXT></TEXTE></CONTEXTE>
    <BLOC_TEXTUEL><CONTENU>Contenu integral.</CONTENU></BLOC_TEXTUEL></ARTICLE>""".encode()


def _fixture(tmp_path: Path):
    root = tmp_path / "artifacts"; membership = root / "membership"
    membership.mkdir(parents=True)
    shard = membership / "shard.jsonl.gz"
    row = {"id": "00000000-0000-0000-0000-000000000001", "payload": {
        "point_id": "00000000-0000-0000-0000-000000000001",
        "provenance": "loda", "title": "Titre historique",
        "url": subject.PREFIX + "LEGIARTI0001"}}
    with gzip.GzipFile(filename="", mode="wb", fileobj=shard.open("wb"), mtime=0) as out:
        out.write((json.dumps(row) + "\n").encode())
    (membership / "RECEIPT.json").write_text(json.dumps({
        "rows": 1, "shards": [{"file": shard.name, "bytes": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}))
    archive = root / "legi.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        body = _article(); info = tarfile.TarInfo("legi/LEGIARTI0001.xml")
        info.size = len(body); tar.addfile(info, io.BytesIO(body))
    l0 = root / "L0.json"
    l0.write_text(json.dumps({"artifacts": [{
        "uri": "artifact://legi.tar.gz", "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
    return root, membership, l0


def test_exact_loda_projection_uses_membership_and_full_official_xml(tmp_path):
    root, membership, l0 = _fixture(tmp_path)
    output = root / "loda"
    run = subject.derive(artifact_root=root, membership=membership,
                         l0_receipt=l0, output=output, expected_points=1,
                         expected_source_ids=1)
    assert run["status"] == "complete"
    assert run["counts"]["production_source_ids"] == 1
    with gzip.open(output / "markdown.jsonl.gz", "rt") as source:
        row = json.loads(source.readline())
    assert row["url"] == subject.PREFIX + "LEGIARTI0001"
    assert row["baseline_production_point_ids"] == [
        "00000000-0000-0000-0000-000000000001"]
    assert "Contenu integral." in row["markdown"]
    assert row["source_archive"]["member"] == "legi/LEGIARTI0001.xml"


def test_selector_does_not_broaden_to_same_domain(tmp_path):
    root, membership, l0 = _fixture(tmp_path)
    shard = membership / "shard.jsonl.gz"
    row = {"id": "00000000-0000-0000-0000-000000000001", "payload": {
        "point_id": "00000000-0000-0000-0000-000000000001",
        "provenance": "legi_code", "domain": "legifrance.gouv.fr",
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI0001"}}
    with gzip.GzipFile(filename="", mode="wb", fileobj=shard.open("wb"), mtime=0) as out:
        out.write((json.dumps(row) + "\n").encode())
    receipt = {"rows": 1, "shards": [{"file": shard.name,
        "bytes": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}
    (membership / "RECEIPT.json").write_text(json.dumps(receipt))
    try:
        subject.derive(artifact_root=root, membership=membership,
                       l0_receipt=l0, output=root / "loda", expected_points=1,
                       expected_source_ids=1)
    except ValueError as error:
        assert "LODA production point count drift" in str(error)
    else:
        raise AssertionError("same-domain non-LODA row was selected")


def test_preserves_multiple_baseline_points_for_one_official_source_id(tmp_path):
    root, membership, l0 = _fixture(tmp_path)
    shard = membership / "shard.jsonl.gz"
    rows = [{"id": f"00000000-0000-0000-0000-{ordinal:012d}", "payload": {
        "point_id": f"00000000-0000-0000-0000-{ordinal:012d}",
        "provenance": "loda", "title": f"Titre {ordinal}",
        "url": subject.PREFIX + "LEGIARTI0001"}} for ordinal in (1, 2)]
    with gzip.GzipFile(filename="", mode="wb", fileobj=shard.open("wb"), mtime=0) as out:
        for row in rows:
            out.write((json.dumps(row) + "\n").encode())
    (membership / "RECEIPT.json").write_text(json.dumps({"rows": 2, "shards": [{
        "file": shard.name, "bytes": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}))
    output = root / "loda"
    run = subject.derive(artifact_root=root, membership=membership,
                         l0_receipt=l0, output=output, expected_points=2,
                         expected_source_ids=1)
    assert run["counts"]["production_points"] == 2
    assert run["counts"]["production_source_ids"] == 1
    assert run["production_point_multiplicities"] == {
        "groups": 1, "excess_points": 1}
    evidence = json.loads((output / "production-point-multiplicities.json").read_text())
    assert evidence["groups"][0]["baseline_production_point_ids"] == [
        row["id"] for row in rows]


def test_rejects_duplicate_baseline_point_identity(tmp_path):
    root, membership, l0 = _fixture(tmp_path)
    shard = membership / "shard.jsonl.gz"
    point_id = "00000000-0000-0000-0000-000000000001"
    row = {"id": point_id, "payload": {
        "point_id": point_id, "provenance": "loda",
        "url": subject.PREFIX + "LEGIARTI0001"}}
    with gzip.GzipFile(filename="", mode="wb", fileobj=shard.open("wb"), mtime=0) as out:
        out.write((json.dumps(row) + "\n").encode())
        out.write((json.dumps(row) + "\n").encode())
    (membership / "RECEIPT.json").write_text(json.dumps({"rows": 2, "shards": [{
        "file": shard.name, "bytes": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}))

    try:
        subject.derive(artifact_root=root, membership=membership,
                       l0_receipt=l0, output=root / "loda", expected_points=2,
                       expected_source_ids=1)
    except ValueError as error:
        assert str(error) == f"duplicate LODA production point ID: {point_id}"
    else:
        raise AssertionError("duplicate point identity was accepted")


def test_prefers_official_current_archive_cohort_and_preserves_resolution(tmp_path):
    root, membership, l0 = _fixture(tmp_path)
    archive = root / "legi.tar.gz"
    candidates = [
        ("legi/global/code_et_TNC_non_vigueur/TNC_non_vigueur/"
         "LEGIARTI0001.xml", _article(state="ABROGE", end="2024-12-01")),
        ("legi/global/code_et_TNC_en_vigueur/TNC_en_vigueur/"
         "LEGIARTI0001.xml", _article(state="ABROGE_DIFF", end="2024-12-01")),
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for name, body in candidates:
            info = tarfile.TarInfo(name); info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    l0.write_text(json.dumps({"artifacts": [{
        "uri": "artifact://legi.tar.gz", "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}))

    output = root / "loda"
    run = subject.derive(artifact_root=root, membership=membership,
                         l0_receipt=l0, output=output, expected_points=1,
                         expected_source_ids=1)
    assert run["counts"]["archive_candidate_rows"] == 2
    assert run["counts"]["source_record_version_resolution_groups"] == 1
    with gzip.open(output / "markdown.jsonl.gz", "rt") as source:
        row = json.loads(source.readline())
    assert row["source_version"]["status"] == "ABROGE_DIFF"
    assert "code_et_TNC_en_vigueur" in row["source_archive"]["member"]
    resolution = json.loads(
        (output / "source-record-version-resolution.json").read_text())
    assert resolution["groups"][0]["method"] == \
        "official-current-cohort-precedence"
    assert len(resolution["groups"][0]["candidates"]) == 2


def test_divergent_official_current_candidates_fail_closed():
    candidates = [{
        "archive_member": f"legi/global/code_et_TNC_en_vigueur/{index}.xml",
        "archive_cohort": "official-current",
        "source_record_sha256": digest,
    } for index, digest in enumerate(("a" * 64, "b" * 64), 1)]
    try:
        subject._resolve_candidates("LEGIARTI0001", candidates)
    except ValueError as error:
        assert "ambiguous divergent immutable LODA ID" in str(error)
        assert "current_digests=2" in str(error)
    else:
        raise AssertionError("divergent current-cohort candidates were accepted")
