"""Offline regressions for Fedlex short-article retention (task 358b).

Run: python3 -m pytest tests/test_fedlex_dump.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import json
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.helpers.repo_files import repo_file_or_skip

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE))
# The public cut withholds the task-358 supersede builder; skip there (task 2411).
R2_BUILDER = repo_file_or_skip(pathlib.Path(ROOT, "tasks", "358-scripts", "build_r2_supersede.py"))


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _temporary_stubs(**mods):
    """Install import stubs for the duration of a module load, then restore.

    Task 432: these stubs used to be dropped into `sys.modules` at import time
    and left there. pytest collects every test file into ONE process, so from
    that moment on every LATER test module in the run imported the stubs instead
    of the real `fetch_process` / `index_bakeoff` / `qdrant_client`. A plain
    `pytest pipeline/` therefore reported failures in files that had nothing
    wrong with them (test_process_html, test_pack_fetch, test_model_pin), while
    each of those files passed on its own — the suite was order-dependent, and a
    newcomer running it saw red for no reason.

    The modules loaded below capture their references during the `with` block,
    so restoring afterwards does not affect them.
    """
    saved = {name: sys.modules.get(name) for name in mods}
    for name, module in mods.items():
        if sys.modules.get(name) is None:
            sys.modules[name] = module
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# `fetch_process` is imported only for an unrelated lazy-import side effect in
# production; keep these parser/projection contract tests dependency-light.
# `project` is pure, but stage_pack imports its GPU embedding runner at module
# load time.  Stub only those unrelated runners so this remains an offline
# projection-contract test on minimal test environments.
_qdrant_stub = ModuleType("qdrant_client")
_qdrant_stub.QdrantClient = object
_qdrant_stub.models = SimpleNamespace()

with _temporary_stubs(fetch_process=ModuleType("fetch_process"),
                      qdrant_client=_qdrant_stub):
    # the loader moved into the package (task 683); a path-loaded shim copy would not see
    # this file's monkeypatches, so the test imports the module that runs.
    sys.path.insert(0, str(ROOT))
    import cytir.sources.families.fedlex as fedlex
    r2 = _load(R2_BUILDER, "fedlex_r2_repeal_test")
    import cytir.index.stage_pack_b1 as stage

AKN_XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act><preface><docTitle>Obligationenrecht</docTitle></preface><body>
    <article eId="art_159"><num>Art. 159</num>
      <authorialNote>[Aufgehoben durch Ziff. II Art. 6 Ziff. 1 des BG vom 25. Juni 1971.]</authorialNote>
    </article>
    <article eId="art_160"><num>Art. 160</num>
      <authorialNote>[Redaktioneller Hinweis ohne Aufhebung.]</authorialNote>
    </article>
    <article eId="art_161"><num>Art. 161</num><paragraph>Kurze echte Norm.</paragraph></article>
    <article eId="art_162"><num>Art. 162</num></article>
    <article eId="art_163"><num>Art. 163</num><paragraph>&#x2026;</paragraph>
      <authorialNote>[Aufgehoben durch Ziff. II Art. 6 Ziff. 1 des BG vom 25. Juni 1971.]</authorialNote>
    </article>
  </body></act>
</akomaNtoso>'''

HIERARCHY_XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act><preface><docTitle>Obligationenrecht</docTitle></preface><body>
    <title eId="title_8"><num>Achter Titel</num><heading>Die Miete</heading>
      <chapter eId="chapter_1"><num>Erstes Kapitel</num><heading>Allgemeine Bestimmungen</heading>
        <article eId="art_257"><num>Art. 257</num><paragraph>Der Mietzins ist das Entgelt.</paragraph></article>
      </chapter>
    </title>
  </body></act>
</akomaNtoso>'''


def _source_manifest(tmp_path, *, bad_url=False, failed=False):
    elis = [f"cc/2026/{number}" for number in range(18)]
    entries = []
    for eli in elis:
        url = fedlex.xml_url(eli, "20260701", "de")
        entries.append({"eli": eli, "languages": {"de": {
            "status": "parse_failed", "error": "historical parser failure",
            "source_version": "20260701", "source_version_url": url} if failed else {
                "status": "success", "source_version": "20260701", "title_prefix": "Pinned law (PIN; SR 1)",
                "source_version_url": url + "-wrong" if bad_url else url}}})
    path = tmp_path / "source-versions.json"
    path.write_text(json.dumps({"schema": fedlex.SOURCE_VERSION_MANIFEST_SCHEMA,
                                "elis": entries}), encoding="utf-8")
    return elis, f"{hashlib.sha256(path.read_bytes()).hexdigest()}={path}"


class _XMLResponse:
    def __init__(self, content, url):
        self.status_code, self.content, self.url = 200, content, url


def test_pure_authorial_repeal_becomes_explicit_pointer():
    rows = fedlex.parse_akn(
        AKN_XML, "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "de",
        "cc/27/317_321_377", {"de": {"short": "OR", "sr": "220"}})
    row = {row["base_anchor"]: row for row in rows}["art_159"]
    assert row["base_anchor"] == "art_159"
    assert "anchor" not in row  # I1 tier B rows carry no native anchor (task 1035c)
    assert row["status"] == "repealed"
    assert "Aufgehoben durch" in row["snippet"]
    assert "Aufgehoben durch" in row["text"]


def test_short_provision_and_authoritative_source_note_are_retained():
    rows = fedlex.parse_akn(
        AKN_XML, "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "de",
        "cc/27/317_321_377", {"de": {"short": "OR", "sr": "220"}})
    by_anchor = {row["base_anchor"]: row for row in rows}
    assert set(by_anchor) == {"art_159", "art_160", "art_161", "art_163"}
    assert by_anchor["art_160"]["status"] == "source_note"
    assert "status" not in by_anchor["art_161"]
    # Ellipsis is an AKN placeholder, not a real operative provision.
    assert by_anchor["art_163"]["status"] == "repealed"


def test_channel_split_keeps_hierarchy_sparse_and_article_local_dense():
    rows = fedlex.parse_akn(
        HIERARCHY_XML,
        "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "de",
        "cc/27/317_321_377", {"de": {"short": "OR", "sr": "220"}})
    legacy_rows = fedlex.parse_akn(
        HIERARCHY_XML,
        "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "de",
        "cc/27/317_321_377", {"de": {"short": "OR", "sr": "220"}},
        representation="legacy")
    row = rows[0]
    assert row["text"] == "Art. 257 Der Mietzins ist das Entgelt."
    assert row["dense_text"] == f"{row['title']}. Fedlex. {row['text']}"   # R1 (task 1040): compose_index_text
    assert "Die Miete" not in row["dense_text"]
    assert "Die Miete" in row["sparse_text"]
    assert "Allgemeine Bestimmungen" in row["sparse_text"]
    # The door mints I1 tier B (task 1035c): the id follows the digest of the row's text, so a
    # representation that changes the text changes the id; the url (the address) stays stable.
    assert [row["url"] for row in rows] == [row["url"] for row in legacy_rows]
    for r in rows + legacy_rows:
        assert r["point_id"] == fedlex.stamp_i1_identity(dict(r))["point_id"]
    assert [r["text"] for r in rows] != [r["text"] for r in legacy_rows]
    assert [r["point_id"] for r in rows] != [r["point_id"] for r in legacy_rows]


def test_pinned_source_manifest_uses_exact_historical_url_without_live_date_resolution(monkeypatch):
    eli, lang, date = "cc/27/317_321_377", "de", "20260701"
    version_url = fedlex.xml_url(eli, date, lang)
    requested = []

    def no_live_resolution(_eli):
        raise AssertionError("pinned source mode must not call resolve_date")

    class Client:
        def get(self, url):
            requested.append(url)
            return _XMLResponse(HIERARCHY_XML, url)

    monkeypatch.setattr(fedlex, "resolve_date", no_live_resolution)
    monkeypatch.setattr(fedlex, "resolve_law_meta", lambda *_args: (_ for _ in ()).throw(AssertionError("pinned mode must not resolve live metadata")))
    monkeypatch.setattr(fedlex, "H", Client())
    rows = fedlex.load_law(eli, [lang], source_versions={eli: {lang: {
        "status": "success", "source_version": date,
        "source_version_url": version_url, "title_prefix": "Pinned law (PIN; SR 1)"}}})
    assert requested == [version_url]
    assert {row["source_version"] for row in rows} == {date}
    assert {row["source_version_url"] for row in rows} == {version_url}
    assert rows[0]["title"].startswith("Pinned law (PIN; SR 1) —")
    assert rows[0]["dense_text"] == f"{rows[0]['title']}. Fedlex. {rows[0]['text']}"  # channel_split default; R1 surface


def test_pinned_source_manifest_rejects_missing_and_url_mismatch_coverage(tmp_path):
    elis, binding = _source_manifest(tmp_path)
    with pytest.raises(ValueError, match="lacks .*en coverage"):
        fedlex.load_source_version_manifest(binding, elis, ["de", "en"])

    _elis, bad_binding = _source_manifest(tmp_path, bad_url=True)
    with pytest.raises(ValueError, match="URL/version mismatch"):
        fedlex.load_source_version_manifest(bad_binding, elis, ["de"])

    _elis, failed_binding = _source_manifest(tmp_path, failed=True)
    pinned = fedlex.load_source_version_manifest(failed_binding, elis, ["de"])
    assert pinned[elis[0]]["de"]["status"] == "parse_failed"


def test_pinned_prior_records_is_required_and_sha_bound(tmp_path):
    with pytest.raises(ValueError, match="requires --prior-records"):
        fedlex.resolve_prior_records(None, pinned=True)
    path = tmp_path / "prior.jsonl"
    path.write_text("{}\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fedlex.resolve_prior_records(f"{digest}={path}", pinned=True) == path
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fedlex.resolve_prior_records(f"{'0' * 64}={path}", pinned=True)
    assert fedlex.resolve_prior_records(str(path), pinned=False) == path


def test_pinned_mode_deterministically_skips_historical_parse_failed_url(monkeypatch):
    eli, lang, date = "cc/27/317_321_377", "en", "20260701"
    version_url = fedlex.xml_url(eli, date, lang)

    class Client:
        def get(self, url):
            raise AssertionError("historical parse_failed input must not be fetched")

    monkeypatch.setattr(fedlex, "resolve_law_meta", lambda *_args: {})
    monkeypatch.setattr(fedlex, "H", Client())
    rows = fedlex.load_law(eli, [lang], source_versions={eli: {lang: {
        "status": "parse_failed", "error": "historical parser failure",
        "source_version": date, "source_version_url": version_url}}})
    assert rows == []


def test_pinned_fetch_redirect_or_failure_is_fatal(monkeypatch):
    eli, lang, date = "cc/27/317_321_377", "de", "20260701"
    version_url = fedlex.xml_url(eli, date, lang)

    class Client:
        def get(self, _url):
            return _XMLResponse(HIERARCHY_XML, version_url + "?redirected")

    monkeypatch.setattr(fedlex, "resolve_law_meta", lambda *_args: {})
    monkeypatch.setattr(fedlex, "H", Client())
    with pytest.raises(RuntimeError, match="pinned Fedlex source failed.*URL drift"):
        fedlex.load_law(eli, [lang], source_versions={eli: {lang: {
            "status": "success", "source_version": date,
            "source_version_url": version_url}}})


def test_live_mode_keeps_current_date_resolution_when_no_manifest_is_supplied(monkeypatch):
    eli, lang, date = "cc/27/317_321_377", "de", "20260701"
    version_url = fedlex.xml_url(eli, date, lang)

    class Client:
        def get(self, url):
            assert url == version_url
            return _XMLResponse(HIERARCHY_XML, url)

    monkeypatch.setattr(fedlex, "resolve_date", lambda actual_eli: date if actual_eli == eli else None)
    monkeypatch.setattr(fedlex, "resolve_law_meta", lambda *_args: {})
    monkeypatch.setattr(fedlex, "H", Client())
    rows = fedlex.load_law(eli, [lang])
    assert rows and {row["source_version"] for row in rows} == {date}


def test_legacy_representation_reproduces_hierarchy_in_both_channels():
    rows = fedlex.parse_akn(
        HIERARCHY_XML,
        "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "de",
        "cc/27/317_321_377", {"de": {"short": "OR", "sr": "220"}},
        representation="legacy")
    row = rows[0]
    assert "Die Miete" in row["text"]
    assert "dense_text" not in row
    assert "sparse_text" not in row
    args = SimpleNamespace(attribution=None, license_version=None)
    _pid, dense, sparse, _payload = stage.project(
        row, args, {"consent_state": "dump"})
    assert dense == sparse
    assert "Die Miete" in dense


def test_stage_projection_uses_independent_text_channels():
    record = {
        "url": "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_257",
        "anchor": "art_257", "text": "Art. 257 Der Mietzins.",
        "dense_text": "Art. 257 Der Mietzins.",
        "sparse_text": "Achter Titel: Die Miete. Art. 257 Der Mietzins.",
        "snippet": "Der Mietzins.", "title": "OR — Art. 257", "site_name": "Fedlex",
        "lang": "de", "domain": "fedlex.admin.ch", "tier": "A",
        "source_type": "official", "trust_tier": 1.0,
    }
    args = SimpleNamespace(attribution=None, license_version=None)
    _pid, dense, sparse, _payload = stage.project(
        record, args, {"consent_state": "dump"})
    assert "Die Miete" not in dense
    assert "Die Miete" in sparse
    assert "Fedlex" in dense and "Fedlex" in sparse


@pytest.mark.parametrize("lang,notice", [
    ("fr", "Abrogé par le ch. II de la loi fédérale du 25 juin 1971."),
    ("fr", "Abrogée par le ch. II de la loi fédérale du 25 juin 1971."),
    ("it", "Abrogato dal n. II della legge federale del 25 giugno 1971."),
    ("en", "Repealed by No. II of the Federal Act of 25 June 1971."),
])
def test_repeal_marker_is_multilingual(lang, notice):
    xml = AKN_XML.replace(
        b"[Aufgehoben durch Ziff. II Art. 6 Ziff. 1 des BG vom 25. Juni 1971.]",
        ("[" + notice + "]").encode())
    rows = fedlex.parse_akn(
        xml, f"https://www.fedlex.admin.ch/eli/cc/27/317_321_377/{lang}",
        lang, "cc/27/317_321_377",
        {lang: {"short": "CO", "sr": "220"}})
    assert rows[0]["status"] == "repealed"
    assert notice in rows[0]["text"]


def test_r2_preserves_pointer_identity_and_refuses_delete_only_group():
    key = ("cc/27/317_321_377", "de", "art_159")
    url = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_159"
    new = {"stable-id": {"payload": {"url": url, "status": "repealed"},
                         "vector": {"dense": [0.0]}}}
    rows = r2.build_rows(new, {key: ("stable-id", url)}, {key: ["stable-id"]})
    assert rows[0]["points"][0]["payload"]["status"] == "repealed"
    assert rows[0]["old_ids"] == []

    with pytest.raises(RuntimeError, match="no replacement"):
        r2.build_rows({}, {key: ("stable-id", url)}, {})


def test_stage_projection_preserves_repealed_status():
    record = {
        "url": "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_159",
        "anchor": "art_159", "text": "Aufgehoben durch Bundesgesetz.",
        "snippet": "Aufgehoben durch Bundesgesetz.", "title": "OR — Art. 159",
        "lang": "de", "domain": "fedlex.admin.ch", "tier": "A",
        "source_type": "official", "trust_tier": 1.0, "status": "repealed",
    }
    args = SimpleNamespace(attribution=None, license_version=None)
    _pid, _dense, _sparse, payload = stage.project(
        record, args, {"consent_state": "dump"})
    assert payload["status"] == "repealed"


def test_removed_anchor_gets_evidenced_tombstone_without_old_text():
    url = ("https://www.fedlex.admin.ch/eli/cc/1959/679_705_685/"
           "de#disp_u2/art_108")
    prior = [{
        "url": url, "title": "Strassenverkehrsgesetz",
        "snippet": "OLD OPERATIVE TEXT MUST NOT SURVIVE",
        "domain": "fedlex.admin.ch", "source_type": "official", "lang": "de",
    }]
    state = {("cc/1959/679_705_685", "de"): {
        "source_version": "20260701",
        "source_version_url": "https://source.example/current.xml",
        "raw_article_anchors": {"art_108"},
    }}
    rows = fedlex.removed_source_tombstones(prior, [], state)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "removed_from_current_source"
    assert row["source_version"] == "20260701"
    assert row["source_version_url"] == "https://source.example/current.xml"
    assert row["source_anchor_evidence"] == \
        "absent_from_current_akn_article_eids"
    assert row["base_anchor"] == "disp_u2/art_108"
    assert "OLD OPERATIVE TEXT" not in row["text"]


def test_raw_source_anchor_presence_is_parser_failure_not_tombstone():
    url = ("https://www.fedlex.admin.ch/eli/cc/1959/679_705_685/"
           "de#disp_u2/art_108")
    prior = [{
        "url": url, "domain": "fedlex.admin.ch", "source_type": "official",
    }]
    state = {("cc/1959/679_705_685", "de"): {
        "source_version": "20260701",
        "source_version_url": "https://source.example/current.xml",
        "raw_article_anchors": {"disp_u2/art_108"},
    }}
    with pytest.raises(RuntimeError, match="parser omission"):
        fedlex.removed_source_tombstones(prior, [], state)


def test_r2_requires_nonzero_removed_source_tombstones():
    tombstone_rows = [{
        "points": [{"payload": {
            "status": "removed_from_current_source"}}],
        "old_ids": [],
    }]
    assert r2.validate_rows(
        tombstone_rows, require_tombstones=True) == {
            "delete_only_rows": 0,
            "removed_source_tombstone_points": 1,
        }
    with pytest.raises(RuntimeError, match="expected.*tombstones"):
        r2.validate_rows([{"points": [{"payload": {}}]}],
                         require_tombstones=True)


def test_stage_projection_preserves_source_version_evidence():
    record = {
        "url": ("https://www.fedlex.admin.ch/eli/cc/1959/679_705_685/"
                "de#disp_u2/art_108"),
        "anchor": "disp_u2/art_108",
        "text": "Fedlex source-state pointer.",
        "snippet": "Fedlex source-state pointer.",
        "title": "SVG — source pointer", "lang": "de",
        "domain": "fedlex.admin.ch", "tier": "A",
        "source_type": "official", "trust_tier": 1.0,
        "status": "removed_from_current_source",
        "source_version": "20260701",
        "source_version_url": "https://source.example/current.xml",
        "source_anchor_evidence": "absent_from_current_akn_article_eids",
        "base_anchor": "disp_u2/art_108",
    }
    args = SimpleNamespace(attribution=None, license_version=None)
    _pid, _dense, _sparse, payload = stage.project(
        record, args, {"consent_state": "dump"})
    assert payload["status"] == "removed_from_current_source"
    assert payload["source_version"] == "20260701"
    assert payload["source_version_url"] == "https://source.example/current.xml"
    assert payload["source_anchor_evidence"] == \
        "absent_from_current_akn_article_eids"
    assert payload["base_anchor"] == "disp_u2/art_108"
