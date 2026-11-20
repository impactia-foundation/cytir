"""read_url (task 2709): the fedlex twin derivation, the side-table loader, the generator's size check."""
import json
import logging

from cytir.serve import read_url as RU
from cytir.sources import fedlex_versions as FV
from tests.helpers.repo_files import repo_file_or_skip

TABLE = {"schema": "aw.fedlex-versions.v1", "_langs": ["de", "fr", "it"],
         "_dropped": ["cc/24/233_245_233/it"],
         "cc/27/317_321_377": "20261001", "cc/24/233_245_233": "20260701"}
BASE = "https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli"


def _rd(url, table=TABLE):
    return RU.read_url_for({"url": url}, table)


def test_de_with_article_fragment():
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336_c") == (
        f"{BASE}/cc/27/317_321_377/20261001/de/html/"
        "fedlex-data-admin-ch-eli-cc-27-317_321_377-20261001-de-html.html#art_336_c")


def test_fr_and_it_and_no_fragment():
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/fr").endswith(
        "20261001-fr-html.html")
    assert "/20260701/de/html/" in _rd("https://www.fedlex.admin.ch/eli/cc/24/233_245_233/de#art_8")


def test_dropped_pair_unverified_lang_missing_date_unknown_domain_are_none():
    assert _rd("https://www.fedlex.admin.ch/eli/cc/24/233_245_233/it#art_8") is None   # dropped
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en") is None          # not verified
    assert _rd("https://www.fedlex.admin.ch/eli/cc/1999/404/de") is None                # no date
    assert _rd("https://example.org/eli/cc/27/317_321_377/de") is None
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377") is None
    assert RU.read_url_for({"title": "x"}, TABLE) is None
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", None) is None


def test_attach_is_additive(monkeypatch):
    monkeypatch.setattr(RU, "_table", TABLE)
    rows = [{"url": "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_1"},
            {"url": "https://en.wikipedia.org/wiki/Matterhorn"}]
    RU.attach_read_urls(rows)
    assert "read_url" in rows[0] and "read_url" not in rows[1]


def test_missing_table_disables_with_one_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="aw_server.read_url"):
        assert RU.load_table(tmp_path / "absent.json") is None
    assert len([r for r in caplog.records if "disabled" in r.message]) == 1


def test_bad_table_disables(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("[1]")
    assert RU.load_table(p) is None


def test_shipped_table_shape():
    path = repo_file_or_skip("data/fedlex-versions.v1.json")
    doc = json.loads(path.read_text())
    assert doc["schema"] == "aw.fedlex-versions.v1"
    laws = {k: v for k, v in doc.items() if "/" in k}
    assert laws and all(len(v) == 8 and v.isdigit() for v in laws.values())


def test_generator_content_check_shell_dropped_short_law_kept():
    shell = b"<html><body><app-root></app-root>" + b"x" * 9000
    short = b'<div id="lawcontent"><div id="art_1">Art. 1</div></div>'

    def get(url):
        return (200, shell if "/fr/" in url else short)
    dates = {"cc/a/1": "20260101", "cc/b/2": None}
    out = FV.build_table(["cc/a/1", "cc/b/2"], resolve=dates.get, get=get,
                         sleep=lambda s: None, log=lambda m: None, built_at="T")
    assert out["cc/a/1"] == "20260101" and "cc/b/2" not in out
    assert out["_dropped"] == ["cc/a/1/fr"] and out["_verified"] == 3


def test_big_body_without_anchor_is_dropped():
    assert not FV.is_real_twin(200, b"y" * 200000)


def test_generator_drops_non_200_and_all_failed_eli():
    out = FV.build_table(["cc/a/1"], resolve=lambda e: "20260101",
                         get=lambda u: (404, b""), sleep=lambda s: None, log=lambda m: None)
    assert "cc/a/1" not in out and len(out["_dropped"]) == 4


EN_TABLE = {**TABLE, "_langs": ["de", "fr", "it", "en"], "_dropped": ["cc/24/233_245_233/en"]}


def test_en_gets_read_url_and_other_langs_unchanged():
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en#art_336_c", EN_TABLE) == (
        f"{BASE}/cc/27/317_321_377/20261001/en/html/"
        "fedlex-data-admin-ch-eli-cc-27-317_321_377-20261001-en-html.html#art_336_c")
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", EN_TABLE).endswith("-de-html.html")


def test_en_absent_from_langs_or_dropped_or_rm_is_none():
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en") is None  # old 3-lang table
    assert _rd("https://www.fedlex.admin.ch/eli/cc/24/233_245_233/en", EN_TABLE) is None  # no EN twin
    assert _rd("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/rm", EN_TABLE) is None


def test_shipped_table_declares_en_and_drops_shell_acts():
    p = repo_file_or_skip("data/fedlex-versions.v1.json")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert "en" in doc["_langs"]
    assert all(x.endswith("/en") for x in doc["_dropped"])
