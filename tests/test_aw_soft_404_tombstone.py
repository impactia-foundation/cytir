"""`soft_404`, the third PURE, export-backed index-text tombstone reason (task 1363, ADR-123 am.1).
The rows are the shape of the federal error pages the served index holds (title + short snippet of
one error template); the detector is the committed `cytir.sources.soft404` file."""
from __future__ import annotations

import json

import pytest

from cytir.distribute import pack
from cytir.index import point_delete as PD
from cytir.index import promote_queue as Q
from cytir.ops import index_text_tombstones as ITT
from cytir.sources import soft404

S4 = Q.SOFT_404
ERR_TITLE = "Ein Fehler ist aufgetreten"
ERR_SNIP = "Ein Fehler ist aufgetreten. Die Seite konnte nicht angezeigt werden."
REAL_TITLE = "Art. 336 OR - Missbräuchliche Kündigung"
REAL_SNIP = "Die Kündigung ist missbräuchlich, wenn eine Partei sie ausspricht aus einem der folgenden Gründe."


def row(i, url, title, snippet, **kw):
    return {"id": f"00000000-0000-5000-8000-{i:012d}",
            "payload": {"url": url, "title": title, "snippet": snippet, "provenance": "web",
                        "source_type": "official", "domain": "elcom.admin.ch",
                        "content_digest": f"d{i}", **kw}}


ERR1 = row(1, "https://www.elcom.admin.ch/de/500", ERR_TITLE, ERR_SNIP)
ERR2 = row(2, "https://www.elcom.admin.ch/de/404", ERR_TITLE, ERR_SNIP)
REAL = row(3, "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336", REAL_TITLE, REAL_SNIP)
CARD = row(4, "https://www.elcom.admin.ch/de/500x", ERR_TITLE, ERR_SNIP, source_type="entity_card",
           entity_kind="organization", qid="Q1")


MDN = row(5, "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
          "404 Not Found - HTTP | MDN", "The HTTP 404 Not Found response status code says the server cannot find the resource.")
WP = row(6, "https://example.org/blog/fix-page-not-found",
         "How to fix the 'Page not found' error in WordPress", "Permalinks often cause this error; resave them.")
NOTICE = row(7, "https://www.admin.ch/gov/it/start/ordinanza.html", "Ordinanza sul personale federale",
             "La pagina richiesta non è disponibile in italiano. Ordinanza del 3 luglio 2001.")
BAK = row(8, "https://www.bak.admin.ch/bak/it/home/pagina-defecta.html", "Pagina non trovata",
          "La pagina richiesta non è disponibile in italiano.")
BOTH = row(9, "https://www.admin.ch/gov/it/start/xyz.html", "Pagina non trovata",
           "La pagina richiesta non è disponibile in italiano.")


class FakeTwin:
    def __init__(self, rows, page=2):
        self.rows = {r["id"]: dict(r["payload"]) for r in rows}
        self.page = page
        self.scroll_bodies = []

    def api(self, url, path, body=None, method=None):
        if path.endswith("/points/scroll"):
            self.scroll_bodies.append(body)
            ids = list(self.rows)
            start = ids.index(body["offset"]) if body.get("offset") else 0
            chunk = ids[start:start + self.page]
            nxt = ids[start + self.page] if start + self.page < len(ids) else None
            keep = body["with_payload"]
            return {"points": [{"id": i, "payload": {k: v for k, v in self.rows[i].items() if k in keep}}
                               for i in chunk], "next_page_offset": nxt}
        if body is None:                                   # the collection info (points_count)
            return {"points_count": len(self.rows)}
        keep = body["with_payload"]
        return [{"id": i, "payload": (dict(self.rows[i]) if keep is True else
                                      {k: v for k, v in self.rows[i].items() if keep and k in keep})}
                for i in body["ids"] if i in self.rows]


@pytest.fixture
def twin(monkeypatch):
    t = FakeTwin([ERR1, ERR2, REAL, CARD])
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    return t


def tomb(r, reason=S4):
    return {"id": r["id"], "url": r["payload"]["url"], "reason": reason}


def validate(tombs):
    return Q.validate_tombstones(tombs, pack_urls=set(), target_url="http://twin",
                                 collection="aw_e10_twin", export_ids={t["id"] for t in tombs})


def test_the_fixture_rows_are_what_the_detector_says():
    assert soft404.detect(ERR_TITLE, ERR_SNIP) is not None
    assert soft404.detect(REAL_TITLE, REAL_SNIP) is None


def test_the_reason_is_export_backed_queue_only_and_never_a_takedown():
    assert S4 in Q.TOMBSTONE_REASONS and S4 in Q.EXPORT_BACKED_REASONS
    assert S4 not in pack.TOMBSTONE_REASONS and S4 not in pack.TAKEDOWN_REASONS


def test_matching_stored_rows_pass(twin):
    assert validate([tomb(ERR1), tomb(ERR2)])["by_reason"] == {S4: 2}


def test_a_row_the_detector_does_not_match_is_refused_and_named(twin):
    with pytest.raises(Q.QueueError, match=REAL["id"] + ".*does not match the soft-404 detector"):
        validate([tomb(ERR1), tomb(REAL)])


def test_the_rule_reads_the_stored_row_not_the_pack(twin):
    t = dict(tomb(REAL), title=ERR_TITLE, snippet=ERR_SNIP, detail="soft-404 page")
    with pytest.raises(Q.QueueError, match="does not match"):
        validate([t])


def test_the_title_signal_alone_is_not_enough_a_page_that_discusses_errors_is_refused(twin):
    for r in (MDN, WP, NOTICE):
        twin.rows[r["id"]] = dict(r["payload"])
        assert soft404.detect(r["payload"]["title"], r["payload"]["snippet"]) is not None   # gate 1 fires
        with pytest.raises(Q.QueueError, match="only discusses errors"):
            validate([tomb(r)])


def test_an_error_path_or_title_and_snippet_each_matching_pass_the_second_gate(twin):
    for r in (BAK, BOTH):
        twin.rows[r["id"]] = dict(r["payload"])
    assert validate([tomb(BAK), tomb(BOTH)])["by_reason"] == {S4: 2}
    assert soft404.error_path(BAK["payload"]["url"]) and not soft404.error_path(BOTH["payload"]["url"])
    assert not soft404.error_path(MDN["payload"]["url"])               # depth keeps the docs path out


def test_plan_names_the_rows_the_second_gate_refuses(twin):
    for r in (MDN, WP, NOTICE):
        twin.rows[r["id"]] = dict(r["payload"])
    res = ITT.classify(S4, [{"id": r["id"]} for r in (ERR1, MDN, WP, NOTICE)], "http://twin", "aw_e10_twin")
    assert [t["id"] for t in res["pass"]] == [ERR1["id"]]
    assert set(res["failed"]) == {MDN["id"], WP["id"], NOTICE["id"]}


def test_a_scroll_that_returns_nothing_is_an_error_not_the_end(twin, tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "_api", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="under-reports"):
        ITT.census(S4, "http://twin", "aw_e10_twin", tmp_path / "x")


def test_a_card_is_refused(twin):
    with pytest.raises(Q.QueueError, match="entity card"):
        validate([tomb(CARD)])


def test_a_url_that_is_not_the_stored_document_is_refused(twin):
    t = dict(tomb(ERR1), url="https://www.elcom.admin.ch/de/other")
    with pytest.raises(Q.QueueError, match="not the tombstone's url"):
        validate([t])


def test_a_pack_mixing_reasons_is_refused():
    Q.check_superseded_generation_pack({}, [], [tomb(ERR1), tomb(ERR2)])
    with pytest.raises(Q.QueueError, match="only that reason"):
        Q.check_superseded_generation_pack({}, [], [tomb(ERR1), tomb(ERR2, Q.BINARY_CONTENT)])
    with pytest.raises(Q.QueueError, match="carries no points"):
        Q.check_superseded_generation_pack({}, [{"id": "p"}], [tomb(ERR1)])


def test_the_census_yields_exactly_the_matching_rows(twin, tmp_path):
    out = tmp_path / "cand.jsonl"
    res = ITT.census(S4, "http://twin", "aw_e10_twin", out)
    got = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [g["id"] for g in got] == [ERR1["id"], ERR2["id"]]          # the real row and the card: out
    assert set(got[0]) == {"id", "url", "title", "snippet", "match", "error_path"} and got[0]["match"]
    assert res["scanned"] == 4 and res["candidates"] == 2 and res["points_count"] == 4
    assert len(res["detector_sha256"]) == 64
    assert len(twin.scroll_bodies) == 2                                  # paged, two rows a page
    assert set(twin.scroll_bodies[0]["with_payload"]) == set(ITT.CENSUS_FIELDS)
    assert twin.scroll_bodies[0]["with_vector"] is False


def test_the_census_is_for_soft_404_only(twin, tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        ITT.census(Q.BINARY_CONTENT, "http://twin", "aw_e10_twin", tmp_path / "x")


def test_plan_passes_the_matching_rows_and_names_the_refused(twin):
    res = ITT.classify(S4, [{"id": r["id"]} for r in (ERR1, ERR2, REAL, CARD)], "http://twin", "aw_e10_twin")
    assert [t["id"] for t in res["pass"]] == [ERR1["id"], ERR2["id"]]
    assert set(res["failed"]) == {REAL["id"], CARD["id"]}
    s = ITT.summary(S4, res, 4)
    assert s["tombstones"] == 2 and s["refused_by_the_drains_rule"] == 2 and s["twin_count_delta"] == -2


def test_compose_accepts_the_reason_and_records_the_detector_sha(twin, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ITT.UGS, "write_export",
                        lambda *a, **k: {"path": "p", "sha256": "s", "rows": 2})

    def fake_pack(queue_dir, pack_id, part, **kw):
        seen.update(kw, ids=[t["id"] for t in part])
        return {"pack_id": pack_id, "tombstones_count": len(part), "tombstones_by_reason": {S4: len(part)},
                "export": kw["extra"]["export"]}
    monkeypatch.setattr(ITT.ST, "write_pack", fake_pack)
    out = ITT.compose(S4, [{"id": ERR1["id"]}, {"id": ERR2["id"]}, {"id": REAL["id"]}], "http://twin",
                      "aw_e10_twin", export_dir=str(tmp_path), queue_dir=str(tmp_path), pack_prefix="t1363",
                      max_ids=5000, note="task 1363 soft_404 (ruling 61 Q2)")
    assert out[0]["tombstones_by_reason"] == {S4: 2}                      # the real row never enters
    assert seen["reason"] == S4 and seen["task"] == 1363 and seen["ids"] == [ERR1["id"], ERR2["id"]]
    assert "soft-404-signals.v1.json sha256=" + ITT.detector_sha256() in seen["note"]
